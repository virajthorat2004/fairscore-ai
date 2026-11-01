"""
FAIRScore 3.0 learning loop: keep the AI engine honest when it is used on a population it was not trained on.

The AI engine was trained on 679 Czech loans (Berka). A lender in India will see different applicants (covariate
shift) and a different default rate (label / prior shift), and repayment outcomes arrive only 6-12 months after a
loan is disbursed. This module adapts the engine in two stages and never changes the live engine on its own:

  Day 0 (applications, no outcomes yet)
    * Drift: PSI per model input, live applicants vs the training applicants.
    * Expected default rate, two label-free ways: the mean PD (right if only the applicants changed: covariate
      shift) and the EM estimate of Saerens, Latinne & Decaestecker (2002) (right if only the default rate changed:
      prior shift). Outcomes are needed to tell the two apart; src/shift_study.py shows the EM correction is badly
      wrong under covariate shift, so neither is applied to PDs: a gap between them raises an alarm.
    * Shift-aware uncertainty: weighted Mondrian conformal prediction (Tibshirani et al., 2019). A classifier
      that tells training applicants from live applicants gives each training case a weight
      w(x) = p_live(x) / p_train(x); the conformal p-values are computed on the re-weighted training cases, so the
      "at most alpha of would-be defaulters auto-approved" guarantee holds for the live population under
      covariate shift. (Mondrian = class-conditional: it is already valid under pure prior shift, because p(x | y)
      does not change.)

  Day N (outcomes recorded)
    * Bayesian recalibration: PD' = sigmoid(a + b * logit(PD)), with a Gaussian prior centred on "no change"
      (a = 0, b = 1). With few local outcomes the prior dominates and the engine is barely moved; with many it
      follows the data. Plain Platt scaling (no prior) is unstable below ~200 outcomes.
    * Learning offers (src/starter_ladder.py) carry their propensity; outcomes are weighted by 1 / propensity so the
      recalibration sees the whole applicant population, not only the approved part (inverse-propensity weighting).
    * Champion / challenger: the recalibrated engine (out-of-fold, so it is not judged on the outcomes it was fitted
      to) and, once enough defaults are recorded, a retrained challenger (Berka + local loans) are compared with the
      current engine with bootstrap intervals. The model-risk team decides; nothing is promoted automatically.

The functions are pure (arrays in, arrays out) so the same code runs in the research study
(src/shift_study.py) and on a lender's recorded outcomes (`python src/learning_loop.py`).
"""
import json
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-4
IPW_CAP = 5.0     # inverse-propensity weights are capped (src/exploration_study.py: uncapped 1 / 0.05 = 20 adds variance)


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


# ----------------------------------------------------------------------------------------------- calibration
def bayes_recalibrate(pd_, y, prior_sd=(0.5, 0.25), prior_mean=(0.0, 1.0), weights=None):
    """MAP estimate of (a, b) in PD' = sigmoid(a + b * logit(PD)) with independent Gaussian priors.
    prior_sd=(inf, inf) gives ordinary Platt scaling. `weights` (e.g. 1 / propensity) weight each outcome's
    likelihood. Returns (a, b)."""
    z, y = logit(pd_), np.asarray(y, float)
    w = np.ones(len(z)) if weights is None else np.asarray(weights, float)
    m, s = np.asarray(prior_mean, float), np.asarray(prior_sd, float)

    def nll(theta):
        eta = theta[0] + theta[1] * z
        val = np.sum(w * (np.logaddexp(0, eta) - y * eta))
        pen = np.sum(np.where(np.isfinite(s), (theta - m) ** 2 / (2 * s ** 2), 0.0))
        return val + pen

    def grad(theta):
        eta = theta[0] + theta[1] * z
        r = w * (sigmoid(eta) - y)
        g = np.array([r.sum(), (r * z).sum()])
        return g + np.where(np.isfinite(s), (theta - m) / s ** 2, 0.0)

    res = minimize(nll, m.copy(), jac=grad, method="L-BFGS-B", bounds=[(-6, 6), (0.05, 5)])
    return float(res.x[0]), float(res.x[1])


def apply_recalibration(pd_, a, b):
    return sigmoid(a + b * logit(pd_))


def em_prior_shift(pd_unlabelled, prior_source, iters=200, tol=1e-7):
    """Saerens et al. (2002): new default rate from unlabelled, calibrated source PDs (label-shift assumption).
    Returns (prior_target, intercept_shift) where intercept_shift is the log-odds change to add to logit(PD)."""
    p = np.clip(np.asarray(pd_unlabelled, float), EPS, 1 - EPS)
    pi_s, pi_t = float(prior_source), float(np.mean(p))
    for _ in range(iters):
        r1, r0 = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
        post = r1 * p / (r1 * p + r0 * (1 - p))
        new = float(np.mean(post))
        if abs(new - pi_t) < tol:
            pi_t = new
            break
        pi_t = new
    pi_t = float(np.clip(pi_t, 1e-3, 0.95))
    return pi_t, float(logit(pi_t) - logit(pi_s))


# ------------------------------------------------------------------------------------------------- density ratio
class DensityRatio:
    """Probabilistic classifier training-vs-live; w(x) = p_live(x) / p_train(x). Picklable.

    * inputs are sign-log transformed (the ratios are skewed) so a linear classifier is not dominated by outliers;
    * the training (calibration) cases get CROSS-FITTED weights: each is weighted by a classifier that did not see
      it, otherwise the classifier memorises them as "training" and their weights are biased low;
    * weights are clipped and scaled to mean 1 on the training cases: with few calibration defaulters one extreme
      weight would otherwise dominate a p-value (the known fragility of weighted conformal prediction);
    * `shift_detected`: the cross-validated AUC of the classifier is more than 2 standard errors above 0.5. When no
      shift can be detected, weighting only adds noise and the standard conformal p-values are used."""

    def __init__(self, X_source, X_target, clip=(0.2, 5.0), C=0.1, folds=5, seed=0, groups_target=None):
        """groups_target: optional customer / loan ids of the live cases, so repeat applications of one customer
        never sit on both sides of a cross-validation split (that would fake a detectable shift)."""
        from sklearn.model_selection import StratifiedGroupKFold
        Xs, Xt = _prep(X_source), _prep(X_target)
        Z = np.vstack([Xs, Xt])
        t = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
        self.clip, self.n_source, self.n_target = clip, len(Xs), len(Xt)

        def model():
            return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
        q_cv = np.zeros(len(t))
        gt = np.arange(len(Xt)) if groups_target is None else pd.factorize(np.asarray(groups_target))[0]
        groups = np.r_[-1 - np.arange(len(Xs)), gt]
        for tr_, te_ in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(Z, t, groups):
            q_cv[te_] = model().fit(Z[tr_], t[tr_]).predict_proba(Z[te_])[:, 1]
        self.clf = model().fit(Z, t)
        self.separability_auc = float(roc_auc_score(t, q_cv))           # 0.5 = no visible shift
        a, n1, n2 = self.separability_auc, len(np.unique(gt)), len(Xs)    # Hanley & McNeil (1982) standard error
        q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
        self.auc_se = float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2)))
        self.shift_detected = bool(self.separability_auc > 0.5 + 2 * self.auc_se)
        raw = self._odds(q_cv[: len(Xs)])
        self.scale = 1.0 / np.mean(raw)
        self._source_w = np.clip(raw * self.scale, *clip)

    @staticmethod
    def _odds(q):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return q / (1 - q)                     # class_weight='balanced' already removes the n_t / n_s factor

    def source_weights(self):
        """Cross-fitted weights of the training (calibration) cases, in their original order."""
        return self._source_w.copy()

    def __call__(self, X):
        return np.clip(self._odds(self.clf.predict_proba(_prep(X))[:, 1]) * self.scale, *self.clip)


def fit_density_ratio(X_source, X_target, **kw):
    return DensityRatio(X_source, X_target, **kw)


def shift_aware_pvalues(pd_cal, y_cal, pd_test, X_test, density):
    """FAIRScore 3.0 conformal p-values: weighted Mondrian when a covariate shift is detected, standard otherwise."""
    if density is None or not density.shift_detected:
        return mondrian_pvalues(pd_cal, y_cal, pd_test)
    return mondrian_pvalues(pd_cal, y_cal, pd_test, density.source_weights(), density(X_test))


def _prep(X):
    X = np.asarray(X, float)
    return np.sign(X) * np.log1p(np.abs(X))


def effective_sample_size(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if len(w) else 0.0


# ------------------------------------------------------------------------------------------------ conformal
def mondrian_pvalues(pd_cal, y_cal, pd_test, w_cal=None, w_test=None):
    """(Weighted) Mondrian conformal p-values for 'repays' (p0) and 'defaults' (p1).
    Non-conformity: PD for a repayer, 1 - PD for a defaulter (same as src/ai_engine.py). With all weights equal
    this is exactly the engine's unweighted p-value. Vectorised over test cases."""
    pd_cal, y_cal, pd_test = np.asarray(pd_cal, float), np.asarray(y_cal, int), np.asarray(pd_test, float)
    w_cal = np.ones(len(pd_cal)) if w_cal is None else np.asarray(w_cal, float)
    w_test = np.ones(len(pd_test)) if w_test is None else np.asarray(w_test, float)
    out = []
    for cls, score_cal, score_test in ((0, pd_cal, pd_test), (1, 1 - pd_cal, 1 - pd_test)):
        m = y_cal == cls
        s, ws = score_cal[m], w_cal[m]
        o = np.argsort(s)
        s, ws = s[o], ws[o]
        tail = np.r_[np.cumsum(ws[::-1])[::-1], 0.0]          # sum of weights with score >= s[k]
        k = np.searchsorted(s, score_test, side="left")
        out.append((tail[k] + w_test) / (ws.sum() + w_test))
    return out[0], out[1]


def outcome_sets(p0, p1, alpha):
    in0, in1 = np.asarray(p0) > alpha, np.asarray(p1) > alpha
    return np.where(in0 & ~in1, "repays", np.where(in1 & ~in0, "defaults", "uncertain"))


# ----------------------------------------------------------------------------------------------- monitoring
def psi(expected, actual, bins=10):
    from monitoring import psi as _psi
    return _psi(expected, actual, bins)


def drift_table(X_train, X_live, names):
    rows = []
    for j, n in enumerate(names):
        v = psi(X_train[:, j], X_live[:, j])
        rows.append({"input": n, "psi": round(v, 3),
                     "status": "stable" if v < 0.10 else "watch" if v < 0.25 else "major shift",
                     "train_median": float(np.median(X_train[:, j])), "live_median": float(np.median(X_live[:, j]))})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) < bins:
        return float(abs(p.mean() - y.mean()))
    q = pd.qcut(p, bins, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True)
    return float((g.size() / len(p) * (g["p"].mean() - g["y"].mean()).abs()).sum())


def champion_challenger(y, pd_champion, pd_challenger, n_boot=1000, seed=0):
    """Bootstrap comparison on recorded outcomes. Positive 'brier_gain' = challenger better."""
    y = np.asarray(y, int)
    rng = np.random.default_rng(seed)
    gains, auc_d = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        gains.append(brier_score_loss(y[i], pd_champion[i]) - brier_score_loss(y[i], pd_challenger[i]))
        auc_d.append(roc_auc_score(y[i], pd_challenger[i]) - roc_auc_score(y[i], pd_champion[i]))
    g, a = np.array(gains), np.array(auc_d)
    return {"brier_champion": brier_score_loss(y, pd_champion), "brier_challenger": brier_score_loss(y, pd_challenger),
            "ece_champion": ece(pd_champion, y), "ece_challenger": ece(pd_challenger, y),
            "brier_gain": float(g.mean()), "brier_gain_ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
            "share_resamples_challenger_better": float(np.mean(g > 0)),
            "auc_change": float(a.mean()), "auc_change_ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}


# ------------------------------------------------------------------------------------------- adaptation object
def build_adaptation(bundle, X_live=None, pd_labelled=None, y_labelled=None, prior_sd=(0.5, 0.25), min_live=30):
    """Everything the live engine needs to adapt to the lender's population. Pure: the caller decides whether to
    save it. Day-0 parts need only live applications; recalibration needs outcomes."""
    out = {"n_live": 0 if X_live is None else int(len(X_live)), "n_outcomes": 0 if y_labelled is None else int(len(y_labelled)),
           "prior_source": float(np.mean(bundle["oof"]["y"])), "a": 0.0, "b": 1.0,
           "density": None, "notes": []}
    if X_live is not None and len(X_live) >= min_live:
        import ai_engine as ae
        pd_live = ae.predict_pd(bundle, X_live)
        pi_t, _ = em_prior_shift(pd_live, out["prior_source"])
        # two label-free estimates of the live default rate: under covariate shift (p(y|x) unchanged) it is the
        # mean PD; under pure prior shift it is the EM estimate. They cannot be told apart without outcomes, and
        # the EM correction is badly wrong under covariate shift (src/shift_study.py), so neither changes a PD:
        # a large gap between them is an ALARM that asks for outcomes to be recorded and reviewed.
        out.update({"default_rate_if_covariate_shift": float(np.mean(pd_live)), "default_rate_if_prior_shift": pi_t,
                    "prior_alarm": bool(abs(logit(pi_t) - logit(np.mean(pd_live))) > 0.5)})
        if "cal_X" in bundle:
            out["density"] = fit_density_ratio(bundle["cal_X"], X_live)
            out["separability_auc"] = out["density"].separability_auc
        out["notes"].append(f"{len(X_live)} unlabelled applications: expected default rate {np.mean(pd_live):.1%} "
                            f"(mean PD), {pi_t:.1%} if only the default rate changed (training {out['prior_source']:.1%})")
    if y_labelled is not None and len(y_labelled) >= 20 and np.asarray(y_labelled).sum() >= 2:
        a, b = bayes_recalibrate(pd_labelled, y_labelled, prior_sd=prior_sd)
        out.update({"a": a, "b": b})
        out["notes"].append(f"recalibrated on {len(y_labelled)} recorded outcomes ({int(np.sum(y_labelled))} defaults)")
    return out


def cv_recalibrated(pd_, y, weights=None, prior_sd=(0.5, 0.25), folds=5, seed=0):
    """Out-of-fold recalibrated PDs: each outcome is predicted by a recalibration fitted WITHOUT it."""
    from sklearn.model_selection import StratifiedKFold
    pd_, y = np.asarray(pd_, float), np.asarray(y, int)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    out = np.zeros(len(y))
    k = max(2, min(folds, int(y.sum()), int((1 - y).sum())))
    for tr_, te_ in StratifiedKFold(k, shuffle=True, random_state=seed).split(pd_, y):
        a, b = bayes_recalibrate(pd_[tr_], y[tr_], prior_sd=prior_sd, weights=w[tr_])
        out[te_] = apply_recalibration(pd_[te_], a, b)
    return out


def adapted_pd(pd_, adaptation):
    if not adaptation:
        return np.asarray(pd_, float)
    return apply_recalibration(pd_, adaptation["a"], adaptation["b"])


# -------------------------------------------------------------------------------------------- lender's data
def _recorded(path=None):
    """Live applications and recorded outcomes from the audit store (AI inputs are stored per assessment).
    Returns X (all applications), lab (has an outcome), y (1 = defaulted), w (1 / propensity)."""
    import audit_store
    h = audit_store.history(path)
    if "ai_inputs" not in h:
        return None, None, None, None
    h = h[h["ai_inputs"].notna()]
    X = np.array([json.loads(v) for v in h["ai_inputs"]], float) if len(h) else np.zeros((0, 24))
    bad = h["outcome"].isin({"defaulted 90+ days", "written off"})
    good = h["outcome"].isin({"repaid", "repaying"})
    lab = (bad | good).values
    prop = pd.to_numeric(h.get("propensity"), errors="coerce").fillna(1.0).clip(lower=0.01).values
    return X, lab, bad.values.astype(int), np.minimum(1.0 / prop, IPW_CAP)


def challenger(bundle, X_local, y_local, w_local, local_weight=3.0, folds=5):
    """Retrained challenger: the engine's recipe on Berka + the lender's loans (local loans up-weighted, learning
    offers by 1 / propensity). Returns (out-of-fold PDs of the local loans, model fitted on everything)."""
    import ai_engine as ae
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold
    Xs, ys = bundle["cal_X"], bundle["oof"]["y"]

    def fit(Xl, yl, wl):
        m = CatBoostClassifier(**ae.CATBOOST_PARAMS, monotone_constraints=ae.MONOTONE)
        return m.fit(np.vstack([Xs, Xl]), np.r_[ys, yl], sample_weight=np.r_[np.ones(len(ys)), local_weight * wl])
    oof = np.zeros(len(y_local))
    for tr_, te_ in StratifiedKFold(folds, shuffle=True, random_state=0).split(X_local, y_local):
        oof[te_] = fit(X_local[tr_], y_local[tr_], w_local[tr_]).predict_proba(X_local[te_])[:, 1]
    return oof, fit(X_local, y_local, w_local)


def main(path=None, out_dir=None):
    """Lender-side run: drift, day-0 checks and (when outcomes exist) recalibration + champion/challenger.
    Writes results/learning_loop_report.json and models/ai_engine_adaptation.pkl (a PROPOSAL: the live engine
    uses it only after the model-risk team sets `use_adaptation = true` in config/policy.toml [learning])."""
    import ai_engine as ae
    from pathlib import Path
    from config import PROJECT_ROOT
    from policy_config import POLICY
    L = POLICY["learning"]
    models_dir = Path(out_dir) if out_dir else PROJECT_ROOT / "models"
    results_dir = Path(out_dir) if out_dir else PROJECT_ROOT / "results"
    bundle = ae.load()
    X, lab, y, w = _recorded(path)
    report = {"applications": 0 if X is None else int(len(X)), "outcomes": 0 if lab is None else int(lab.sum()),
              "defaults": 0 if lab is None else int(y[lab].sum())}
    if X is None or len(X) < L["min_live_applications"]:
        report["status"] = (f"Fewer than {L['min_live_applications']} assessments with AI inputs recorded: "
                            f"nothing to adapt yet.")
        print(report["status"])
    else:
        if bundle.get("cal_X") is not None:
            report["drift"] = drift_table(bundle["cal_X"], X, ae.FEATURES).to_dict("records")
        pd_all = ae.predict_pd(bundle, X)
        n_lab = int(lab.sum())
        ad = build_adaptation(bundle, X, pd_all[lab] if n_lab else None, y[lab] if n_lab else None,
                              prior_sd=L["prior_sd"], min_live=L["min_live_applications"])
        if n_lab >= 20 and y[lab].sum() >= 2:                 # weighted refit (learning offers count 1 / propensity)
            ad["a"], ad["b"] = bayes_recalibrate(pd_all[lab], y[lab], prior_sd=L["prior_sd"], weights=w[lab])
        report.update({k: v for k, v in ad.items() if k != "density"})
        if ad.get("density") is not None:
            report["shift_detected"] = ad["density"].shift_detected
        if n_lab >= 50 and y[lab].sum() >= 5:
            oof = cv_recalibrated(pd_all[lab], y[lab], w[lab], prior_sd=L["prior_sd"])
            report["recalibration_vs_current"] = champion_challenger(y[lab], pd_all[lab], oof)
        if n_lab >= L["challenger_min_outcomes"] and y[lab].sum() >= L["challenger_min_defaults"] and "cal_X" in bundle:
            oof_c, model = challenger(bundle, X[lab], y[lab], w[lab])
            report["retrained_challenger_vs_current"] = champion_challenger(y[lab], pd_all[lab], oof_c)
            with open(models_dir / "ai_engine_challenger_model.pkl", "wb") as fh:
                pickle.dump(model, fh)
        path_ad = models_dir / "ai_engine_adaptation.pkl"
        with open(path_ad, "wb") as fh:
            pickle.dump(ad, fh)
        report["proposal"] = str(path_ad)
        print(json.dumps({k: v for k, v in report.items() if k != "drift"}, indent=2, default=str))
    out = results_dir / "learning_loop_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    main()
