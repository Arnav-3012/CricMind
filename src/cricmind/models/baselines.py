"""Phase 2 baselines on val: global -> phase -> phase x wickets -> logistic -> LightGBM, no player identity.

uv run python -m cricmind.models.baselines      # writes reports/p2/baselines.json

Tuning (D-032): C and the tree count are chosen by fitting on train years before TUNE_HOLDOUT_YEAR and scoring
that year; the final models are refit on all of train. Val is scored once and never influences a choice.
The test seasons are never loaded.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cricmind import config
from cricmind.eval.metrics import (
    bootstrap_ci,
    brier,
    ece,
    log_loss,
    paired_bootstrap_diff,
    row_log_loss,
)
from cricmind.features.context import CATEGORICAL, FEATURES, NUMERIC, PHASES, build_context, competition_vocab
from cricmind.features.splits import load_deliveries

K = len(config.OUTCOME_CLASSES)
# lean: a fixed additive guard so no class gets p = 0 in a thin cell. The real prior-strength fit is
# D-010's job in Phase 5; with ~1.6M train rows these cells hold thousands of balls and the guard is inert.
ALPHA = 1.0


def _codes(series: pd.Series, categories: list[str]) -> np.ndarray:
    """Category id per row, -1 for missing or unseen."""
    return pd.Categorical(series, categories=categories).codes.astype("int64")  # codes are int8 by default


class GlobalFreq:
    name = "global_freq"

    def fit(self, train: pd.DataFrame) -> "GlobalFreq":
        counts = np.bincount(train["y"], minlength=K)
        self.p = counts / counts.sum()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.tile(self.p, (len(df), 1))


class PhaseFreq:
    """Class shares by phase. Rows without a phase (reduced matches, D-029) get the parent's probabilities."""

    name = "phase_freq"

    def __init__(self, parent: GlobalFreq):
        self.parent = parent

    def fit(self, train: pd.DataFrame) -> "PhaseFreq":
        self.parent.fit(train)
        codes = _codes(train["phase"], PHASES)
        ok = codes >= 0
        counts = np.bincount(codes[ok] * K + train["y"].to_numpy()[ok], minlength=len(PHASES) * K)
        counts = counts.reshape(len(PHASES), K)
        self.table = (counts + ALPHA * self.parent.p) / (counts.sum(axis=1, keepdims=True) + ALPHA)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        out = self.parent.predict_proba(df)
        codes = _codes(df["phase"], PHASES)
        ok = codes >= 0
        out[ok] = self.table[codes[ok]]
        return out


class PhaseWicketsFreq:
    """Class shares by phase x wickets band; masked rows fall back to PhaseFreq (and through it to global)."""

    name = "phase_wickets_freq"

    def __init__(self, parent: PhaseFreq):
        self.parent = parent

    @staticmethod
    def _band(df: pd.DataFrame) -> np.ndarray:
        return np.digitize(df["wickets_before"].to_numpy(), config.WICKET_BAND_EDGES)

    def fit(self, train: pd.DataFrame) -> "PhaseWicketsFreq":
        self.parent.fit(train)
        n_bands = len(config.WICKET_BAND_EDGES) + 1
        codes = _codes(train["phase"], PHASES)
        ok = codes >= 0
        cell = codes[ok] * n_bands + self._band(train)[ok]
        counts = np.bincount(cell * K + train["y"].to_numpy()[ok], minlength=len(PHASES) * n_bands * K)
        counts = counts.reshape(len(PHASES), n_bands, K)
        prior = self.parent.table[:, None, :]  # each cell shrinks toward its own phase's shares
        self.table = (counts + ALPHA * prior) / (counts.sum(axis=2, keepdims=True) + ALPHA)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        out = self.parent.predict_proba(df)
        codes = _codes(df["phase"], PHASES)
        ok = codes >= 0
        out[ok] = self.table[codes[ok], self._band(df)[ok]]
        return out


class Logistic:
    """Multinomial logistic regression. Rows with a real phase (unreduced matches) are used as they are.

    A reduced row has no phase (D-030). Encoding that as an all-zero one-hot row, or adding a missing flag,
    would be `is_reduced` by the back door, and filling one fixed phase would bias those rows. So the phase is
    marginalised out for those rows only: in training each one is repeated once per phase with weight equal to
    that phase's train share, and at prediction the three predictions are averaged with the same shares.
    lean: masked numerics (target-based) still get the train median; a linear model cannot isolate that
    point mass the way it could the all-zero row - revisit if a variant without them scores differently."""

    name = "logistic"
    C_GRID = (0.01, 0.1, 1.0)  # lean: three values; at ~1.4M rows C barely moves the fit

    def __init__(self, C: float | None = None):
        self.C = C

    def _levels(self, train: pd.DataFrame) -> dict[str, list[str]]:
        return {"phase": PHASES, "competition": sorted(train["competition"].dropna().unique())}

    def _X(self, df: pd.DataFrame, phase: str | None = None) -> np.ndarray:
        """Design matrix. `phase` forces one phase on every row (used for the masked rows' copies)."""
        blocks = [self.num.transform(df[NUMERIC])]
        for col in CATEGORICAL:
            values = pd.Series(phase, index=df.index) if col == "phase" and phase else df[col]
            codes = _codes(values, self.levels[col])
            onehot = np.zeros((len(df), len(self.levels[col])))
            ok = codes >= 0
            onehot[np.flatnonzero(ok), codes[ok]] = 1.0
            blocks.append(onehot)
        return np.hstack(blocks)

    def _expand(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Training matrix, labels and weights: unreduced rows once, each reduced row once per phase."""
        masked = df["phase"].isna().to_numpy()
        real, part = df[~masked], df[masked]
        X, y, w = [self._X(real)], [real["y"].to_numpy()], [np.ones(len(real))]
        for phase, share in zip(PHASES, self.phase_share, strict=True):
            if masked.any():  # sklearn's transform rejects an empty frame
                X.append(self._X(part, phase))
                y.append(part["y"].to_numpy())
                w.append(np.full(len(part), share))
        return np.vstack(X), np.concatenate(y), np.concatenate(w)

    def tune(self, fit_part: pd.DataFrame, holdout: pd.DataFrame) -> dict[float, float]:
        """Pick C by holdout log-loss; returns the whole grid for the report."""
        scores = {
            C: log_loss(Logistic(C).fit(fit_part).predict_proba(holdout), holdout["y"].to_numpy())
            for C in self.C_GRID
        }
        self.C = min(scores, key=scores.get)
        return scores

    def fit(self, train: pd.DataFrame) -> "Logistic":
        if self.C is None:
            raise ValueError("call tune() first or pass C")
        self.levels = self._levels(train)
        # Phase shares from the rows that have a phase (unreduced matches); fitted on train only.
        shares = train["phase"].value_counts(normalize=True).reindex(PHASES).fillna(0.0)
        self.phase_share = shares.to_numpy()
        self.num = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        self.num.fit(train[NUMERIC])
        self.lr = LogisticRegression(C=self.C, max_iter=300, random_state=config.SEED)
        X, y, w = self._expand(train)
        self.lr.fit(X, y, sample_weight=w)
        assert list(self.lr.classes_) == list(range(K))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        masked = df["phase"].isna().to_numpy()
        out = np.empty((len(df), K))
        if (~masked).any():
            out[~masked] = self.lr.predict_proba(self._X(df[~masked]))
        if masked.any():
            part = df[masked]
            out[masked] = sum(
                share * self.lr.predict_proba(self._X(part, phase))
                for phase, share in zip(PHASES, self.phase_share, strict=True)
            )
        return out


class Lgbm:
    """LightGBM multiclass on the same features; masked values stay NaN (native missing handling, D-029)."""

    name = "lightgbm"
    MAX_TREES = 2000

    def __init__(
        self, n_estimators: int | None = None, features: list[str] | None = None, seed: int = config.SEED
    ):
        self.n_estimators = n_estimators
        self.features = features or FEATURES  # a shorter list is how the reduced-gap check drops inputs
        self.seed = seed  # Phase 3 averages three seeds (D-038(6))

    def _model(self, n_estimators: int) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=100,
            colsample_bytree=0.8,
            random_state=self.seed,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for col in CATEGORICAL:
            if col in X:
                X[col] = pd.Categorical(X[col], categories=self.levels[col])
        return X

    def tune(self, fit_part: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, float]:
        """Early-stop on the holdout year to pick the tree count (val is not touched)."""
        self.levels = {"phase": PHASES, "competition": sorted(fit_part["competition"].dropna().unique())}
        model = self._model(self.MAX_TREES)
        model.fit(
            self._X(fit_part),
            fit_part["y"],
            eval_X=self._X(holdout),
            eval_y=holdout["y"],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        self.n_estimators = int(model.best_iteration_ or self.MAX_TREES)
        best = next(iter(model.best_score_.values()))["multi_logloss"]  # Phase 3 compares arms on this
        return {"best_iteration": self.n_estimators, "holdout_log_loss": float(best)}

    def fit(self, train: pd.DataFrame) -> "Lgbm":
        if self.n_estimators is None:
            raise ValueError("call tune() first or pass n_estimators")
        self.levels = {"phase": PHASES, "competition": sorted(train["competition"].dropna().unique())}
        self.model = self._model(self.n_estimators).fit(self._X(train), train["y"])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._X(df))


def _check_proba(P: np.ndarray, n: int, name: str) -> None:
    if P.shape != (n, K) or not np.isfinite(P).all() or not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{name}: predictions are not an (n, {K}) probability matrix")


def _slice_metrics(P: np.ndarray, y: np.ndarray, row_loss: np.ndarray, match_ids: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0}
    return {
        "n": int(len(y)),
        "log_loss": float(row_loss.mean()),
        "brier": brier(P, y),
        "ece": ece(P, y),
        "log_loss_ci": bootstrap_ci(row_loss, match_ids),
    }


def score_models(models: list, val: pd.DataFrame) -> list[dict]:
    """Score every model on the identical val rows: all / unreduced-only / reduced-only slices, bootstrap CIs
    over matches, and a paired comparison with the previous model in the list (D-029)."""
    y = val["y"].to_numpy()
    ids = val["match_id"].to_numpy()
    reduced = val["is_reduced"].to_numpy(dtype=bool)
    slices = {"all": np.ones(len(val), dtype=bool), "unreduced": ~reduced, "reduced": reduced}
    results, prev_name, prev_loss = [], None, None
    for model in models:
        P = model.predict_proba(val)
        _check_proba(P, len(val), model.name)
        row_loss = row_log_loss(P, y)
        entry = {"name": model.name}
        for label, mask in slices.items():
            entry[label] = _slice_metrics(P[mask], y[mask], row_loss[mask], ids[mask])
        if prev_loss is None:
            entry["beats_previous"] = None
        else:
            paired = paired_bootstrap_diff(row_loss, prev_loss, ids)
            entry["paired_vs_previous"] = {"previous": prev_name, **paired}
            entry["beats_previous"] = bool(paired["value"] < 0)
            entry["beats_previous_ci_excludes_0"] = bool(paired["hi"] < 0)
        results.append(entry)
        prev_name, prev_loss = model.name, row_loss
    return results


def run(processed_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    train_d = load_deliveries(("train",), processed_dir)
    vocab = competition_vocab(train_d)  # train rows only
    train = build_context(train_d, vocab)
    val = build_context(load_deliveries(("val",), processed_dir), vocab)
    del train_d
    assert set(train["split"]) == {"train"} and set(val["split"]) == {"val"}  # the test seasons stay unread

    fit_part = train[train["year"] < config.TUNE_HOLDOUT_YEAR]
    holdout = train[train["year"] == config.TUNE_HOLDOUT_YEAR]
    glob = GlobalFreq()
    phase = PhaseFreq(glob)
    phase_wk = PhaseWicketsFreq(phase)
    logistic, lgbm = Logistic(), Lgbm()
    tuning = {
        "holdout_year": config.TUNE_HOLDOUT_YEAR,
        "fit_rows": len(fit_part),
        "holdout_rows": len(holdout),
        "logistic_holdout_log_loss_by_C": {str(c): v for c, v in logistic.tune(fit_part, holdout).items()},
        "logistic_C": logistic.C,
        "lightgbm": lgbm.tune(fit_part, holdout),
    }
    models = [glob, phase, phase_wk, logistic, lgbm]
    for m in models:
        m.fit(train)  # refit on all of train with the chosen settings

    baselines = score_models(models, val)
    last = baselines[-1]
    report = {
        "snapshot": config.DATA_SNAPSHOT,
        "seed": config.SEED,
        "classes": config.OUTCOME_CLASSES,
        "rows": {
            "train": len(train),
            "val": len(val),
            "val_matches": int(val["match_id"].nunique()),
            "train_rows_masked": int(train["is_reduced"].sum()),
            "val_rows_masked": int(val["is_reduced"].sum()),
        },
        "tuning": tuning,
        "baselines": baselines,
        # D-016 compares two MLPs on the same val rows, so its threshold is read against the *paired* width.
        "d016_calibration": {
            "lightgbm_log_loss_ci_width": last["all"]["log_loss_ci"]["width"],
            "paired_ci_width_lightgbm_vs_logistic": last["paired_vs_previous"]["width"],
        },
    }
    out_dir = out_dir or config.REPORTS / "p2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baselines.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


if __name__ == "__main__":
    for b in run()["baselines"]:
        print(f"{b['name']:20s} val log-loss {b['all']['log_loss']:.4f}  beats prev: {b['beats_previous']}")
