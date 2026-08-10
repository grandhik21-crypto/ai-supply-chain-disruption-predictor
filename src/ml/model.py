"""
Reusable XGBoost model for predicting shipment disruptions.

What this file does (step by step):
  1. Load the engineered ML features CSV
  2. Create / use a disruption label (0 = no disruption, 1 = disruption)
  3. Split into train and test sets
  4. Tune hyperparameters with cross-validation
  5. Train the final XGBoost model
  6. Evaluate Accuracy, Precision, Recall, F1, ROC-AUC
  7. Save the trained model with joblib

Example:
    predictor = DisruptionPredictor()
    metrics = predictor.run_pipeline()
    predictor.save_model()
"""

from __future__ import annotations  # Modern type hint support

from dataclasses import dataclass, field  # For settings and metrics classes
from pathlib import Path  # For file paths

import joblib  # Saves / loads the trained model to disk
import numpy as np  # Numeric arrays
import pandas as pd  # Tables
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from xgboost import XGBClassifier  # Gradient-boosted tree model

from config.settings import MODEL_PATH, RANDOM_SEED  # Configurable model path/seed
from src.features.feature_engineering import DEFAULT_ML_OUTPUT_PATH
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Where the trained model file is saved (override with SC_MODEL_PATH)
DEFAULT_MODEL_PATH = MODEL_PATH

# Numeric columns used as model inputs (features)
NUMERIC_FEATURES: tuple[str, ...] = (
    "risk_score",
    "rolling_lead_time_7d",
    "lead_time_variance_7d",
    "inventory_coverage",
    "supplier_reliability",
    "rolling_sentiment_7d",
    "sentiment_velocity",
    "negative_news_count_7d",
    "sentiment_score_daily",
    "article_count",
)

# Text columns turned into 0/1 columns (one-hot encoding)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "region",
    "category",
)

# Name of the column we want to predict
TARGET_COLUMN = "disruption"

# Small grid of XGBoost settings to try during tuning
DEFAULT_PARAM_GRID: dict[str, list] = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}


@dataclass(frozen=True)
class ModelMetrics:
    """Holds evaluation scores for the trained model."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    best_params: dict

    def as_dict(self) -> dict[str, float | dict]:
        """Convert metrics to a plain dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "cv_accuracy_mean": self.cv_accuracy_mean,
            "cv_accuracy_std": self.cv_accuracy_std,
            "best_params": self.best_params,
        }


@dataclass
class DisruptionPredictor:
    """
    Train and use an XGBoost model that predicts shipment disruption.

    Disruption = 1 means a shipment is likely to be disrupted.
    Disruption = 0 means it looks normal.
    """

    features_path: Path = field(default_factory=lambda: DEFAULT_ML_OUTPUT_PATH)
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    test_size: float = 0.2  # 20% of data held out for testing
    random_state: int = RANDOM_SEED  # Makes train/test split reproducible
    cv_folds: int = 5  # Number of cross-validation folds
    param_grid: dict[str, list] = field(default_factory=lambda: dict(DEFAULT_PARAM_GRID))

    # Filled in after training
    model: XGBClassifier | None = field(default=None, init=False, repr=False)
    feature_names_: list[str] = field(default_factory=list, init=False, repr=False)
    metrics_: ModelMetrics | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Step 1–2: Load data and prepare labels / features
    # ------------------------------------------------------------------

    def load_features(self, path: Path | str | None = None) -> pd.DataFrame:
        """Step 1 — Load the engineered ML features CSV."""
        csv_path = Path(path) if path is not None else self.features_path
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Features file not found: {csv_path}. "
                "Run: python3 scripts/run_feature_engineering.py"
            )
        logger.info("Loading ML features from '%s'", csv_path)
        df = pd.read_csv(csv_path)
        logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))
        return df

    @staticmethod
    def create_disruption_labels(df: pd.DataFrame) -> pd.Series:
        """
        Step 2a — Build a disruption label when the CSV has no real labels yet.

        Simple business rules (demo / placeholder target):
          - High risk score (>= 70), OR
          - Low inventory coverage (< 15 days), OR
          - Low reliability (< 0.85) with negative news

        Returns 1 = disruption likely, 0 = normal.
        """
        risk = df.get("risk_score", pd.Series(0, index=df.index)).fillna(0)
        inventory = df.get("inventory_coverage", pd.Series(999, index=df.index)).fillna(999)
        reliability = df.get("supplier_reliability", pd.Series(1, index=df.index)).fillna(1)
        neg_news = df.get("negative_news_count_7d", pd.Series(0, index=df.index)).fillna(0)
        sentiment = df.get("rolling_sentiment_7d", pd.Series(0.5, index=df.index)).fillna(0.5)

        disruption = (
            (risk >= 70)
            | (inventory < 15)
            | ((reliability < 0.85) & (neg_news > 0))
            | (sentiment < 0.35)
        ).astype(int)

        # Add a little random noise so the model must learn patterns
        # (not just memorize a perfect rule)
        rng = np.random.default_rng(42)
        flip_mask = rng.random(len(df)) < 0.05  # flip ~5% of labels
        disruption = disruption.where(~flip_mask, 1 - disruption)

        logger.info(
            "Created disruption labels: %d positive (%.1f%%)",
            int(disruption.sum()),
            100 * float(disruption.mean()),
        )
        return disruption

    def prepare_xy(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Step 2b — Build feature matrix X and target vector y.

        - Uses existing 'disruption' column if present
        - Otherwise creates demo labels
        - One-hot encodes region/category
        - Fills missing numeric values with column medians
        """
        data = df.copy()

        # Create target if missing
        if TARGET_COLUMN not in data.columns:
            data[TARGET_COLUMN] = self.create_disruption_labels(data)

        y = data[TARGET_COLUMN].astype(int)

        # Keep only feature columns that exist
        numeric_cols = [c for c in NUMERIC_FEATURES if c in data.columns]
        categorical_cols = [c for c in CATEGORICAL_FEATURES if c in data.columns]

        X = data[numeric_cols + categorical_cols].copy()

        # Fill missing numbers with median (vectorized)
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())

        # Turn text categories into 0/1 columns
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

        # Make sure all values are float for XGBoost
        X = X.astype(float)

        self.feature_names_ = list(X.columns)
        logger.info(
            "Prepared X shape=%s, y positives=%d/%d",
            X.shape,
            int(y.sum()),
            len(y),
        )
        return X, y

    # ------------------------------------------------------------------
    # Step 3–5: Split, cross-validate, tune, train
    # ------------------------------------------------------------------

    @staticmethod
    def _min_class_count(y: pd.Series) -> int:
        """Size of the smallest class — limits how many CV folds are possible."""
        counts = y.value_counts()
        return int(counts.min()) if not counts.empty else 0

    def _usable_cv_folds(self, y: pd.Series) -> int:
        """
        Pick a fold count this dataset can actually support.

        Stratified k-fold needs at least k examples of every class, so small
        or heavily imbalanced datasets need fewer folds (or none at all).
        """
        min_count = self._min_class_count(y)
        if min_count < 2:
            return 0  # Cross-validation is impossible with a single-member class
        return max(2, min(self.cv_folds, min_count))

    def split_train_test(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Step 3 — Split data into train (80%) and test (20%) sets."""
        # Stratifying needs at least 2 examples per class; fall back if not
        stratify = y if self._min_class_count(y) >= 2 else None
        if stratify is None:
            logger.warning(
                "Not enough examples per class to stratify — using a random split."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,  # Keep same disruption % in train and test
        )
        logger.info(
            "Train/test split: train=%d, test=%d",
            len(X_train),
            len(X_test),
        )
        return X_train, X_test, y_train, y_test

    def _base_estimator(self) -> XGBClassifier:
        """Create a default XGBoost classifier (before tuning)."""
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
        )

    def cross_validate_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> dict[str, float]:
        """
        Step 4a — Cross-validation on the training set.

        Splits training data into cv_folds parts, trains on each fold,
        and reports average accuracy / F1 / ROC-AUC.
        """
        folds = self._usable_cv_folds(y_train)
        if folds == 0:
            logger.warning(
                "Skipping cross-validation: the smallest class has fewer than "
                "2 examples in the training set."
            )
            return {
                "cv_accuracy_mean": 0.0,
                "cv_accuracy_std": 0.0,
                "cv_f1_mean": 0.0,
                "cv_roc_auc_mean": 0.0,
            }

        if folds < self.cv_folds:
            logger.warning(
                "Reducing cross-validation folds from %d to %d (small class size).",
                self.cv_folds,
                folds,
            )

        logger.info("Running %d-fold cross-validation...", folds)
        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.random_state,
        )
        scores = cross_validate(
            self._base_estimator(),
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1,
        )
        summary = {
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "cv_accuracy_std": float(scores["test_accuracy"].std()),
            "cv_f1_mean": float(scores["test_f1"].mean()),
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
        }
        logger.info(
            "CV accuracy=%.3f ± %.3f | F1=%.3f | ROC-AUC=%.3f",
            summary["cv_accuracy_mean"],
            summary["cv_accuracy_std"],
            summary["cv_f1_mean"],
            summary["cv_roc_auc_mean"],
        )
        return summary

    def tune_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> XGBClassifier:
        """
        Step 4b — Try different XGBoost settings and pick the best ones.

        Uses GridSearchCV with stratified cross-validation.
        Scoring metric: F1 (balances precision and recall).
        """
        folds = self._usable_cv_folds(y_train)
        if folds == 0:
            # Not enough data per class to search — train a single default model
            logger.warning(
                "Skipping hyperparameter search (too few examples per class). "
                "Training a default model instead."
            )
            model = self._base_estimator()
            model.fit(X_train, y_train)
            return model

        logger.info("Tuning hyperparameters with GridSearchCV...")
        cv = StratifiedKFold(
            n_splits=min(folds, 3),  # slightly fewer folds for speed
            shuffle=True,
            random_state=self.random_state,
        )
        search = GridSearchCV(
            estimator=self._base_estimator(),
            param_grid=self.param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,  # Retrain best model on full training set
            verbose=0,
        )
        search.fit(X_train, y_train)
        logger.info("Best params: %s", search.best_params_)
        logger.info("Best CV F1: %.3f", search.best_score_)
        return search.best_estimator_

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, *, tune: bool = True
    ) -> XGBClassifier:
        """Step 5 — Train the final XGBoost model (optionally after tuning)."""
        if tune:
            self.model = self.tune_hyperparameters(X_train, y_train)
        else:
            logger.info("Training XGBoost without hyperparameter search...")
            self.model = self._base_estimator()
            self.model.fit(X_train, y_train)
        logger.info("Model training complete")
        return self.model

    # ------------------------------------------------------------------
    # Step 6: Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        *,
        cv_summary: dict[str, float] | None = None,
        best_params: dict | None = None,
    ) -> ModelMetrics:
        """
        Step 6 — Score the model on the held-out test set.

        Metrics:
          - Accuracy  = overall % correct
          - Precision = of predicted disruptions, how many were real
          - Recall    = of real disruptions, how many we caught
          - F1        = balance of precision and recall
          - ROC-AUC   = how well the model ranks risk overall
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # ROC-AUC needs both classes present in the test set
        if y_test.nunique() < 2:
            logger.warning(
                "Test set contains only one class — ROC-AUC is not defined; "
                "reporting 0.0."
            )
            roc_auc = 0.0
        else:
            roc_auc = float(roc_auc_score(y_test, y_proba))

        metrics = ModelMetrics(
            accuracy=round(float(accuracy_score(y_test, y_pred)), 4),
            precision=round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            recall=round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            f1=round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            roc_auc=round(roc_auc, 4),
            cv_accuracy_mean=round(float((cv_summary or {}).get("cv_accuracy_mean", 0.0)), 4),
            cv_accuracy_std=round(float((cv_summary or {}).get("cv_accuracy_std", 0.0)), 4),
            best_params=best_params or {},
        )
        self.metrics_ = metrics

        logger.info(
            "Test metrics | Acc=%.3f Prec=%.3f Rec=%.3f F1=%.3f AUC=%.3f",
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.roc_auc,
        )
        return metrics

    # ------------------------------------------------------------------
    # Step 7: Save / load with joblib
    # ------------------------------------------------------------------

    def save_model(self, path: Path | str | None = None) -> Path:
        """Step 7 — Save the trained model (and feature names) with joblib."""
        if self.model is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        save_path = Path(path) if path is not None else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "feature_names": self.feature_names_,
            "metrics": self.metrics_.as_dict() if self.metrics_ else {},
            "target_column": TARGET_COLUMN,
        }
        joblib.dump(payload, save_path)
        logger.info("Saved model to '%s'", save_path)
        return save_path

    def load_model(self, path: Path | str | None = None) -> XGBClassifier:
        """Load a previously saved model from disk."""
        load_path = Path(path) if path is not None else self.model_path
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        payload = joblib.load(load_path)
        self.model = payload["model"]
        self.feature_names_ = list(payload.get("feature_names", []))
        logger.info("Loaded model from '%s'", load_path)
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict disruption class (0 or 1) for new rows."""
        if self.model is None:
            raise RuntimeError("Model is not loaded/trained.")
        X_aligned = self._align_features(X)
        return self.model.predict(X_aligned)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict disruption probability (0.0–1.0) for new rows."""
        if self.model is None:
            raise RuntimeError("Model is not loaded/trained.")
        X_aligned = self._align_features(X)
        return self.model.predict_proba(X_aligned)[:, 1]

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make sure new data has the same columns the model was trained on."""
        aligned = X.copy()
        for col in self.feature_names_:
            if col not in aligned.columns:
                aligned[col] = 0.0
        return aligned[self.feature_names_].astype(float)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        *,
        tune: bool = True,
        save: bool = True,
    ) -> ModelMetrics:
        """
        Run the full ML pipeline end-to-end.

        1. Load features
        2. Prepare X and y
        3. Train/test split
        4. Cross-validation
        5. Hyperparameter tuning + training
        6. Evaluate on test set
        7. Save model with joblib
        """
        logger.info("=== Starting disruption ML pipeline ===")

        df = self.load_features()
        X, y = self.prepare_xy(df)
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)

        cv_summary = self.cross_validate_model(X_train, y_train)
        self.train(X_train, y_train, tune=tune)

        best_params = {}
        if self.model is not None:
            best_params = self.model.get_params()
            # Keep only the tuned knobs for a cleaner metrics report
            best_params = {
                k: best_params[k]
                for k in ("n_estimators", "max_depth", "learning_rate", "subsample")
                if k in best_params
            }

        metrics = self.evaluate(
            X_test,
            y_test,
            cv_summary=cv_summary,
            best_params=best_params,
        )

        if save:
            self.save_model()

        logger.info("=== Disruption ML pipeline complete ===")
        return metrics
