"""
Clinical Data Processor for MRD Multimodal Prediction.

This module provides :class:`ClinicalProcessor`, a scikit-learn–based pipeline
for loading, cleaning, encoding, and scaling clinical tabular data stored in a
CSV file.  It is designed to be fitted on the training split and then applied
deterministically to validation / test patients.

Key features
------------
* Automatic detection of numeric vs. categorical columns.
* ``StandardScaler`` for numeric features; ``OneHotEncoder`` for categorical
  features (with unknown-category handling).
* Median imputation for numeric missing values; mode imputation (``"MISSING"``
  sentinel) for categorical missing values.
* Pickle-based serialization of the fitted state for reproducible inference.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class ClinicalProcessor:
    """End-to-end clinical tabular data preprocessor.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing clinical data.
    patient_id_col : str
        Column name that uniquely identifies each patient (e.g.
        ``"patient_id"``).
    label_col : str
        Column name for the binary MRD label (expected values: 0 or 1).

    Attributes
    ----------
    df : pd.DataFrame
        Raw clinical DataFrame loaded from *csv_path*.
    numeric_cols : list[str]
        Names of numeric feature columns (determined at :meth:`fit`).
    categorical_cols : list[str]
        Names of categorical feature columns (determined at :meth:`fit`).
    is_fitted : bool
        ``True`` after :meth:`fit` has been called.

    Examples
    --------
    >>> proc = ClinicalProcessor("clinical.csv", "pid", "mrd_label")
    >>> proc.fit(train_patient_ids)
    >>> features = proc.transform("patient_001")   # np.ndarray
    >>> label    = proc.get_label("patient_001")    # 0 or 1
    >>> proc.save("processor.pkl")
    """

    # Columns that are never treated as features
    _EXCLUDE_COLS_TEMPLATE: set = set()  # populated dynamically with id/label

    def __init__(
        self,
        csv_path: Union[str, Path],
        patient_id_col: str = "patient_id",
        label_col: str = "mrd_label",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.patient_id_col = patient_id_col
        self.label_col = label_col

        # Load CSV and index by patient ID
        self.df: pd.DataFrame = pd.read_csv(self.csv_path)
        if self.patient_id_col not in self.df.columns:
            raise ValueError(
                f"Patient ID column '{self.patient_id_col}' not found in "
                f"{self.csv_path}.  Available columns: {list(self.df.columns)}"
            )
        if self.label_col not in self.df.columns:
            raise ValueError(
                f"Label column '{self.label_col}' not found in "
                f"{self.csv_path}.  Available columns: {list(self.df.columns)}"
            )
        if self.df[self.patient_id_col].astype(str).duplicated().any():
            raise ValueError("Clinical CSV contains duplicate patient IDs.")

        # Cast patient IDs to string for consistent lookups
        self.df[self.patient_id_col] = self.df[self.patient_id_col].astype(str)
        self.df = self.df.set_index(self.patient_id_col)

        # Columns excluded from feature engineering
        self._exclude_cols: set = {self.label_col}

        # Fitted components (populated by `fit`)
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self._scaler: Optional[StandardScaler] = None
        self._encoder: Optional[OneHotEncoder] = None
        self._numeric_fill_values: Optional[pd.Series] = None
        self._categorical_fill_values: Optional[pd.Series] = None
        self.is_fitted: bool = False

    # ──────────────────────────────────────────────────────────────────
    # Column classification helpers
    # ──────────────────────────────────────────────────────────────────

    def _identify_columns(self, df_subset: pd.DataFrame) -> None:
        """Classify feature columns into numeric and categorical lists.

        Columns in ``_exclude_cols`` are skipped.  Numeric types
        (``np.number``) are treated as continuous; everything else is
        categorical.

        Parameters
        ----------
        df_subset : pd.DataFrame
            Subset of the full DataFrame (training patients only).
        """
        feature_cols = [c for c in df_subset.columns if c not in self._exclude_cols]
        self.numeric_cols = [
            c for c in feature_cols if pd.api.types.is_numeric_dtype(df_subset[c])
        ]
        self.categorical_cols = [
            c for c in feature_cols if c not in self.numeric_cols
        ]
        logger.info(
            "Identified %d numeric and %d categorical feature columns.",
            len(self.numeric_cols),
            len(self.categorical_cols),
        )

    # ──────────────────────────────────────────────────────────────────
    # Fit / Transform
    # ──────────────────────────────────────────────────────────────────

    def fit(self, patient_ids: Sequence[str]) -> "ClinicalProcessor":
        """Fit scalers and encoders on the training set.

        Steps:

        1. Identify numeric / categorical columns.
        2. Compute per-column fill values (median for numeric, mode for
           categorical).
        3. Fit a ``StandardScaler`` on numeric columns.
        4. Fit a ``OneHotEncoder`` on categorical columns (with
           ``handle_unknown="ignore"`` so unseen categories at inference
           produce all-zero rows).

        Parameters
        ----------
        patient_ids : sequence of str
            Patient identifiers belonging to the training split.

        Returns
        -------
        ClinicalProcessor
            ``self``, for method chaining.

        Raises
        ------
        KeyError
            If any *patient_ids* are not found in the loaded CSV.
        """
        patient_ids = [str(pid) for pid in patient_ids]
        missing = set(patient_ids) - set(self.df.index)
        if missing:
            raise KeyError(
                f"{len(missing)} patient IDs not found in CSV: "
                f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
            )

        df_train = self.df.loc[patient_ids].copy()
        self._identify_columns(df_train)

        # --- Numeric ---
        if self.numeric_cols:
            self._numeric_fill_values = (
                df_train[self.numeric_cols].median().fillna(0.0)
            )
            df_numeric_filled = df_train[self.numeric_cols].fillna(
                self._numeric_fill_values
            )
            self._scaler = StandardScaler()
            self._scaler.fit(df_numeric_filled.values)
        else:
            self._numeric_fill_values = pd.Series(dtype=float)
            self._scaler = None

        # --- Categorical ---
        if self.categorical_cols:
            fill_values = {}
            for column in self.categorical_cols:
                modes = df_train[column].dropna().mode()
                fill_values[column] = modes.iloc[0] if not modes.empty else "MISSING"
            self._categorical_fill_values = pd.Series(fill_values)
            df_cat_filled = (
                df_train[self.categorical_cols]
                .fillna(self._categorical_fill_values)
                .astype(str)
            )
            self._encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                dtype=np.float32,
            )
            self._encoder.fit(df_cat_filled.values)
        else:
            self._categorical_fill_values = pd.Series(dtype=object)
            self._encoder = None

        self.is_fitted = True
        logger.info(
            "ClinicalProcessor fitted: %d numeric + %d one-hot = %d features.",
            len(self.numeric_cols),
            self._encoder.get_feature_names_out().shape[0] if self._encoder else 0,
            self.get_feature_dim(),
        )
        return self

    def transform(self, patient_id: str) -> np.ndarray:
        """Transform a single patient's clinical data into a feature vector.

        Parameters
        ----------
        patient_id : str
            Patient identifier.

        Returns
        -------
        np.ndarray
            1-D float32 array of length :meth:`get_feature_dim`.

        Raises
        ------
        RuntimeError
            If the processor has not been fitted yet.
        KeyError
            If *patient_id* is not found in the CSV.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "ClinicalProcessor has not been fitted. Call fit() first."
            )

        patient_id = str(patient_id)
        if patient_id not in self.df.index:
            raise KeyError(f"Patient '{patient_id}' not found in clinical data.")

        row = self.df.loc[patient_id]

        parts: List[np.ndarray] = []

        # Numeric features
        if self.numeric_cols and self._scaler is not None:
            numeric_vals = (
                row[self.numeric_cols]
                .fillna(self._numeric_fill_values)
                .values.astype(np.float64)
                .reshape(1, -1)
            )
            parts.append(self._scaler.transform(numeric_vals).flatten())

        # Categorical features
        if self.categorical_cols and self._encoder is not None:
            cat_vals = (
                row[self.categorical_cols]
                .fillna(self._categorical_fill_values)
                .astype(str)
                .values.reshape(1, -1)
            )
            parts.append(self._encoder.transform(cat_vals).flatten())

        if not parts:
            return np.array([], dtype=np.float32)

        return np.concatenate(parts).astype(np.float32)

    def fit_transform(
        self, patient_ids: Sequence[str]
    ) -> Dict[str, np.ndarray]:
        """Convenience method: fit on *patient_ids* then transform each one.

        Parameters
        ----------
        patient_ids : sequence of str
            Patient identifiers (training split).

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from patient_id to its processed feature vector.
        """
        self.fit(patient_ids)
        return {str(pid): self.transform(pid) for pid in patient_ids}

    # ──────────────────────────────────────────────────────────────────
    # Label access
    # ──────────────────────────────────────────────────────────────────

    def get_label(self, patient_id: str) -> int:
        """Return the binary MRD label for a patient.

        Parameters
        ----------
        patient_id : str
            Patient identifier.

        Returns
        -------
        int
            0 or 1.

        Raises
        ------
        KeyError
            If *patient_id* is not found in the CSV.
        ValueError
            If the label value is not 0 or 1.
        """
        patient_id = str(patient_id)
        if patient_id not in self.df.index:
            raise KeyError(f"Patient '{patient_id}' not found in clinical data.")

        label = int(self.df.loc[patient_id, self.label_col])
        if label not in (0, 1):
            raise ValueError(
                f"Expected binary label (0 or 1) for patient '{patient_id}', "
                f"got {label}."
            )
        return label

    # ──────────────────────────────────────────────────────────────────
    # Feature dimension
    # ──────────────────────────────────────────────────────────────────

    def get_feature_dim(self) -> int:
        """Return the total number of features after preprocessing.

        Returns
        -------
        int
            Sum of scaled numeric features and one-hot encoded categorical
            features.

        Raises
        ------
        RuntimeError
            If the processor has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "ClinicalProcessor has not been fitted. Call fit() first."
            )
        dim = len(self.numeric_cols)
        if self._encoder is not None:
            dim += self._encoder.get_feature_names_out().shape[0]
        return dim

    # ──────────────────────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted processor to disk via pickle.

        The saved state includes the scaler, encoder, column lists, and
        fill values—but **not** the raw DataFrame (which is reloaded from
        *csv_path* on :meth:`load`).

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``"processor.pkl"``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "csv_path": str(self.csv_path),
            "patient_id_col": self.patient_id_col,
            "label_col": self.label_col,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "scaler": self._scaler,
            "encoder": self._encoder,
            "numeric_fill_values": self._numeric_fill_values,
            "categorical_fill_values": self._categorical_fill_values,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("ClinicalProcessor saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClinicalProcessor":
        """Load a previously saved :class:`ClinicalProcessor`.

        The raw CSV is re-read from the path stored at save time, so the
        CSV must still be accessible at the original location.

        Parameters
        ----------
        path : str or Path
            Path to the pickle file created by :meth:`save`.

        Returns
        -------
        ClinicalProcessor
            A fully restored, fitted processor instance.
        """
        path = Path(path)
        with open(path, "rb") as f:
            state: Dict[str, Any] = pickle.load(f)  # noqa: S301

        instance = cls(
            csv_path=state["csv_path"],
            patient_id_col=state["patient_id_col"],
            label_col=state["label_col"],
        )
        instance.numeric_cols = state["numeric_cols"]
        instance.categorical_cols = state["categorical_cols"]
        instance._scaler = state["scaler"]
        instance._encoder = state["encoder"]
        instance._numeric_fill_values = state["numeric_fill_values"]
        instance._categorical_fill_values = state["categorical_fill_values"]
        instance.is_fitted = state["is_fitted"]

        logger.info("ClinicalProcessor loaded from %s", path)
        return instance

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        dim = self.get_feature_dim() if self.is_fitted else "?"
        return (
            f"ClinicalProcessor({status}, "
            f"numeric={len(self.numeric_cols)}, "
            f"categorical={len(self.categorical_cols)}, "
            f"feature_dim={dim})"
        )
