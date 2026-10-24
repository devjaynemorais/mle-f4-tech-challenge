"""Detecção estatística de drift entre referência e dados atuais."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats
from scipy.spatial.distance import jensenshannon

DEFAULT_EXCLUDED_COLUMNS = {"SK_ID_CURR", "TARGET", "origem", "inadimplente"}
NUMERIC_METHODS = {"ks", "psi", "kl", "js"}


@dataclass
class DriftResult:
    """Resultado de um método de drift aplicado a uma feature."""

    feature_name: str
    method: str
    statistic: float
    p_value: float | None
    drift_detected: bool
    threshold: float


@dataclass
class DriftReport:
    """Resultados agregados de drift para um conjunto de features."""

    results: list[DriftResult]
    n_features_total: int
    n_features_drifted: int
    drifted_features: list[str]


class DriftDetector:
    """Executa testes e métricas de drift feature a feature."""

    PSI_STABLE = 0.10
    PSI_MODERATE = 0.25

    def __init__(
        self,
        alpha: float = 0.05,
        psi_threshold: float = 0.25,
        n_bins: int = 10,
    ) -> None:
        """Configura significância, limiar PSI e número máximo de bins."""
        if not 0 < alpha < 1:
            raise ValueError("alpha deve estar no intervalo (0, 1)")
        if psi_threshold <= 0:
            raise ValueError("psi_threshold deve ser positivo")
        if n_bins < 2:
            raise ValueError("n_bins deve ser pelo menos 2")
        self.alpha = alpha
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins

    @staticmethod
    def _numeric_array(values: ArrayLike) -> np.ndarray:
        """Converte valores numéricos e representa inválidos como NaN."""
        series = pd.Series(np.asarray(values).ravel())
        return pd.to_numeric(series, errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )

    @staticmethod
    def _require_samples(reference: np.ndarray, current: np.ndarray) -> None:
        """Garante amostras finitas nos dois períodos."""
        if reference.size == 0 or current.size == 0:
            raise ValueError("ambos os períodos precisam ter valores válidos")

    def ks_test(
        self,
        reference: ArrayLike,
        current: ArrayLike,
        feature_name: str = "feature",
    ) -> DriftResult:
        """Aplica KS bilateral e decide pelo p-valor comparado a alpha."""
        ref = self._numeric_array(reference)
        cur = self._numeric_array(current)
        ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
        if ref.size == 0 or cur.size == 0:
            return DriftResult(
                feature_name, "ks", float("nan"), None, False, self.alpha
            )
        statistic, p_value = stats.ks_2samp(ref, cur)
        return DriftResult(
            feature_name,
            "ks",
            float(statistic),
            float(p_value),
            bool(p_value < self.alpha),
            self.alpha,
        )

    def _bin_count(self, n_bins: int | None) -> int:
        """Resolve e valida o número de bins de uma métrica."""
        bins = self.n_bins if n_bins is None else n_bins
        if bins < 2:
            raise ValueError("n_bins deve ser pelo menos 2")
        return bins

    @staticmethod
    def _reference_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
        """Cria limites por quantis da referência, incluindo extremos abertos."""
        finite = reference[np.isfinite(reference)]
        if finite.size == 0:
            return np.array([-np.inf, np.inf])
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        interior = np.unique(np.quantile(finite, quantiles))
        return np.concatenate(([-np.inf], interior, [np.inf]))

    def _histogram_probabilities(
        self,
        reference: ArrayLike,
        current: ArrayLike,
        n_bins: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calcula proporções em bins fixos da referência e em um bin nulo."""
        ref, cur = self._numeric_array(reference), self._numeric_array(current)
        self._require_samples(ref, cur)
        edges = self._reference_edges(ref, self._bin_count(n_bins))
        ref_values, cur_values = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
        ref_counts = np.histogram(ref_values, bins=edges)[0]
        cur_counts = np.histogram(cur_values, bins=edges)[0]
        ref_counts = np.append(ref_counts, ref.size - ref_values.size)
        cur_counts = np.append(cur_counts, cur.size - cur_values.size)
        return self._smoothed_probabilities(ref_counts, cur_counts)

    @staticmethod
    def _smoothed_probabilities(
        reference_counts: np.ndarray, current_counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Suaviza bins vazios para evitar divisão por zero e log de zero."""
        epsilon = 1e-8
        bins = reference_counts.size
        reference = (reference_counts + epsilon) / (
            reference_counts.sum() + epsilon * bins
        )
        current = (current_counts + epsilon) / (current_counts.sum() + epsilon * bins)
        return reference, current

    def psi(
        self,
        reference: ArrayLike,
        current: ArrayLike,
        feature_name: str = "feature",
        n_bins: int | None = None,
    ) -> DriftResult:
        """Calcula PSI com cortes de quantis da referência e nulos explícitos."""
        ref, cur = self._histogram_probabilities(reference, current, n_bins)
        statistic = float(np.sum((cur - ref) * np.log(cur / ref)))
        return DriftResult(
            feature_name,
            "psi",
            statistic,
            None,
            statistic >= self.psi_threshold,
            self.psi_threshold,
        )

    def kl_divergence(
        self,
        reference: ArrayLike,
        current: ArrayLike,
        feature_name: str = "feature",
        n_bins: int | None = None,
    ) -> DriftResult:
        """Calcula KL(reference || current) em bins de referência."""
        ref, cur = self._histogram_probabilities(reference, current, n_bins)
        statistic = float(np.sum(ref * np.log(ref / cur)))
        return DriftResult(feature_name, "kl", statistic, None, statistic > 0.1, 0.1)

    def js_divergence(
        self,
        reference: ArrayLike,
        current: ArrayLike,
        feature_name: str = "feature",
        n_bins: int | None = None,
    ) -> DriftResult:
        """Calcula a divergência Jensen-Shannon, não a distância ao quadrado."""
        ref, cur = self._histogram_probabilities(reference, current, n_bins)
        statistic = float(jensenshannon(ref, cur, base=2) ** 2)
        return DriftResult(feature_name, "js", statistic, None, statistic > 0.05, 0.05)

    def chi2_test(
        self,
        reference: ArrayLike,
        current: ArrayLike,
        feature_name: str = "feature",
    ) -> DriftResult:
        """Compara proporções categóricas por qui-quadrado, incluindo nulos."""
        ref = np.asarray(reference, dtype=object).ravel()
        cur = np.asarray(current, dtype=object).ravel()
        if ref.size == 0 or cur.size == 0:
            raise ValueError("ambos os períodos precisam ter valores")
        codes, categories = pd.factorize(
            pd.Series(np.concatenate((ref, cur)), dtype=object),
            sort=False,
            use_na_sentinel=False,
        )
        return self._categorical_result(codes, len(ref), len(categories), feature_name)

    def _categorical_result(
        self,
        codes: np.ndarray,
        n_reference: int,
        n_categories: int,
        feature_name: str,
    ) -> DriftResult:
        """Monta o resultado qui-quadrado a partir das categorias codificadas."""
        ref_counts = np.bincount(codes[:n_reference], minlength=n_categories)
        cur_counts = np.bincount(codes[n_reference:], minlength=n_categories)
        if n_categories < 2:
            return DriftResult(feature_name, "chi2", 0.0, 1.0, False, self.alpha)
        statistic, p_value, _, _ = stats.chi2_contingency(
            np.vstack((ref_counts, cur_counts)), correction=False
        )
        return DriftResult(
            feature_name,
            "chi2",
            float(statistic),
            float(p_value),
            bool(p_value < self.alpha),
            self.alpha,
        )

    @staticmethod
    def _infer_columns(
        reference_df: pd.DataFrame, excluded: set[str]
    ) -> tuple[list[str], list[str]]:
        """Infere grupos numéricos e categóricos pelos tipos da referência."""
        numeric = set(reference_df.select_dtypes(include=[np.number]).columns)
        numeric_cols = [c for c in reference_df.columns if c in numeric - excluded]
        categorical_cols = [
            c for c in reference_df.columns if c not in numeric and c not in excluded
        ]
        return numeric_cols, categorical_cols

    @staticmethod
    def _validate_columns(
        reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: list[str]
    ) -> None:
        """Exige que todas as features selecionadas existam nos dois períodos."""
        missing = set(columns) - set(reference_df.columns) | (
            set(columns) - set(current_df.columns)
        )
        if missing:
            raise ValueError(f"colunas de comparação ausentes: {sorted(missing)}")

    @staticmethod
    def _validate_methods(methods: list[str]) -> None:
        """Rejeita nomes de métricas numéricas não suportados."""
        unsupported = set(methods) - NUMERIC_METHODS
        if unsupported:
            raise ValueError(f"métodos numéricos não suportados: {sorted(unsupported)}")

    @staticmethod
    def _excluded_columns(
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        include_columns: list[str] | None,
        exclude_columns: list[str] | None,
    ) -> set[str]:
        """Resolve exclusões padrão ou limita a seleção a colunas explícitas."""
        explicit_exclusions = set(exclude_columns or [])
        if include_columns is None:
            return DEFAULT_EXCLUDED_COLUMNS | explicit_exclusions
        missing = set(include_columns) - set(reference_df.columns)
        missing |= set(include_columns) - set(current_df.columns)
        if missing:
            raise ValueError(f"colunas de comparação ausentes: {sorted(missing)}")
        return set(reference_df.columns) - set(include_columns) | explicit_exclusions

    def detect_all(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        numerical_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        methods: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        include_columns: list[str] | None = None,
    ) -> DriftReport:
        """Aplica métodos numéricos e qui-quadrado às colunas selecionadas."""
        excluded = self._excluded_columns(
            reference_df, current_df, include_columns, exclude_columns
        )
        inferred_numeric, inferred_categorical = self._infer_columns(
            reference_df, excluded
        )
        numeric_candidates = (
            inferred_numeric if numerical_cols is None else numerical_cols
        )
        categorical_candidates = (
            inferred_categorical if categorical_cols is None else categorical_cols
        )
        numerical_cols = [col for col in numeric_candidates if col not in excluded]
        categorical_cols = [
            col for col in categorical_candidates if col not in excluded
        ]
        methods = ["ks", "psi"] if methods is None else list(dict.fromkeys(methods))
        self._validate_methods(methods)
        columns = numerical_cols + categorical_cols
        self._validate_columns(reference_df, current_df, columns)
        results = self._numeric_results(
            reference_df, current_df, numerical_cols, methods
        )
        results.extend(
            self._categorical_results(reference_df, current_df, categorical_cols)
        )
        return self._build_report(results, len(set(columns)))

    def _numeric_results(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        columns: list[str],
        methods: list[str],
    ) -> list[DriftResult]:
        """Executa os métodos numéricos para cada coluna solicitada."""
        method_map = {
            "ks": self.ks_test,
            "psi": self.psi,
            "kl": self.kl_divergence,
            "js": self.js_divergence,
        }
        return [
            method_map[method](
                reference_df[col].to_numpy(), current_df[col].to_numpy(), col
            )
            for col in columns
            for method in methods
        ]

    def _categorical_results(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        columns: list[str],
    ) -> list[DriftResult]:
        """Executa qui-quadrado para cada coluna categórica."""
        return [
            self.chi2_test(
                reference_df[col].to_numpy(), current_df[col].to_numpy(), col
            )
            for col in columns
        ]

    @staticmethod
    def _build_report(results: list[DriftResult], n_features: int) -> DriftReport:
        """Agrega a lista de resultados e as features sinalizadas."""
        drifted = sorted({item.feature_name for item in results if item.drift_detected})
        return DriftReport(results, n_features, len(drifted), drifted)

    @staticmethod
    def interpret_psi(psi_value: float) -> str:
        """Classifica PSI nos intervalos estável, moderado e severo."""
        if psi_value < DriftDetector.PSI_STABLE:
            return "estável (PSI < 0.10)"
        if psi_value < DriftDetector.PSI_MODERATE:
            return "drift moderado (0.10 ≤ PSI < 0.25)"
        return "drift severo (PSI ≥ 0.25)"
