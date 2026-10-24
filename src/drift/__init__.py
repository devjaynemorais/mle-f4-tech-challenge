"""API pública de detecção e avaliação de drift."""

from src.drift.detect import DriftDetector, DriftReport, DriftResult
from src.drift.evaluation import (
    generate_drift_summary,
    plot_cdf_comparison,
    plot_distribution_comparison,
    plot_drift_heatmap,
    plot_psi_bars,
)

__all__ = [
    "DriftDetector",
    "DriftReport",
    "DriftResult",
    "generate_drift_summary",
    "plot_cdf_comparison",
    "plot_distribution_comparison",
    "plot_drift_heatmap",
    "plot_psi_bars",
]
