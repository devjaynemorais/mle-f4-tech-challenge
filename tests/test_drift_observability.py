"""Drift como métricas e alertas; predição em lote do período de produção."""

import numpy as np
import pandas as pd
import pytest

from src.drift.detect import DriftReport, DriftResult
from src.drift.predict_batch import score_frame, summarize_scores
from src.observability import drift_metrics
from src.observability.alerts import drift_alerts, emit_alerts
from src.observability.drift_metrics import summarize_drift
from src.utils.config_loader import load_config


def _report(n_psi: int, n_drifted: int) -> DriftReport:
    results = [
        DriftResult(
            f"f{i}", "psi", 0.5 if i < n_drifted else 0.01, None, i < n_drifted, 0.25
        )
        for i in range(n_psi)
    ]
    results.append(DriftResult("f0", "ks", 0.3, 0.0, True, 0.05))
    drifted = sorted({r.feature_name for r in results if r.drift_detected})
    return DriftReport(results, n_psi, len(drifted), drifted)


def test_summary_counts_features_per_method() -> None:
    summary = summarize_drift(_report(10, 3))

    assert summary.totals == {"psi": 10, "ks": 1}
    assert summary.drifted == {"psi": 3, "ks": 1}
    assert summary.share("psi") == pytest.approx(0.3)
    assert summary.max_psi == 0.5
    assert summary.share("missing") == 0.0


def test_summary_ignores_non_finite_statistics() -> None:
    report = DriftReport(
        [DriftResult("a", "ks", float("nan"), None, False, 0.05)], 1, 0, []
    )

    assert summarize_drift(report).ks == {}


@pytest.mark.parametrize(
    ("n_drifted", "expected"),
    [(0, []), (1, []), (2, ["DataDriftDetected"]), (5, ["DataDriftSevere"])],
)
def test_alert_thresholds_follow_the_configured_shares(n_drifted, expected) -> None:
    summary = summarize_drift(_report(100, n_drifted))

    alerts = drift_alerts(summary, "stable_vs_drifted", load_config().observability)

    assert [alert.name for alert in alerts] == expected


def test_real_summary_numbers_trigger_the_critical_alert() -> None:
    summary = summarize_drift(_report(114, 7))

    alerts = drift_alerts(summary, "stable_vs_drifted", load_config().observability)

    assert alerts[0].severity == "critical"
    assert alerts[0].value == pytest.approx(7 / 114)


def test_emit_alerts_writes_alert_events(tmp_path) -> None:
    summary = summarize_drift(_report(100, 5))
    emit_alerts(drift_alerts(summary, "stable_vs_drifted", load_config().observability))

    text = (tmp_path / "events-root/logs/pipeline_events.jsonl").read_text()
    assert '"event": "alert_raised"' in text
    assert "DataDriftSevere" in text


def test_drift_registry_exposes_the_documented_metric_names() -> None:
    summary = summarize_drift(_report(10, 3))

    registry = drift_metrics.build_drift_registry(summary, 1)

    assert registry.get_sample_value("drift_share", {"method": "psi"}) == pytest.approx(
        0.3
    )
    assert registry.get_sample_value("drift_features_total", {"method": "psi"}) == 10
    assert registry.get_sample_value("drift_features_drifted", {"method": "psi"}) == 3
    assert registry.get_sample_value("drift_max_psi") == 0.5
    assert registry.get_sample_value("drift_feature_psi", {"feature": "f0"}) == 0.5
    assert registry.get_sample_value("drift_alerts_active") == 1


class _FixedModel:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probability = np.linspace(0.1, 0.9, len(features))
        return np.column_stack([1 - probability, probability])


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"SK_ID_CURR": [1, 2, 3, 4], "TARGET": [0, 0, 1, 1], "AMT_CREDIT": [1.0] * 4}
    )


def test_score_frame_applies_threshold_and_keeps_only_id_and_scores() -> None:
    scored = score_frame(_FixedModel(), _frame(), load_config())

    assert list(scored.columns) == ["SK_ID_CURR", "default_probability", "prediction"]
    assert scored["prediction"].tolist() == [
        "non_default",
        "non_default",
        "default",
        "default",
    ]


def test_summary_has_auc_only_when_a_real_target_is_given() -> None:
    scored = score_frame(_FixedModel(), _frame(), load_config())

    with_target = summarize_scores(scored, _frame()["TARGET"])
    without_target = summarize_scores(scored)

    assert with_target["roc_auc"] == 1.0
    assert "roc_auc" not in without_target
    assert without_target["positive_rate"] == 0.5
    assert without_target["rows"] == 4


def test_scoring_registry_publishes_available_stats_only() -> None:
    registry = drift_metrics.build_scoring_registry(
        {"rows": 4.0, "score_mean": 0.5, "positive_rate": 0.5}
    )

    assert registry.get_sample_value("model_batch_score_mean") == 0.5
    assert registry.get_sample_value("model_batch_rows") == 4
    assert registry.get_sample_value("model_batch_roc_auc") is None
