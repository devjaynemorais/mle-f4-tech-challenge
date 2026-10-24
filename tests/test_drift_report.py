from pathlib import Path

import pandas as pd
import pytest

from src.drift.report import generate_report, load_feature_periods


def _periods(rows: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = range(rows)
    reference = pd.DataFrame(
        {
            "SK_ID_CURR": [900_000 + value for value in index],
            "TARGET": [value % 2 for value in index],
            "AMT_INCOME_TOTAL": [1000.0 + value for value in index],
            "CODE_GENDER": ["F" if value % 2 else "M" for value in index],
        }
    )
    current = reference.copy()
    current["SK_ID_CURR"] += 1_000_000_000
    current["AMT_INCOME_TOTAL"] += 500
    current["CODE_GENDER"] = "F"
    return reference, current


def test_load_feature_periods_drops_id_and_target_for_130_features(tmp_path) -> None:
    reference, current = _periods()
    reference_features = pd.DataFrame(
        {f"FEATURE_{index:03d}": [index + 1.0] * 80 for index in range(128)}
    )
    current_features = pd.DataFrame(
        {f"FEATURE_{index:03d}": [index + 2.0] * 80 for index in range(128)}
    )
    reference = pd.concat([reference, reference_features], axis=1)
    current = pd.concat([current, current_features], axis=1)
    reference_path, current_path = tmp_path / "stable.csv", tmp_path / "drift.csv"
    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    loaded_reference, loaded_current = load_feature_periods(
        reference_path, current_path, "SK_ID_CURR", "TARGET"
    )

    assert loaded_reference.shape == loaded_current.shape == (80, 130)
    assert loaded_reference.columns.tolist() == loaded_current.columns.tolist()
    assert "SK_ID_CURR" not in loaded_reference
    assert "TARGET" not in loaded_reference


def test_load_feature_periods_rejects_missing_file(tmp_path) -> None:
    reference, current = _periods()
    reference_path = tmp_path / "stable.csv"
    reference.to_csv(reference_path, index=False)

    with pytest.raises(FileNotFoundError):
        load_feature_periods(
            reference_path, tmp_path / "missing.csv", "SK_ID_CURR", "TARGET"
        )


def test_load_feature_periods_rejects_incompatible_schema(tmp_path) -> None:
    reference, current = _periods()
    current = current.rename(columns={"AMT_INCOME_TOTAL": "INCOME"})
    reference_path, current_path = tmp_path / "stable.csv", tmp_path / "drift.csv"
    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    with pytest.raises(ValueError, match="esquema"):
        load_feature_periods(reference_path, current_path, "SK_ID_CURR", "TARGET")


def test_load_feature_periods_rejects_empty_period(tmp_path) -> None:
    reference, current = _periods()
    reference_path, current_path = tmp_path / "stable.csv", tmp_path / "drift.csv"
    reference.iloc[:0].to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    with pytest.raises(ValueError, match="vazio"):
        load_feature_periods(reference_path, current_path, "SK_ID_CURR", "TARGET")


def test_load_feature_periods_requires_at_least_one_feature(tmp_path) -> None:
    period = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1]})
    reference_path, current_path = tmp_path / "stable.csv", tmp_path / "drift.csv"
    period.to_csv(reference_path, index=False)
    period.to_csv(current_path, index=False)

    with pytest.raises(ValueError, match="features"):
        load_feature_periods(reference_path, current_path, "SK_ID_CURR", "TARGET")


def test_generate_report_writes_evidently_html_and_per_feature_csv(tmp_path) -> None:
    reference, current = _periods()
    reference_path, current_path = tmp_path / "stable.csv", tmp_path / "drift.csv"
    html_path, summary_path = tmp_path / "report.html", tmp_path / "summary.csv"
    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    report = generate_report(
        reference_path,
        current_path,
        html_path,
        summary_path,
        "SK_ID_CURR",
        "TARGET",
        alpha=0.05,
        psi_threshold=0.25,
        n_bins=10,
    )

    summary = pd.read_csv(summary_path)
    assert report.n_features_total == 2
    assert set(report.drifted_features) == {"AMT_INCOME_TOTAL", "CODE_GENDER"}
    assert html_path.is_file() and html_path.stat().st_size > 0
    assert summary.feature.unique().tolist() == ["AMT_INCOME_TOTAL", "CODE_GENDER"]
    methods = summary.groupby("feature")["method"].apply(set).to_dict()
    assert methods == {"AMT_INCOME_TOTAL": {"ks", "psi"}, "CODE_GENDER": {"chi2"}}
    assert summary.columns.tolist() == [
        "feature",
        "method",
        "statistic",
        "p_value",
        "threshold",
        "drift_detected",
    ]
    assert not {"SK_ID_CURR", "TARGET"}.intersection(summary.columns)
    assert "900000" not in summary_path.read_text(encoding="utf-8")


def test_generate_report_keeps_reserved_features_and_all_null_numeric(
    tmp_path, monkeypatch
) -> None:
    reference, current = _periods()
    reference["origem"] = ["branch" if value % 2 else "web" for value in range(80)]
    current["origem"] = "web"
    reference["inadimplente"] = [value % 2 for value in range(80)]
    current["inadimplente"] = [value % 2 for value in range(80)]
    current["AMT_INCOME_TOTAL"] = float("nan")
    reference_path, current_path = tmp_path / "stable.csv", tmp_path / "drift.csv"
    html_path, summary_path = tmp_path / "report.html", tmp_path / "summary.csv"
    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)
    monkeypatch.setattr(
        "src.drift.report._save_evidently_report",
        lambda ref, cur, path: path.write_text("<html></html>", encoding="utf-8"),
    )

    report = generate_report(
        reference_path,
        current_path,
        html_path,
        summary_path,
        "SK_ID_CURR",
        "TARGET",
        alpha=0.05,
        psi_threshold=0.25,
        n_bins=10,
    )

    summary = pd.read_csv(summary_path)
    assert report.n_features_total == 4
    assert set(summary.feature) == {
        "AMT_INCOME_TOTAL",
        "CODE_GENDER",
        "origem",
        "inadimplente",
    }
    income_ks = summary.query("feature == 'AMT_INCOME_TOTAL' and method == 'ks'")
    income_psi = summary.query("feature == 'AMT_INCOME_TOTAL' and method == 'psi'")
    assert len(income_ks) == len(income_psi) == 1
    assert pd.isna(income_ks.iloc[0].statistic)
    assert income_psi.iloc[0].drift_detected


def test_evidently_receives_stable_reference_and_drifted_current(
    tmp_path, monkeypatch
) -> None:
    from src.drift import report as report_module

    reference, current = _periods(rows=16)
    received = {}

    class Snapshot:
        def save_html(self, path: str) -> None:
            Path(path).write_text("<html></html>", encoding="utf-8")

    class FakeReport:
        def __init__(self, presets) -> None:
            received["presets"] = presets

        def run(self, **kwargs):
            received.update(kwargs)
            return Snapshot()

    monkeypatch.setattr(report_module, "EvidentlyReport", FakeReport)
    monkeypatch.setattr(report_module, "DataDriftPreset", lambda: "data-drift")
    html_path = tmp_path / "report.html"

    report_module._save_evidently_report(reference, current, html_path)

    assert received["presets"] == ["data-drift"]
    assert received["current_data"] is current
    assert received["reference_data"] is reference
    assert html_path.is_file()
