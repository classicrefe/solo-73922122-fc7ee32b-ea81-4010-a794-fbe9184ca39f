"""FastAPI 接口测试：上传、示例载入、判定、错误处理全流程。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import SAMPLE_FILES, app

client = TestClient(app)

DEMO = (Path(__file__).resolve().parent.parent / "samples" / "demo_monthly_reports.csv").read_text(
    encoding="utf-8"
)


def upload_csv(text: str, filename: str = "r.csv"):
    return client.post(
        "/api/upload",
        files={"file": (filename, text.encode("utf-8"), "text/csv")},
    )


def test_health_and_index():
    assert client.get("/api/health").json() == {"status": "ok"}
    resp = client.get("/")
    assert resp.status_code == 200
    assert "公交客流" in resp.text


def test_upload_and_analyze_rise_flow():
    resp = upload_csv(DEMO)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["record_count"] == 11
    assert len(payload["versions"]) == 3

    result = client.post(
        "/api/analyze",
        json={
            "dataset_id": payload["dataset_id"],
            "versions": ["V2020", "V2021", "V2022"],
            "start_month": "2020-07",
            "end_month": "2022-07",
        },
    ).json()
    assert result["verdict"] == "rise"
    assert result["verdict_label"] == "必涨"
    assert len(result["chain"]) >= 2
    assert result["product_low"] > 1


def test_upload_bad_csv_returns_422_with_errors():
    resp = upload_csv("foo,bar\n1,2\n")
    assert resp.status_code == 422
    assert "errors" in resp.json()


def test_sample_endpoints_cover_all_verdicts():
    # 矛盾示例
    key = "conflict"
    ds = client.post(f"/api/samples/{key}/load").json()
    result = client.post(
        "/api/analyze",
        json={
            "dataset_id": ds["dataset_id"],
            "versions": ["VA", "VB"],
            "start_month": "2023-01",
            "end_month": "2023-06",
        },
    ).json()
    assert result["verdict"] == "conflict"
    assert len(result["conflicts"]) >= 2

    # 缺依据示例
    ds2 = client.post("/api/samples/disconnected/load").json()
    result2 = client.post(
        "/api/analyze",
        json={
            "dataset_id": ds2["dataset_id"],
            "versions": ["VX", "VY"],
            "start_month": "2024-06",
            "end_month": "2025-06",
        },
    ).json()
    assert result2["verdict"] == "no_basis"


def test_unknown_dataset_id_404():
    resp = client.post(
        "/api/analyze",
        json={
            "dataset_id": "nope",
            "versions": ["V1"],
            "start_month": "2020-01",
            "end_month": "2020-02",
        },
    )
    assert resp.status_code == 404


def test_empty_version_selection_rejected():
    payload = upload_csv(DEMO).json()
    resp = client.post(
        "/api/analyze",
        json={
            "dataset_id": payload["dataset_id"],
            "versions": [],
            "start_month": "2020-07",
            "end_month": "2022-07",
        },
    )
    assert resp.status_code == 422


def test_records_endpoint():
    payload = upload_csv(DEMO).json()
    resp = client.get(f"/api/datasets/{payload['dataset_id']}/records", params={"version": "V2022"})
    assert resp.status_code == 200
    assert all(r["version"] == "V2022" for r in resp.json()["records"])


def test_all_sample_files_present():
    for key, path in SAMPLE_FILES.items():
        assert path.exists(), f"示例 {key} 缺失：{path}"
