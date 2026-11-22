"""FastAPI 接口测试：上传、示例载入、判定、错误处理全流程。"""

from __future__ import annotations

import json
from pathlib import Path

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


# --------------------------------------------------------------------------- #
# 换基摘要：导出 / 上传 / 拼接查询
# --------------------------------------------------------------------------- #

DEMO_VERSIONS = ["V2020", "V2021", "V2022"]


def load_demo_dataset() -> str:
    return client.post("/api/samples/demo/load").json()["dataset_id"]


def test_export_endpoint_returns_exact_pairs_and_download():
    dataset_id = load_demo_dataset()
    resp = client.post(
        "/api/summaries/export",
        json={
            "dataset_id": dataset_id,
            "versions": DEMO_VERSIONS,
            "months": ["2020-07", "2021-07", "2022-07"],
            "name": "测试摘要",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["format"] == "bus-transfer-summary"
    assert payload["months"] == ["2020-07", "2021-07", "2022-07"]
    assert len(payload["pairs"]) == 3
    for pair in payload["pairs"]:
        assert "/" in pair["low"] and "/" in pair["high"]  # 精确有理数
    cross = next(
        p for p in payload["pairs"]
        if (p["from"], p["to"]) == ("2020-07", "2022-07")
    )
    assert cross["low"] == "46361/44600"
    assert cross["high"] == "47241/44200"


def test_export_endpoint_rejects_conflicting_records():
    dataset_id = client.post("/api/samples/conflict/load").json()["dataset_id"]
    resp = client.post(
        "/api/summaries/export",
        json={
            "dataset_id": dataset_id,
            "versions": ["VA", "VB"],
            "months": ["2023-01", "2023-06"],
        },
    )
    assert resp.status_code == 422
    payload = resp.json()
    assert "拒绝导出" in payload["error"]
    assert {c["row_no"] for c in payload["conflicts"]} >= {3, 5}


def test_export_endpoint_validation():
    dataset_id = load_demo_dataset()
    resp = client.post(
        "/api/summaries/export",
        json={"dataset_id": dataset_id, "versions": [], "months": ["2020-07"]},
    )
    assert resp.status_code == 422
    resp = client.post(
        "/api/summaries/export",
        json={"dataset_id": dataset_id, "versions": DEMO_VERSIONS, "months": []},
    )
    assert resp.status_code == 422
    resp = client.post(
        "/api/summaries/export",
        json={"dataset_id": "nope", "versions": DEMO_VERSIONS, "months": ["2020-07"]},
    )
    assert resp.status_code == 404


def _export(dataset_id: str, months: list[str], name: str) -> dict:
    resp = client.post(
        "/api/summaries/export",
        json={
            "dataset_id": dataset_id,
            "versions": DEMO_VERSIONS,
            "months": months,
            "name": name,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_upload_register_and_query_merged_summaries():
    dataset_id = load_demo_dataset()
    first = _export(dataset_id, ["2020-07", "2021-01", "2021-07"], "A")
    second = _export(dataset_id, ["2021-07", "2022-01", "2022-07"], "B")

    up = client.post(
        "/api/summaries/upload",
        files={"file": ("a.json", first_bytes := json.dumps(first).encode(), "application/json")},
    )
    assert up.status_code == 200, up.text
    assert first_bytes
    reg = client.post("/api/summaries/register", json=second)
    assert reg.status_code == 200
    ids = [up.json()["summary_id"], reg.json()["summary_id"]]

    listed = client.get("/api/summaries").json()
    assert listed["max_for_query"] == 2
    assert {s["summary_id"] for s in listed["summaries"]} >= set(ids)

    resp = client.post(
        "/api/summaries/query",
        json={"summary_ids": ids, "start_month": "2020-07", "end_month": "2022-07"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["verdict"] == "rise"
    assert payload["ratio_low"] == "16317/15833"
    assert len(payload["single_bounds"]) == 2


def test_query_reports_no_basis_for_unconnected_summaries():
    client.post("/api/summary-samples/demo_first/load")
    client.post("/api/summary-samples/demo_far/load")
    resp = client.post(
        "/api/summaries/query",
        json={
            "summary_ids": ["demo-first", "demo-far"],
            "start_month": "2020-07",
            "end_month": "2024-06",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "no_basis"
    assert "共有月份" in payload["message"]


def test_query_stops_on_conflict_and_names_summaries_and_pairs():
    client.post("/api/summary-samples/demo_first/load")
    client.post("/api/summary-samples/demo_conflict/load")
    resp = client.post(
        "/api/summaries/query",
        json={
            "summary_ids": ["demo-first", "demo-conflict"],
            "start_month": "2020-07",
            "end_month": "2021-07",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "conflict"
    assert "停止换算" in payload["message"]
    names = {c["summary_name"] for c in payload["conflicts"]}
    assert names == {"2020—2021 批次摘要", "口径异常的 2021 摘要"}
    assert ("2021-01", "2021-07") in {
        (c["from_month"], c["to_month"]) for c in payload["conflicts"]
    }


def test_query_requires_one_or_two_summaries():
    resp = client.post(
        "/api/summaries/query",
        json={"summary_ids": [], "start_month": "2020-07", "end_month": "2021-07"},
    )
    assert resp.status_code == 422
    resp = client.post(
        "/api/summaries/query",
        json={"summary_ids": ["no-such-summary"], "start_month": "2020-07", "end_month": "2021-07"},
    )
    assert resp.status_code == 404


def test_upload_summary_rejects_bad_json_and_bad_format():
    resp = client.post(
        "/api/summaries/upload",
        files={"file": ("bad.json", b"{not json", "application/json")},
    )
    assert resp.status_code == 422
    assert "errors" in resp.json()

    bad = {"format": "wrong", "format_version": 1}
    resp = client.post(
        "/api/summaries/upload",
        files={"file": ("bad.json", json.dumps(bad).encode(), "application/json")},
    )
    assert resp.status_code == 422
    assert any("format" in e for e in resp.json()["errors"])


def test_summary_samples_present():
    listing = client.get("/api/summary-samples").json()
    assert set(listing) == {"demo_first", "demo_second", "demo_conflict", "demo_far"}
    assert all(item["available"] for item in listing.values())


def test_delete_summary():
    client.post("/api/summary-samples/demo_first/load")
    assert client.delete("/api/summaries/demo-first").status_code == 200
    assert client.delete("/api/summaries/demo-first").status_code == 404
