"""FastAPI 网页入口：上传月报 CSV、勾选版次、选定起止月份并给出涨跌判定。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .logic import (
    CSVParseError,
    Dataset,
    VERDICT_LABELS,
    analyze,
    parse_csv,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR.parent / "samples"

app = FastAPI(title="公交客流换基判定", version="1.0.0")

# 演示用内存存储：dataset_id -> Dataset
DATASETS: dict[str, Dataset] = {}

#: 内置示例文件
SAMPLE_FILES = {
    "demo": SAMPLES_DIR / "demo_monthly_reports.csv",
    "conflict": SAMPLES_DIR / "conflict_reports.csv",
    "disconnected": SAMPLES_DIR / "disconnected_reports.csv",
}


class AnalyzeRequest(BaseModel):
    dataset_id: str
    versions: list[str]
    start_month: str
    end_month: str


def _dataset_payload(ds: Dataset, dataset_id: str) -> dict:
    return {
        "dataset_id": dataset_id,
        "versions": [
            {
                "version": v,
                "base_month": ds.version_base.get(v),
                "record_count": len(ds.version_records.get(v, [])),
            }
            for v in ds.versions
        ],
        "months": sorted(ds.months),
        "record_count": len(ds.records),
        "warnings": ds.warnings,
    }


def _record_to_dict(r) -> dict:
    return {
        "row_no": r.row_no,
        "version": r.version,
        "base_month": r.base_month,
        "stat_month": r.stat_month,
        "published_index": r.published_index,
    }


def _result_to_dict(res) -> dict:
    return {
        "verdict": res.verdict,
        "verdict_label": VERDICT_LABELS[res.verdict],
        "start_month": res.start_month,
        "end_month": res.end_month,
        "selected_versions": res.selected_versions,
        "message": res.message,
        "chain": [
            {
                "row_no": st.row_no,
                "version": st.version,
                "base_month": st.base_month,
                "stat_month": st.stat_month,
                "published_index": st.published_index,
                "from_month": st.from_month,
                "to_month": st.to_month,
                "ratio_low": round(st.ratio_low, 6),
                "ratio_high": round(st.ratio_high, 6),
                "bound_kind": st.bound_kind,
            }
            for st in res.chain
        ],
        "product_low": round(res.product_low, 6) if res.product_low is not None else None,
        "product_high": round(res.product_high, 6) if res.product_high is not None else None,
        "ratio_low": round(res.ratio_low, 6) if res.ratio_low is not None else None,
        "ratio_high": round(res.ratio_high, 6) if res.ratio_high is not None else None,
        "conflicts": [_record_to_dict(r) for r in res.conflicts],
        "used_record_count": len(res.used_records),
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/samples")
def list_samples() -> dict:
    return {
        key: {
            "filename": path.name,
            "available": path.exists(),
        }
        for key, path in SAMPLE_FILES.items()
    }


@app.post("/api/samples/{key}/load")
def load_sample(key: str) -> dict:
    if key not in SAMPLE_FILES:
        raise HTTPException(404, f"未知示例：{key}")
    path = SAMPLE_FILES[key]
    if not path.exists():
        raise HTTPException(404, f"示例文件缺失：{path.name}")
    text = path.read_text(encoding="utf-8-sig")
    try:
        ds = parse_csv(text)
    except CSVParseError as exc:
        raise HTTPException(422, detail={"errors": exc.errors})
    dataset_id = uuid.uuid4().hex
    DATASETS[dataset_id] = ds
    return _dataset_payload(ds, dataset_id)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(422, detail={"errors": ["上传文件为空"]})
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("gbk")
        except UnicodeDecodeError:
            raise HTTPException(422, detail={"errors": ["文件编码不受支持，请使用 UTF-8 或 GBK 编码的 CSV"]})
    try:
        ds = parse_csv(text)
    except CSVParseError as exc:
        return JSONResponse(status_code=422, content={"errors": exc.errors})
    dataset_id = uuid.uuid4().hex
    DATASETS[dataset_id] = ds
    return JSONResponse(_dataset_payload(ds, dataset_id))


@app.post("/api/analyze")
def analyze_endpoint(req: AnalyzeRequest) -> dict:
    ds = DATASETS.get(req.dataset_id)
    if ds is None:
        raise HTTPException(404, "数据集不存在或已过期，请重新上传")
    if not req.versions:
        raise HTTPException(422, "请至少勾选一个版次")
    unknown = sorted(set(req.versions) - set(ds.versions))
    if unknown:
        raise HTTPException(422, f"版次不存在于数据中：{'、'.join(unknown)}")
    for label, value in (("起月", req.start_month), ("止月", req.end_month)):
        if not isinstance(value, str) or len(value) != 7 or value[4] != "-":
            raise HTTPException(422, f"{label}格式应为 YYYY-MM")
    res = analyze(ds, req.versions, req.start_month, req.end_month)
    return _result_to_dict(res)


@app.get("/api/datasets/{dataset_id}/records")
def get_records(dataset_id: str, version: Optional[str] = None) -> dict:
    ds = DATASETS.get(dataset_id)
    if ds is None:
        raise HTTPException(404, "数据集不存在或已过期，请重新上传")
    records = ds.records
    if version:
        records = [r for r in records if r.version == version]
    return {"records": [_record_to_dict(r) for r in records]}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
