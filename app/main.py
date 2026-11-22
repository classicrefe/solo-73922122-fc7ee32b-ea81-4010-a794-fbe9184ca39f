"""FastAPI 网页入口：上传月报 CSV、勾选版次、选定起止月份并给出涨跌判定，
并支持导出换基摘要与按共有月份拼接摘要查询。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
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
from .summary import (
    ExportError,
    Summary,
    SummaryError,
    export_summary,
    merge_query,
    parse_summary_dict,
    parse_summary_text,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR.parent / "samples"

app = FastAPI(title="公交客流换基判定", version="1.1.0")

# 演示用内存存储：dataset_id -> Dataset
DATASETS: dict[str, Dataset] = {}
# 演示用内存存储：summary_id -> Summary（接收方上传的摘要）
SUMMARIES: dict[str, Summary] = {}

MAX_SUMMARIES = 2

#: 内置示例文件
SAMPLE_FILES = {
    "demo": SAMPLES_DIR / "demo_monthly_reports.csv",
    "conflict": SAMPLES_DIR / "conflict_reports.csv",
    "disconnected": SAMPLES_DIR / "disconnected_reports.csv",
}
#: 内置摘要示例：由 demo 示例按不同月份切分导出
SUMMARY_SAMPLE_FILES = {
    "demo_first": SAMPLES_DIR / "summary_demo_first.json",
    "demo_second": SAMPLES_DIR / "summary_demo_second.json",
    "demo_conflict": SAMPLES_DIR / "summary_demo_conflict.json",
    "demo_far": SAMPLES_DIR / "summary_demo_far.json",
}


class AnalyzeRequest(BaseModel):
    dataset_id: str
    versions: list[str]
    start_month: str
    end_month: str


class ExportRequest(BaseModel):
    dataset_id: str
    versions: list[str]
    months: list[str]
    name: Optional[str] = None


class SummaryQueryRequest(BaseModel):
    summary_ids: list[str]
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


# --------------------------------------------------------------------------- #
# 换基摘要：导出 / 上传 / 拼接查询
# --------------------------------------------------------------------------- #

@app.post("/api/summaries/export")
def export_summary_endpoint(req: ExportRequest) -> dict:
    ds = DATASETS.get(req.dataset_id)
    if ds is None:
        raise HTTPException(404, "数据集不存在或已过期，请重新上传")
    try:
        summary = export_summary(
            ds, req.versions, req.months, name=(req.name or "").strip() or None
        )
    except ExportError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "error": exc.message,
                "conflicts": exc.conflicts,
                "affected_months": exc.affected_months,
            },
        )
    return summary.to_dict()


@app.post("/api/summaries/upload")
async def upload_summary(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(422, detail={"errors": ["上传文件为空"]})
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, detail={"errors": ["摘要需为 UTF-8 编码的 JSON"]})
    try:
        summary = parse_summary_text(text)
    except SummaryError as exc:
        return JSONResponse(status_code=422, content={"errors": exc.errors})
    if len(SUMMARIES) >= 100:
        # 演示用途：只保留最近的 100 份，避免内存无限增长
        SUMMARIES.pop(next(iter(SUMMARIES)))
    SUMMARIES[summary.summary_id] = summary
    return JSONResponse({"summary_id": summary.summary_id, "summary": summary.meta_dict()})


@app.post("/api/summaries/register")
def register_summary(payload: dict) -> dict:
    """直接提交摘要 JSON 对象（供示例按钮与脚本使用）。"""
    try:
        summary = parse_summary_dict(payload)
    except SummaryError as exc:
        raise HTTPException(422, detail={"errors": exc.errors})
    SUMMARIES[summary.summary_id] = summary
    return {"summary_id": summary.summary_id, "summary": summary.meta_dict()}


@app.get("/api/summaries")
def list_summaries() -> dict:
    return {
        "summaries": [
            s.meta_dict() for s in SUMMARIES.values()
        ],
        "max_for_query": MAX_SUMMARIES,
    }


@app.delete("/api/summaries/{summary_id}")
def delete_summary(summary_id: str) -> dict:
    if summary_id not in SUMMARIES:
        raise HTTPException(404, "摘要不存在")
    del SUMMARIES[summary_id]
    return {"deleted": summary_id}


@app.post("/api/summaries/query")
def query_summaries(req: SummaryQueryRequest) -> dict:
    ids: list[str] = []
    for sid in req.summary_ids:
        if sid not in ids:
            ids.append(sid)
    if not 1 <= len(ids) <= MAX_SUMMARIES:
        raise HTTPException(
            422, f"请选择一至{MAX_SUMMARIES}份摘要进行拼接查询"
        )
    missing = [sid for sid in ids if sid not in SUMMARIES]
    if missing:
        raise HTTPException(404, f"摘要不存在或已过期：{'、'.join(missing)}")
    selected = [SUMMARIES[sid] for sid in ids]
    result = merge_query(selected, req.start_month, req.end_month)
    return result.to_dict()


@app.get("/api/summary-samples")
def list_summary_samples() -> dict:
    return {
        key: {"filename": path.name, "available": path.exists()}
        for key, path in SUMMARY_SAMPLE_FILES.items()
    }


@app.post("/api/summary-samples/{key}/load")
def load_summary_sample(key: str) -> dict:
    if key not in SUMMARY_SAMPLE_FILES:
        raise HTTPException(404, f"未知摘要示例：{key}")
    path = SUMMARY_SAMPLE_FILES[key]
    if not path.exists():
        raise HTTPException(404, f"示例文件缺失：{path.name}")
    try:
        summary = parse_summary_text(path.read_text(encoding="utf-8-sig"))
    except SummaryError as exc:
        raise HTTPException(422, detail={"errors": exc.errors})
    SUMMARIES[summary.summary_id] = summary
    return {"summary_id": summary.summary_id, "summary": summary.meta_dict()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
