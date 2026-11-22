"""换基摘要：导出、校验、按共有月份拼接查询。

摘要的数学含义
==============
原判定页只回答“涨/跌/无法证实”，而换基摘要需要把若干月份的**全部可行
比例关系**移交给下一批人。设原始约束图（只含勾选版次的记录）为 ``G``，
所选月份集合为 ``S``。导出时对 ``S`` 中每一对月份 ``(a, b)`` 计算

    F(b)/F(a) ∈ [lo(a,b), hi(a,b)]

其中端点取 ``G`` 中 ``a`` 与 ``b`` 之间的**最短路**（可经过任意未移交的
月份）。把这些“最紧的”月对区间写进摘要，得到的摘要图 ``H`` 满足

    dist_H(a, b) = dist_G(a, b)   对所有 a, b ∈ S

对差分约束系统，消去其它变量后剩余的投影约束恰好就是这些两两最紧
约束，因此 ``H`` 在 ``S`` 上的可行域与 ``G`` 在 ``S`` 上的投影**完全一致**
（导出前后所选月份可同时成立的客流比例组合完全相同）。这一点由
:mod:`tests.test_summary` 用穷举/采样与逐对最短路双重校验。

矛盾与缺依据
============
* 导出：若任一所选月份所在连通分量存在负权环（记录互相矛盾），拒绝导出
  并指出冲突记录；
* 拼接：两份摘要按共有月份并入同一张图。两月无向不连通 => 缺依据；
  连通但所在分量有负权环 => 停算，并列出环上每条边来自哪份摘要、哪个
  月对。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from typing import Iterable, Optional

from .exact import (
    ConstraintEdge,
    ONE,
    component_of,
    edges_within,
    find_negative_cycle,
    fraction_decimal,
    fraction_text,
    interval_edges,
    min_products,
    pair_bounds,
    parse_fraction_text,
    record_edges,
)
from .logic import Dataset, MONTH_RE

SUMMARY_FORMAT = "bus-transfer-summary"
SUMMARY_FORMAT_VERSION = 1
#: 同口径：公布指数与真值最大偏差 1/2
SUMMARY_TOLERANCE = Fraction(1, 2)


class SummaryError(ValueError):
    """摘要结构校验失败，errors 内列出全部问题。"""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class ExportError(ValueError):
    """导出被拒绝（月份未知、存在矛盾等）。"""

    def __init__(self, message: str, *, conflicts: Optional[list[dict]] = None,
                 affected_months: Optional[list[str]] = None):
        self.message = message
        self.conflicts = conflicts or []
        self.affected_months = affected_months or []
        super().__init__(message)


@dataclass(frozen=True)
class SummaryPair:
    """``low <= F(to_month)/F(from_month) <= high``，端点均为精确有理数。"""

    from_month: str
    to_month: str
    low: Fraction
    high: Fraction

    def to_dict(self) -> dict:
        return {
            "from": self.from_month,
            "to": self.to_month,
            "low": fraction_text(self.low),
            "high": fraction_text(self.high),
        }


@dataclass
class Summary:
    summary_id: str
    name: str
    months: list[str]
    pairs: list[SummaryPair]
    created_at: str
    source: dict = field(default_factory=dict)
    tolerance: Fraction = SUMMARY_TOLERANCE

    def to_dict(self) -> dict:
        return {
            "format": SUMMARY_FORMAT,
            "format_version": SUMMARY_FORMAT_VERSION,
            "summary_id": self.summary_id,
            "name": self.name,
            "created_at": self.created_at,
            "tolerance": fraction_text(self.tolerance),
            "months": list(self.months),
            "pairs": [p.to_dict() for p in self.pairs],
            "source": dict(self.source),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def meta_dict(self) -> dict:
        return {
            "summary_id": self.summary_id,
            "name": self.name,
            "created_at": self.created_at,
            "months": list(self.months),
            "pair_count": len(self.pairs),
            "tolerance": fraction_text(self.tolerance),
            "source": dict(self.source),
        }


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #

def _unique_months(months: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in months:
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def export_summary(
    dataset: Dataset,
    versions: list[str],
    months: list[str],
    *,
    name: Optional[str] = None,
    summary_id: Optional[str] = None,
) -> Summary:
    """按所选版次与月份导出换基摘要。

    摘要仅含所选月份及其两两之间的精确比值闭区间；区间经过未移交月份
    计算，但完整保留了所选月份上的可行比例组合。
    """
    selected_versions = sorted(set(versions))
    if not selected_versions:
        raise ExportError("请至少选择一个版次")
    unknown_versions = sorted(set(selected_versions) - set(dataset.versions))
    if unknown_versions:
        raise ExportError(f"版次不存在于数据中：{'、'.join(unknown_versions)}")

    selected_months = _unique_months(months)
    if not selected_months:
        raise ExportError("请至少选择一个要移交的月份")
    bad_months = [m for m in selected_months if not MONTH_RE.match(m)]
    if bad_months:
        raise ExportError(f"月份格式应为 YYYY-MM：{'、'.join(bad_months)}")
    unknown_months = [m for m in selected_months if m not in dataset.months]
    if unknown_months:
        raise ExportError(
            f"所选月份不在已上传数据中：{'、'.join(unknown_months)}"
        )

    records = [r for r in dataset.records if r.version in set(selected_versions)]
    edges = record_edges(records)

    # 检查与任一所选月相连的分量是否存在矛盾（负权环）；按分量缓存
    conflicts: list[dict] = []
    conflict_rows: set[int] = set()
    affected: set[str] = set()
    component_cache: dict[frozenset[str], object] = {}
    month_component: dict[str, frozenset[str]] = {}
    for month in selected_months:
        comp = frozenset(component_of(month, edges))
        month_component[month] = comp
        if comp not in component_cache:
            component_cache[comp] = find_negative_cycle(edges, comp)
        cycle = component_cache[comp]
        if cycle is None:
            continue
        affected.add(month)
        for e in cycle:  # type: ignore[union-attr]
            rec = e.meta.record
            if rec is None or rec.row_no in conflict_rows:
                continue
            conflict_rows.add(rec.row_no)
            conflicts.append({
                "row_no": rec.row_no,
                "version": rec.version,
                "base_month": rec.base_month,
                "stat_month": rec.stat_month,
                "published_index": rec.published_index,
            })
    if conflicts:
        months_text = "、".join(sorted(affected))
        rows_text = "、".join(f"第 {c['row_no']} 行" for c in conflicts)
        raise ExportError(
            f"与所选月份（{months_text}）相连的源记录互相矛盾"
            f"（{rows_text} 的指数区间无法同时成立），拒绝导出",
            conflicts=conflicts,
            affected_months=sorted(affected),
        )

    # 每个所选月只跑一次最短路；a->b 的上界来自 a 的单源结果，
    # 下界来自 b 的单源结果（1 / dist_b[a]）。
    dist_cache: dict[str, dict[str, Optional[Fraction]]] = {}
    for month in selected_months:
        comp = month_component[month]
        dist_cache[month] = min_products(edges, month, comp)[0]

    pairs: list[SummaryPair] = []
    for i, a in enumerate(selected_months):
        for b in selected_months[i + 1:]:
            if b not in month_component[a]:
                continue  # 两月不相连：摘要不记录任何区间
            upper = dist_cache[a].get(b)
            rev = dist_cache[b].get(a)
            if upper is None or rev is None:  # 各自分量内应可达
                continue
            low, high = ONE / rev, upper
            if low > high:  # 防御性处理，不应出现
                raise ExportError(
                    f"内部错误：{a} 与 {b} 的比值区间下界大于上界"
                )
            if a <= b:
                pairs.append(SummaryPair(a, b, low, high))
            else:
                pairs.append(SummaryPair(b, a, ONE / high, ONE / low))

    return Summary(
        summary_id=summary_id or uuid.uuid4().hex,
        name=name or f"换基摘要 {datetime.now(timezone.utc):%Y%m%d-%H%M%S}",
        months=selected_months,
        pairs=pairs,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source={
            "versions": selected_versions,
            "record_count": len(records),
            "selected_month_count": len(selected_months),
        },
        tolerance=SUMMARY_TOLERANCE,
    )


# --------------------------------------------------------------------------- #
# 解析 / 校验
# --------------------------------------------------------------------------- #

def parse_summary_dict(data: object) -> Summary:
    """解析并校验一传的摘要对象，收集全部结构问题。"""
    if not isinstance(data, dict):
        raise SummaryError(["摘要必须是 JSON 对象"])
    errors: list[str] = []

    if data.get("format") != SUMMARY_FORMAT:
        errors.append(f"摘要 format 应为 {SUMMARY_FORMAT}")
    version = data.get("format_version")
    if version != SUMMARY_FORMAT_VERSION:
        errors.append(f"摘要 format_version 应为 {SUMMARY_FORMAT_VERSION}")
    tolerance_raw = data.get("tolerance", fraction_text(SUMMARY_TOLERANCE))
    try:
        tolerance = parse_fraction_text(tolerance_raw)
    except ValueError as exc:
        errors.append(f"tolerance 无效：{exc}")
        tolerance = SUMMARY_TOLERANCE
    else:
        if tolerance != SUMMARY_TOLERANCE:
            errors.append(
                f"非同口径摘要：tolerance 应为 {fraction_text(SUMMARY_TOLERANCE)}，"
                f"实际为 {fraction_text(tolerance)}"
            )

    months_raw = data.get("months")
    months: list[str] = []
    if not isinstance(months_raw, list) or not months_raw:
        errors.append("months 必须是非空数组")
    else:
        for m in months_raw:
            if not isinstance(m, str) or not MONTH_RE.match(m):
                errors.append(f"月份格式应为 YYYY-MM：{m!r}")
            elif m in months:
                errors.append(f"months 存在重复月份：{m}")
            else:
                months.append(m)

    pairs_raw = data.get("pairs", [])
    pairs: list[SummaryPair] = []
    if not isinstance(pairs_raw, list):
        errors.append("pairs 必须是数组")
        pairs_raw = []
    seen_pairs: set[tuple[str, str]] = set()
    for idx, item in enumerate(pairs_raw):
        label = f"pairs[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{label} 必须是对象")
            continue
        a = item.get("from")
        b = item.get("to")
        if not isinstance(a, str) or not MONTH_RE.match(a):
            errors.append(f"{label}.from 月份格式应为 YYYY-MM：{a!r}")
            continue
        if not isinstance(b, str) or not MONTH_RE.match(b):
            errors.append(f"{label}.to 月份格式应为 YYYY-MM：{b!r}")
            continue
        if a == b:
            errors.append(f"{label} 的 from 与 to 不能相同：{a}")
            continue
        if a not in months or b not in months:
            errors.append(f"{label} 引用了 months 中不存在的月份：{a} / {b}")
            continue
        low_text = item.get("low")
        high_text = item.get("high")
        try:
            low = parse_fraction_text(low_text)
        except ValueError as exc:
            errors.append(f"{label}.low 无效：{exc}")
            continue
        try:
            high = parse_fraction_text(high_text)
        except ValueError as exc:
            errors.append(f"{label}.high 无效：{exc}")
            continue
        if low > high:
            errors.append(f"{label} 的下界 {low} 大于上界 {high}")
            continue
        key = (a, b)
        if key in seen_pairs:
            errors.append(f"{label} 与前面的月对重复：({a}, {b})")
            continue
        seen_pairs.add(key)
        pairs.append(SummaryPair(a, b, low, high))

    summary_id = data.get("summary_id")
    if not isinstance(summary_id, str) or not summary_id.strip():
        summary_id = uuid.uuid4().hex
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        name = f"摘要 {summary_id[:8]}"
    created_at = data.get("created_at")
    if not isinstance(created_at, str):
        created_at = ""
    source = data.get("source")
    if not isinstance(source, dict):
        source = {}

    if errors:
        raise SummaryError(errors)
    return Summary(
        summary_id=summary_id.strip(),
        name=name.strip(),
        months=months,
        pairs=pairs,
        created_at=created_at,
        source=source,
        tolerance=tolerance,
    )


def parse_summary_text(text: str) -> Summary:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SummaryError([f"摘要不是合法 JSON：{exc.msg}（第 {exc.lineno} 行）"])
    return parse_summary_dict(data)


# --------------------------------------------------------------------------- #
# 拼接查询
# --------------------------------------------------------------------------- #

MERGE_OK = "ok"
MERGE_NO_BASIS = "no_basis"
MERGE_CONFLICT = "conflict"


@dataclass
class MergeResult:
    status: str
    start_month: str
    end_month: str
    message: str
    ratio_low: Optional[Fraction] = None
    ratio_high: Optional[Fraction] = None
    verdict: Optional[str] = None
    # 每份摘要单独能查到的范围（“原先就能查询的范围”），用于对比重算结果
    single_bounds: list[dict] = field(default_factory=list)
    # 冲突涉及的摘要与月对
    conflicts: list[dict] = field(default_factory=list)
    used_summaries: list[dict] = field(default_factory=list)
    tightening: bool = False

    def to_dict(self) -> dict:
        payload = {
            "status": self.status,
            "start_month": self.start_month,
            "end_month": self.end_month,
            "message": self.message,
            "verdict": self.verdict,
            "ratio_low": fraction_text(self.ratio_low) if self.ratio_low is not None else None,
            "ratio_high": fraction_text(self.ratio_high) if self.ratio_high is not None else None,
            "ratio_low_decimal": (
                fraction_decimal(self.ratio_low) if self.ratio_low is not None else None
            ),
            "ratio_high_decimal": (
                fraction_decimal(self.ratio_high) if self.ratio_high is not None else None
            ),
            "single_bounds": self.single_bounds,
            "conflicts": self.conflicts,
            "used_summaries": self.used_summaries,
            "tightening": self.tightening,
        }
        return payload


def _summary_edges(summary: Summary) -> list[ConstraintEdge]:
    edges: list[ConstraintEdge] = []
    for pair in summary.pairs:
        edges.extend(interval_edges(
            pair.from_month,
            pair.to_month,
            pair.low,
            pair.high,
            summary_name=summary.name,
            summary_id=summary.summary_id,
            pair=(pair.from_month, pair.to_month),
        ))
    return edges


def _verdict_from_bounds(low: Fraction, high: Fraction) -> str:
    """区间整体 > 1 必涨；整体 < 1 必跌；否则无法确定。"""
    if low > ONE:
        return "rise"
    if high < ONE:
        return "fall"
    return "unknown"


def _edge_conflict_dict(edge: ConstraintEdge) -> dict:
    meta = edge.meta
    pair = meta.pair
    if pair is None and meta.record is not None:
        pair = (meta.record.base_month, meta.record.stat_month)
    from_month, to_month = pair if pair else ("", "")
    return {
        "summary_id": meta.summary_id,
        "summary_name": meta.summary_name,
        "from_month": from_month,
        "to_month": to_month,
        "kind": meta.kind,
    }


def _dedupe_conflicts(edges: Iterable[ConstraintEdge]) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple] = set()
    for e in edges:
        item = _edge_conflict_dict(e)
        key = (
            item["summary_id"], item["from_month"], item["to_month"], item["kind"]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _single_summary_bound(
    summary: Summary, start: str, end: str
) -> dict:
    """计算“只用这一份摘要”时两月的范围，用于展示合并前后的差异。"""
    base = {
        "summary_id": summary.summary_id,
        "summary_name": summary.name,
        "status": MERGE_NO_BASIS,
        "ratio_low": None,
        "ratio_high": None,
        "message": "",
    }
    if start not in summary.months or end not in summary.months:
        base["message"] = "该摘要未同时包含这两个月份"
        return base
    edges = _summary_edges(summary)
    comp = component_of(start, edges)
    if end not in comp:
        base["message"] = "该摘要中两月不相连"
        return base
    cycle = find_negative_cycle(edges, comp)
    if cycle is not None:
        base["status"] = MERGE_CONFLICT
        base["message"] = "该摘要自身在相关分量存在矛盾"
        base["conflicts"] = _dedupe_conflicts(cycle)
        return base
    bounds = pair_bounds(edges, start, end, comp)
    if bounds is None:
        base["message"] = "该摘要中两月不相连"
        return base
    low, high = bounds
    base.update({
        "status": MERGE_OK,
        "ratio_low": fraction_text(low),
        "ratio_high": fraction_text(high),
        "ratio_low_decimal": fraction_decimal(low),
        "ratio_high_decimal": fraction_decimal(high),
    })
    return base


def merge_query(
    summaries: list[Summary], start_month: str, end_month: str
) -> MergeResult:
    """把一至两份摘要按共有月份拼接，查询止月相对起月的精确比值范围。"""
    if not summaries:
        return MergeResult(
            MERGE_NO_BASIS, start_month, end_month, "请先上传至少一份摘要"
        )
    for label, value in (("起月", start_month), ("止月", end_month)):
        if not isinstance(value, str) or not MONTH_RE.match(value):
            return MergeResult(
                MERGE_NO_BASIS, start_month, end_month,
                f"{label}格式应为 YYYY-MM",
            )

    used = [s.meta_dict() for s in summaries]
    edges: list[ConstraintEdge] = []
    for s in summaries:
        edges.extend(_summary_edges(s))
    all_months = {m for s in summaries for m in s.months}

    single_bounds = [
        _single_summary_bound(s, start_month, end_month) for s in summaries
    ]

    missing = [
        f"{label} {month} 不在已上传摘要中"
        for label, month in (("起月", start_month), ("止月", end_month))
        if month not in all_months
    ]
    if missing:
        return MergeResult(
            MERGE_NO_BASIS, start_month, end_month,
            "；".join(missing) + "，缺少查询依据。",
            single_bounds=single_bounds,
            used_summaries=used,
        )

    if start_month == end_month:
        # 同月比值恒为 1，不依赖任何月对区间
        return MergeResult(
            MERGE_OK, start_month, end_month,
            "起止为同一月份，比值恒为 1",
            ratio_low=ONE, ratio_high=ONE, verdict="unknown",
            single_bounds=single_bounds, used_summaries=used,
        )

    # 摘要声明过但没有月对区间的月份是孤立节点：会因“不连通”而判缺依据。
    comp = component_of(start_month, edges)
    if end_month not in comp:
        if len(summaries) == 1:
            detail = "该摘要中两月不相连"
        else:
            shared = sorted(set(summaries[0].months) & set(summaries[1].months))
            if not shared:
                detail = "两份摘要没有任何共有月份，无法拼接"
            else:
                detail = (
                    f"两份摘要的共有月份为 {'、'.join(shared)}，"
                    "但仍无法把起点月和终点月连起来"
                )
        return MergeResult(
            MERGE_NO_BASIS, start_month, end_month,
            f"{start_month} 与 {end_month} 在合并约束下不连通：{detail}，缺少换算依据。",
            single_bounds=single_bounds,
            used_summaries=used,
        )

    cycle = find_negative_cycle(edges_within(edges, comp), comp)
    if cycle is not None:
        conflicts = _dedupe_conflicts(cycle)
        involved = []
        for c in conflicts:
            label = c["summary_name"] or c["summary_id"]
            involved.append(f"{label} 的月对 ({c['from_month']}, {c['to_month']})")
        return MergeResult(
            MERGE_CONFLICT, start_month, end_month,
            f"{start_month} 与 {end_month} 连通，但所在分量的摘要区间无法同时成立，"
            f"停止换算。冲突涉及：{'；'.join(involved)}。",
            single_bounds=single_bounds,
            conflicts=conflicts,
            used_summaries=used,
        )

    bounds = pair_bounds(edges, start_month, end_month, comp)
    if bounds is None:  # 防御性处理
        return MergeResult(
            MERGE_NO_BASIS, start_month, end_month,
            f"{start_month} 与 {end_month} 缺少换算依据。",
            single_bounds=single_bounds,
            used_summaries=used,
        )
    low, high = bounds
    verdict = _verdict_from_bounds(low, high)

    # 判断合并结果是否比“原先可查询的范围”更紧
    tightening = False
    for sb in single_bounds:
        if sb["status"] == MERGE_OK:
            try:
                slow = parse_fraction_text(sb["ratio_low"])
                shigh = parse_fraction_text(sb["ratio_high"])
            except ValueError:
                continue
            if low > slow or high < shigh:
                tightening = True
                break
    if len(summaries) > 1 and not any(sb["status"] == MERGE_OK for sb in single_bounds):
        tightening = False

    messages = {
        "rise": "合并约束下止月客流必然高于起月（比值下界 > 1）",
        "fall": "合并约束下止月客流必然低于起月（比值上界 < 1）",
        "unknown": "合并约束下比值区间包含 1，无法确定涨跌",
    }
    return MergeResult(
        MERGE_OK, start_month, end_month, messages[verdict],
        ratio_low=low, ratio_high=high, verdict=verdict,
        single_bounds=single_bounds, used_summaries=used,
        tightening=tightening,
    )
