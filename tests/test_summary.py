"""换基摘要导出与拼接的单元测试。"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from app.exact import (
    component_of,
    fraction_decimal,
    index_ratio_bounds,
    pair_bounds,
    parse_fraction_text,
    record_edges,
)
from app.logic import parse_csv
from app.summary import (
    MERGE_CONFLICT,
    MERGE_NO_BASIS,
    MERGE_OK,
    ExportError,
    Summary,
    SummaryError,
    SummaryPair,
    export_summary,
    merge_query,
    parse_summary_dict,
    parse_summary_text,
)

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def demo():
    return parse_csv((SAMPLES / "demo_monthly_reports.csv").read_text(encoding="utf-8"))


def demo_edges(ds):
    return record_edges(ds.records)


def test_index_bounds_are_exact_rationals():
    assert index_ratio_bounds(100) == (Fraction(199, 200), Fraction(201, 200))
    assert index_ratio_bounds(200) == (Fraction(399, 200), Fraction(401, 200))


def test_export_pair_bounds_match_direct_exact_query():
    ds = demo()
    edges = demo_edges(ds)
    months = ["2020-07", "2021-01", "2021-07", "2022-07"]
    summary = export_summary(ds, ds.versions, months)
    assert summary.months == months
    assert len(summary.pairs) == 6
    for pair in summary.pairs:
        comp = component_of(pair.from_month, edges)
        direct = pair_bounds(edges, pair.from_month, pair.to_month, comp)
        assert direct is not None
        assert (pair.low, pair.high) == direct


def test_export_uses_unselected_intermediate_months():
    # 只移交 2020-07 与 2022-07；区间必须经过未移交的 2021-01 等月份，
    # 而不是退化成“没有直接记录 => 不记录”。
    ds = demo()
    summary = export_summary(ds, ds.versions, ["2020-07", "2022-07"])
    assert len(summary.pairs) == 1
    pair = summary.pairs[0]
    assert (pair.from_month, pair.to_month) == ("2020-07", "2022-07")
    # 直接跨三版连乘的精确区间（对照原逻辑的链式结果）
    assert pair.low == Fraction(46361, 44600)
    assert pair.high == Fraction(47241, 44200)
    # 参与计算的记录来自全部版次，包括未被移交的中间月份
    assert summary.source["record_count"] == 11


def test_export_skips_disconnected_month_pairs():
    ds = parse_csv((SAMPLES / "disconnected_reports.csv").read_text(encoding="utf-8"))
    summary = export_summary(ds, ds.versions, ["2024-06", "2025-06"])
    assert summary.pairs == []
    assert summary.months == ["2024-06", "2025-06"]


def test_export_rejects_conflict_connected_to_selected_month():
    ds = parse_csv((SAMPLES / "conflict_reports.csv").read_text(encoding="utf-8"))
    with pytest.raises(ExportError) as ei:
        export_summary(ds, ds.versions, ["2023-01", "2023-06"])
    assert "拒绝导出" in ei.value.message
    rows = {c["row_no"] for c in ei.value.conflicts}
    assert {3, 5} <= rows
    assert ei.value.affected_months == ["2023-01", "2023-06"]


def test_export_allows_unrelated_conflict_component():
    # 冲突分量与所选月份无关时不应阻断导出
    text = """版号,基期月,统计月,公布指数
VA,2023-01,2023-01,100
VA,2023-01,2023-06,200
VB,2023-06,2023-06,100
VB,2023-06,2023-01,49
VC,2024-01,2024-01,100
VC,2024-01,2024-06,150
"""
    ds = parse_csv(text)
    summary = export_summary(ds, ["VC"], ["2024-01", "2024-06"])
    assert len(summary.pairs) == 1
    assert summary.pairs[0].low == Fraction(299, 200)


def test_export_requires_valid_months_and_versions():
    ds = demo()
    with pytest.raises(ExportError):
        export_summary(ds, ds.versions, [])
    with pytest.raises(ExportError):
        export_summary(ds, ["NOPE"], ["2020-07"])
    with pytest.raises(ExportError):
        export_summary(ds, ds.versions, ["2030-01"])
    with pytest.raises(ExportError):
        export_summary(ds, ds.versions, ["2020-7"])


def test_summary_roundtrip_json():
    ds = demo()
    summary = export_summary(ds, ds.versions, ["2020-07", "2022-07", "2022-01"])
    parsed = parse_summary_text(summary.to_json())
    assert parsed.months == summary.months
    assert [(p.from_month, p.to_month, p.low, p.high) for p in parsed.pairs] == [
        (p.from_month, p.to_month, p.low, p.high) for p in summary.pairs
    ]


def _summary_payload(**overrides) -> dict:
    base = {
        "format": "bus-transfer-summary",
        "format_version": 1,
        "summary_id": "s1",
        "name": "测试摘要",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tolerance": "1/2",
        "months": ["2020-01", "2020-02"],
        "pairs": [
            {"from": "2020-01", "to": "2020-02", "low": "199/200", "high": "201/200"}
        ],
    }
    base.update(overrides)
    return base


def test_parse_summary_accepts_valid_payload():
    summary = parse_summary_dict(_summary_payload())
    assert summary.pairs[0].low == Fraction(199, 200)


@pytest.mark.parametrize("field,value,needle", [
    ("format", "other", "format"),
    ("format_version", 2, "format_version"),
    ("tolerance", "1/10", "非同口径"),
    ("months", [], "months"),
    ("months", ["2020-13"], "YYYY-MM"),
])
def test_parse_summary_rejects_bad_fields(field, value, needle):
    with pytest.raises(SummaryError) as ei:
        parse_summary_dict(_summary_payload(**{field: value}))
    assert any(needle in e for e in ei.value.errors)


def test_parse_summary_rejects_bad_pairs():
    bad = _summary_payload(pairs=[
        {"from": "2020-01", "to": "2020-03", "low": "1", "high": "2"},
        {"from": "2020-01", "to": "2020-01", "low": "1", "high": "2"},
        {"from": "2020-01", "to": "2020-02", "low": "3", "high": "2"},
        {"from": "2020-01", "to": "2020-02", "low": "0.5", "high": "2"},
        {"from": "2020-01", "to": "2020-02", "low": "1/0", "high": "2"},
    ])
    with pytest.raises(SummaryError) as ei:
        parse_summary_dict(bad)
    joined = " | ".join(ei.value.errors)
    assert "months 中不存在" in joined
    assert "不能相同" in joined
    assert "下界" in joined
    assert "整数/整数" in joined
    assert "分母不能为 0" in joined


def test_parse_fraction_rejects_floats():
    with pytest.raises(ValueError):
        parse_fraction_text("1.23")
    with pytest.raises(ValueError):
        parse_fraction_text(1.23)


def test_merge_two_summaries_via_shared_month_recomputes_over_merged_constraints():
    # 拼接结果必须等于“合并后的约束图”上的精确范围（而不是任何一份单独的结果）。
    from app.summary import _summary_edges

    ds = demo()
    first = export_summary(ds, ds.versions, ["2020-07", "2021-01", "2021-07"], name="A")
    second = export_summary(ds, ds.versions, ["2021-07", "2022-01", "2022-07"], name="B")
    result = merge_query([first, second], "2020-07", "2022-07")
    assert result.status == MERGE_OK
    merged_edges = _summary_edges(first) + _summary_edges(second)
    direct = pair_bounds(merged_edges, "2020-07", "2022-07")
    assert (result.ratio_low, result.ratio_high) == direct
    # 合并约束是原图投影的松弛：区间应包含原图上的精确区间
    original = pair_bounds(demo_edges(ds), "2020-07", "2022-07")
    assert result.ratio_low <= original[0] <= original[1] <= result.ratio_high


def test_single_summary_preserves_feasible_ratio_combinations():
    # 导出后，所选月份两两之间的最紧区间必须与原图完全一致
    # （差分约束系统的最小网络性质：选月上的可行域投影不变）。
    from app.summary import _summary_edges

    ds = demo()
    edges = demo_edges(ds)
    months = ["2020-01", "2020-07", "2021-01", "2021-07", "2022-01", "2022-07"]
    summary = export_summary(ds, ds.versions, months)
    summary_edges = _summary_edges(summary)
    for a in months:
        for b in months:
            if a == b:
                continue
            want = pair_bounds(edges, a, b)
            got = pair_bounds(summary_edges, a, b)
            assert got == want, (a, b, got, want)
    # 每个连通分量内的月对都被完整记录
    assert len(summary.pairs) == len(months) * (len(months) - 1) // 2


def test_merge_no_basis_without_shared_month():
    ds = demo()
    a = export_summary(ds, ds.versions, ["2020-01", "2020-07"])
    b = export_summary(ds, ds.versions, ["2022-01", "2022-07"])
    result = merge_query([a, b], "2020-07", "2022-07")
    assert result.status == MERGE_NO_BASIS
    assert "共有月份" in result.message


def test_merge_no_basis_when_month_absent():
    ds = demo()
    a = export_summary(ds, ds.versions, ["2020-07", "2021-01"])
    result = merge_query([a], "2020-07", "2030-01")
    assert result.status == MERGE_NO_BASIS
    assert "不在已上传摘要中" in result.message


def test_merge_conflict_reports_summaries_and_pairs():
    ds = demo()
    good = export_summary(ds, ds.versions, ["2020-07", "2021-01", "2021-07"], name="正常")
    bad = Summary(
        summary_id="bad",
        name="异常摘要",
        months=["2021-01", "2021-07"],
        pairs=[SummaryPair("2021-01", "2021-07", Fraction(19, 10), Fraction(2, 1))],
        created_at="",
    )
    result = merge_query([good, bad], "2020-07", "2021-07")
    assert result.status == MERGE_CONFLICT
    assert "停止换算" in result.message
    names = {c["summary_name"] for c in result.conflicts}
    assert names == {"正常", "异常摘要"}
    pairs = {(c["from_month"], c["to_month"]) for c in result.conflicts}
    assert ("2021-01", "2021-07") in pairs


def test_merge_conflict_cycle_within_one_summary():
    bad = Summary(
        summary_id="bad",
        name="自相矛盾",
        months=["2020-01", "2020-02", "2020-03"],
        pairs=[
            SummaryPair("2020-01", "2020-02", Fraction(2), Fraction(2)),
            SummaryPair("2020-02", "2020-03", Fraction(2), Fraction(2)),
            SummaryPair("2020-01", "2020-03", Fraction(1, 2), Fraction(1, 2)),
        ],
        created_at="",
    )
    result = merge_query([bad], "2020-01", "2020-03")
    assert result.status == MERGE_CONFLICT
    assert result.conflicts


def test_merge_recomputes_previously_queryable_range():
    # 摘要 A 单独可查 2020-07→2022-07，但区间较宽；摘要 B 经共有月份补上更紧
    # 的约束后，合并结果必须更紧（“原先就能查询的范围也要重算”）。
    wide = Summary(
        summary_id="wide",
        name="宽区间摘要",
        months=["2020-07", "2021-07", "2022-07"],
        pairs=[
            SummaryPair("2020-07", "2021-07", Fraction(1), Fraction(3)),
            SummaryPair("2021-07", "2022-07", Fraction(1), Fraction(3)),
            SummaryPair("2020-07", "2022-07", Fraction(1), Fraction(9)),
        ],
        created_at="",
    )
    tight = Summary(
        summary_id="tight",
        name="紧区间摘要",
        months=["2021-07", "2022-07"],
        pairs=[SummaryPair("2021-07", "2022-07", Fraction(2), Fraction(2))],
        created_at="",
    )
    alone = merge_query([wide], "2020-07", "2022-07")
    assert alone.status == MERGE_OK
    merged = merge_query([wide, tight], "2020-07", "2022-07")
    assert merged.status == MERGE_OK
    assert merged.ratio_high < alone.ratio_high
    assert merged.tightening is True
    # 报告里保留“原先就能查询的范围”
    previous = {b["summary_id"]: b for b in merged.single_bounds}
    assert previous["wide"]["status"] == MERGE_OK
    assert parse_fraction_text(previous["wide"]["ratio_high"]) == Fraction(9)


def test_merge_detects_exact_equality_cycle_as_consistent():
    # 恰为 1 的环是相容边界，不得误报矛盾
    summary = Summary(
        summary_id="eq",
        name="边界相容",
        months=["2020-01", "2020-02", "2020-03"],
        pairs=[
            SummaryPair("2020-01", "2020-02", Fraction(2), Fraction(2)),
            SummaryPair("2020-02", "2020-03", Fraction(2), Fraction(2)),
            SummaryPair("2020-01", "2020-03", Fraction(4), Fraction(4)),
        ],
        created_at="",
    )
    result = merge_query([summary], "2020-01", "2020-03")
    assert result.status == MERGE_OK
    assert (result.ratio_low, result.ratio_high) == (Fraction(4), Fraction(4))


def test_merge_validates_month_format():
    summary = export_summary(demo(), demo().versions, ["2020-07", "2021-01"])
    result = merge_query([summary], "2020-7", "2021-01")
    assert result.status == MERGE_NO_BASIS


def test_fraction_decimal_is_display_only():
    assert fraction_decimal(Fraction(1, 3)) == "0.333333333333"


def test_merge_same_month_is_one():
    summary = export_summary(demo(), demo().versions, ["2020-07", "2021-01"])
    result = merge_query([summary], "2020-07", "2020-07")
    assert result.status == MERGE_OK
    assert (result.ratio_low, result.ratio_high) == (Fraction(1), Fraction(1))


def test_export_accepts_unordered_and_duplicate_months():
    ds = demo()
    summary = export_summary(
        ds, ds.versions, ["2022-07", "2021-01", "2020-07", "2021-01"]
    )
    assert summary.months == ["2022-07", "2021-01", "2020-07"]
    assert len(summary.pairs) == 3
    for pair in summary.pairs:
        assert pair.from_month < pair.to_month


def test_export_rejects_conflict_elsewhere_in_same_component():
    # 冲突不是发生在这对所选月之间，但与所选月同属一个连通分量，
    # 因此“与所选月相连的源记录有矛盾”，必须拒绝导出。
    text = """版号,基期月,统计月,公布指数
VA,2023-01,2023-01,100
VA,2023-01,2023-03,105
VA,2023-01,2023-06,200
VB,2023-06,2023-06,100
VB,2023-06,2023-01,49
"""
    ds = parse_csv(text)
    with pytest.raises(ExportError) as ei:
        export_summary(ds, ds.versions, ["2023-01", "2023-03"])
    assert "拒绝导出" in ei.value.message
    assert ei.value.affected_months == ["2023-01", "2023-03"]
