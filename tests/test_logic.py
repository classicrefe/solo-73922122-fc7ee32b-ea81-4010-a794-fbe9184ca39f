"""核心判定逻辑的单元测试，直接运行：python -m pytest -q"""

from __future__ import annotations

import math

import pytest

from app.logic import (
    CSVParseError,
    VERDICT_CONFLICT,
    VERDICT_FALL,
    VERDICT_NO_BASIS,
    VERDICT_RISE,
    VERDICT_UNKNOWN,
    analyze,
    parse_csv,
)

DEMO_CSV = """版号,基期月,统计月,公布指数
V2020,2020-01,2020-01,100
V2020,2020-01,2020-07,120
V2020,2020-01,2021-01,108
V2021,2021-01,2021-01,100
V2021,2021-01,2020-07,111
V2021,2021-01,2020-12,100
V2021,2021-01,2021-07,95
V2021,2021-01,2022-01,90
V2022,2022-01,2022-01,100
V2022,2022-01,2021-07,106
V2022,2022-01,2022-07,130
"""


def load(text: str):
    return parse_csv(text)


def all_versions(ds) -> list[str]:
    return ds.versions


def test_parse_valid_csv_and_metadata():
    ds = load(DEMO_CSV)
    assert len(ds.records) == 11
    assert ds.versions == ["V2020", "V2021", "V2022"]
    assert ds.version_base["V2021"] == "2021-01"
    assert "2020-07" in ds.months and "2022-07" in ds.months


def test_parse_accepts_english_headers():
    text = """version,base_month,stat_month,published_index
V1,2020-01,2020-02,103
"""
    ds = load(text)
    assert ds.records[0].version == "V1"
    assert ds.records[0].published_index == 103


def test_parse_collects_all_errors():
    text = """版号,基期月,统计月,公布指数
V1,2020/01,2020-02,-5
,2020-13,2020-02,abc
V1,2020-02,2020-03,110
"""
    with pytest.raises(CSVParseError) as ei:
        load(text)
    joined = " | ".join(ei.value.errors)
    assert "第 2 行" in joined
    assert "第 3 行" in joined
    assert "正整数" in joined
    assert "YYYY-MM" in joined


def test_parse_reports_conflicting_base_month_within_version():
    text = """版号,基期月,统计月,公布指数
V1,2020-01,2020-02,110
V1,2020-03,2020-04,120
"""
    with pytest.raises(CSVParseError) as ei:
        load(text)
    assert any("基期月冲突" in e for e in ei.value.errors)


def test_parse_missing_column():
    with pytest.raises(CSVParseError) as ei:
        load("版号,基期月,统计月\nV1,2020-01,2020-02\n")
    assert "缺少必需列" in ei.value.errors[0]


def test_parse_duplicate_rows_warn_not_fail():
    text = "版号,基期月,统计月,公布指数\nV1,2020-01,2020-01,100\nV1,2020-01,2020-01,100\n"
    ds = load(text)
    assert len(ds.records) == 1
    assert ds.warnings and "重复" in ds.warnings[0]


def test_rise_across_three_base_versions():
    ds = load(DEMO_CSV)
    res = analyze(ds, all_versions(ds), "2020-07", "2022-07")
    assert res.verdict == VERDICT_RISE
    assert res.chain, "必涨时必须给出证明链"
    # 链上每条步骤都来自真实源记录
    rows = {st.row_no for st in res.chain}
    assert rows.issubset({r.row_no for r in ds.records})
    # 月份必须首尾衔接
    assert res.chain[0].from_month == "2020-07"
    assert res.chain[-1].to_month == "2022-07"
    for a, b in zip(res.chain, res.chain[1:]):
        assert a.to_month == b.from_month
    # 连乘下界严格大于 1
    assert res.product_low > 1.0


def test_fall_reverse_direction():
    ds = load(DEMO_CSV)
    res = analyze(ds, all_versions(ds), "2022-07", "2020-07")
    assert res.verdict == VERDICT_FALL
    assert res.chain
    assert res.chain[0].from_month == "2022-07"
    assert res.chain[-1].to_month == "2020-07"
    assert res.product_high < 1.0


def test_unknown_when_interval_contains_one():
    # 同一统计月 100 指数 => 比值区间 [0.995, 1.005] 包含 1
    text = """版号,基期月,统计月,公布指数
V1,2020-01,2020-01,100
V1,2020-01,2020-02,100
"""
    ds = load(text)
    res = analyze(ds, ["V1"], "2020-01", "2020-02")
    assert res.verdict == VERDICT_UNKNOWN
    assert res.ratio_low < 1.0 < res.ratio_high


def test_no_basis_when_disconnected():
    text = """版号,基期月,统计月,公布指数
VX,2024-01,2024-01,100
VX,2024-01,2024-06,110
VY,2025-01,2025-01,100
VY,2025-01,2025-06,120
"""
    ds = load(text)
    res = analyze(ds, ["VX", "VY"], "2024-06", "2025-06")
    assert res.verdict == VERDICT_NO_BASIS
    assert not res.chain


def test_no_basis_when_month_absent():
    ds = load(DEMO_CSV)
    res = analyze(ds, all_versions(ds), "2020-07", "2030-01")
    assert res.verdict == VERDICT_NO_BASIS


def test_conflict_negative_cycle_lists_incompatible_records():
    # VA 称 F(06)/F(01) ∈ [1.995, 2.005]；VB 称 F(01)/F(06) ∈ [0.485, 0.495]
    # 即 F(06)/F(01) ∈ [1/0.495, 1/0.485] ≈ [2.020, 2.062]，与 VA 区间不相交
    text = """版号,基期月,统计月,公布指数
VA,2023-01,2023-01,100
VA,2023-01,2023-06,200
VB,2023-06,2023-06,100
VB,2023-06,2023-01,49
"""
    ds = load(text)
    res = analyze(ds, ["VA", "VB"], "2023-01", "2023-06")
    assert res.verdict == VERDICT_CONFLICT
    assert len(res.conflicts) >= 2
    # 负环必须同时包含两个版次的非平凡记录
    # （CSV 含表头：VA 指数 200 在第 3 行，VB 指数 49 在第 5 行）
    rows = {r.row_no for r in res.conflicts}
    assert {3, 5}.issubset(rows)
    assert {r.version for r in res.conflicts} == {"VA", "VB"}


def test_compatible_duplicate_measurement_not_conflict():
    # 两个版次对同一对月份给出相容区间，系统应可行，不得误报矛盾
    text = """版号,基期月,统计月,公布指数
VA,2023-01,2023-01,100
VA,2023-01,2023-06,150
VB,2023-06,2023-06,100
VB,2023-06,2023-01,67
"""
    ds = load(text)
    res = analyze(ds, ["VA", "VB"], "2023-01", "2023-06")
    assert res.verdict in (VERDICT_RISE, VERDICT_UNKNOWN)
    assert res.verdict != VERDICT_CONFLICT


def test_single_record_boundary_index():
    # 指数 200：比值区间 [1.995, 2.005]，必涨
    text = """版号,基期月,统计月,公布指数
V1,2020-01,2020-01,100
V1,2020-01,2020-02,200
"""
    ds = load(text)
    res = analyze(ds, ["V1"], "2020-01", "2020-02")
    assert res.verdict == VERDICT_RISE
    # 单条记录直接给出比值区间 [1.995, 2.005]
    assert res.product_low == pytest.approx(1.995)
    assert res.product_high == pytest.approx(2.005)
    # 反方向必跌，区间整体取倒数
    rev = analyze(ds, ["V1"], "2020-02", "2020-01")
    assert rev.verdict == VERDICT_FALL
    assert rev.product_low == pytest.approx(1.0 / 2.005)
    assert rev.product_high == pytest.approx(1.0 / 1.995)


def test_chain_products_match_bounds():
    ds = load(DEMO_CSV)
    res = analyze(ds, all_versions(ds), "2020-07", "2022-07")
    lo = math.prod(st.ratio_low for st in res.chain)
    hi = math.prod(st.ratio_high for st in res.chain)
    assert res.product_low == pytest.approx(lo, rel=1e-9)
    assert res.product_high == pytest.approx(hi, rel=1e-9)


def test_version_selection_filters_records():
    ds = load(DEMO_CSV)
    # 只选 V2020：2020-07 与 2022-07 不连通
    res = analyze(ds, ["V2020"], "2020-07", "2022-07")
    assert res.verdict == VERDICT_NO_BASIS


def test_same_start_and_end_month_is_unknown_not_rise_or_fall():
    ds = load(DEMO_CSV)
    res = analyze(ds, all_versions(ds), "2021-07", "2021-07")
    assert res.verdict == VERDICT_UNKNOWN
    assert res.ratio_low == pytest.approx(1.0)
    assert res.ratio_high == pytest.approx(1.0)
    assert res.chain == []
