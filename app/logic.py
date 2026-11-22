"""公交客流月报换基判定核心逻辑。

数据模型
========
每条月报记录（CSV 一行）描述：某版号 ``v`` 以基期月 ``b`` 为 100，
统计月 ``m`` 的公布指数为正整数 ``I``。指数真值记为 ``I*``，满足
``|I* - I| <= 0.5``（题目规定最多相差 0.5）。

设月 ``m`` 的真实客流为 ``F(m)``，由指数定义 ``I* = 100 * F(m)/F(b)``，
因此对每条记录有：

    (I - 0.5) / 100 <= F(m) / F(b) <= (I + 0.5) / 100

取自然对数后成为关于变量 ``x_m = ln F(m)`` 的差分约束：

    ln((I-0.5)/100) <= x_m - x_b <= ln((I+0.5)/100)

这是标准的差分约束系统：
* 边 b -> m，权重 w_hi = ln((I+0.5)/100)，表示 x_m - x_b <= w_hi；
* 边 m -> b，权重 w_lo = ln(100/(I-0.5))，表示 x_b - x_m <= w_lo。

对起月 s、止月 e，令 d = x_e - x_s = ln(F(e)/F(s))：
* 最短距离 s -> e 给出 d 的上界 U（d <= U）；
* 最短距离 e -> s 给出 -d 的上界，即 d 的下界 L = -dist(e, s)。

判定：
* U < 0  => d 恒为负 => F(e) < F(s) => 必跌；
* L > 0  => d 恒为正 => F(e) > F(s) => 必涨；
* 否则可行区间包含 0 => 无法证实涨跌；
* 系统存在负权环 => 相关记录互相矛盾，暂停判定。
* s 与 e 不连通 => 两月缺少换算关系 => 缺依据。

全部计算使用纯标准库，方便容器内运行与单元测试。
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from typing import Optional

#: 公布指数与真值的最大偏差
INDEX_TOLERANCE = 0.5

MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
INT_RE = re.compile(r"^\d+$")

#: 允许的表头（中英文均可）
HEADER_ALIASES = {
    "版号": "version",
    "版本": "version",
    "版次": "version",
    "version": "version",
    "版": "version",
    "基期月": "base_month",
    "基期": "base_month",
    "base_month": "base_month",
    "basemonth": "base_month",
    "统计月": "stat_month",
    "月份": "stat_month",
    "stat_month": "stat_month",
    "month": "stat_month",
    "公布指数": "published_index",
    "指数": "published_index",
    "index": "published_index",
    "published_index": "published_index",
}

REQUIRED_COLUMNS = ("version", "base_month", "stat_month", "published_index")
COLUMN_LABELS = {
    "version": "版号",
    "base_month": "基期月",
    "stat_month": "统计月",
    "published_index": "公布指数",
}


@dataclass(frozen=True)
class Record:
    """一条同口径月报记录。"""

    row_no: int  # CSV 中的物理行号（含表头，从 1 开始）
    version: str
    base_month: str
    stat_month: str
    published_index: int


@dataclass
class Dataset:
    records: list[Record] = field(default_factory=list)
    # 版号 -> 该版基期月（同版基期唯一）
    version_base: dict[str, str] = field(default_factory=dict)
    # 版号 -> 该版全部记录
    version_records: dict[str, list[Record]] = field(default_factory=dict)
    # 全部出现过的月份
    months: set[str] = field(default_factory=set)
    # 解析阶段的非致命提示（如重复行）
    warnings: list[str] = field(default_factory=list)

    @property
    def versions(self) -> list[str]:
        return sorted(self.version_records.keys())


class CSVParseError(ValueError):
    """CSV 结构或字段校验失败，errors 中列出全部问题。"""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _norm_header(name: str) -> Optional[str]:
    key = name.strip().lstrip("﻿").lower()
    return HEADER_ALIASES.get(key) or HEADER_ALIASES.get(name.strip().lstrip("﻿"))


def parse_csv(text: str) -> Dataset:
    """解析并校验月报 CSV。

    要求四列：版号、基期月、统计月、公布指数（接受常见英文别名）。
    收集所有行的问题后一次性抛出 :class:`CSVParseError`。
    """
    if text is None or not text.strip():
        raise CSVParseError(["文件内容为空"])

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise CSVParseError(["文件内容为空"])

    header = [h.strip().lstrip("﻿") for h in rows[0]]
    col_index: dict[str, int] = {}
    unknown: list[str] = []
    for idx, raw in enumerate(header):
        mapped = _norm_header(raw)
        if mapped is None:
            unknown.append(raw)
            continue
        if mapped in col_index:
            raise CSVParseError([f"表头存在重复列：{raw}"])
        col_index[mapped] = idx

    missing = [COLUMN_LABELS[c] for c in REQUIRED_COLUMNS if c not in col_index]
    if missing:
        raise CSVParseError([f"缺少必需列：{'、'.join(missing)}（需要 版号/基期月/统计月/公布指数）"])
    if unknown:
        raise CSVParseError([f"无法识别的表头列：{'、'.join(unknown)}"])

    errors: list[str] = []
    dataset = Dataset()
    seen: set[tuple[str, str, str, int]] = set()

    for offset, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue  # 跳过空行
        if len(row) != len(header):
            errors.append(f"第 {offset} 行：列数为 {len(row)}，应为 {len(header)}")
            continue

        version = row[col_index["version"]].strip()
        base_month = row[col_index["base_month"]].strip()
        stat_month = row[col_index["stat_month"]].strip()
        index_raw = row[col_index["published_index"]].strip()

        row_problems: list[str] = []
        if not version:
            row_problems.append("版号为空")
        if not MONTH_RE.match(base_month):
            row_problems.append(f"基期月格式应为 YYYY-MM：{base_month or '（空）'}")
        if not MONTH_RE.match(stat_month):
            row_problems.append(f"统计月格式应为 YYYY-MM：{stat_month or '（空）'}")
        if not INT_RE.match(index_raw):
            row_problems.append(f"公布指数应为正整数：{index_raw or '（空）'}")
        elif int(index_raw) <= 0:
            row_problems.append(f"公布指数应为正整数：{index_raw}")

        if row_problems:
            errors.append(f"第 {offset} 行：" + "；".join(row_problems))
            continue

        index = int(index_raw)
        key = (version, base_month, stat_month, index)
        if key in seen:
            dataset.warnings.append(f"第 {offset} 行与已有记录完全相同，已忽略重复行")
            continue
        seen.add(key)

        record = Record(offset, version, base_month, stat_month, index)
        dataset.records.append(record)
        dataset.months.add(base_month)
        dataset.months.add(stat_month)
        dataset.version_records.setdefault(version, []).append(record)

        prior_base = dataset.version_base.get(version)
        if prior_base is None:
            dataset.version_base[version] = base_month
        elif prior_base != base_month:
            errors.append(
                f"第 {offset} 行：版号 {version} 的基期月冲突，"
                f"此前为 {prior_base}，本行为 {base_month}"
            )

    if not dataset.records and not errors:
        errors.append("文件没有任何有效数据行")
    if errors:
        raise CSVParseError(errors)
    return dataset


# --------------------------------------------------------------------------- #
# 差分约束图
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Edge:
    u: str
    v: str
    weight: float
    record: Record
    kind: str  # "hi"：F(stat)/F(base) 上界边；"lo"：下界反向边


def _bounds(index: int) -> tuple[float, float]:
    """返回 F(stat)/F(base) 的 (下界, 上界)。"""
    low = (index - INDEX_TOLERANCE) / 100.0
    high = (index + INDEX_TOLERANCE) / 100.0
    return low, high


def build_edges(records: list[Record]) -> list[Edge]:
    """把月报记录转成差分约束有向边。

    基期月与统计月相同的自指记录（基期本身，指数恒为 100）不包含任何
    月份间的比较信息，其自环只会干扰最短距离计算，故跳过。
    """
    edges: list[Edge] = []
    for r in records:
        if r.base_month == r.stat_month:
            continue
        low, high = _bounds(r.published_index)
        # x_stat - x_base <= ln(high)
        edges.append(Edge(r.base_month, r.stat_month, math.log(high), r, "hi"))
        # x_base - x_stat <= ln(1/low)
        edges.append(Edge(r.stat_month, r.base_month, -math.log(low), r, "lo"))
    return edges


def _adjacency(edges: list[Edge]) -> dict[str, list[Edge]]:
    adj: dict[str, list[Edge]] = {}
    for e in edges:
        adj.setdefault(e.u, []).append(e)
        adj.setdefault(e.v, [])
    return adj


def _undirected_connected(edges: list[Edge], start: str, target: str) -> bool:
    adj: dict[str, list[str]] = {}
    nodes = {start, target}
    for e in edges:
        adj.setdefault(e.u, []).append(e.v)
        adj.setdefault(e.v, []).append(e.u)
        nodes.add(e.u)
        nodes.add(e.v)
    if start not in nodes or target not in nodes:
        return False
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur == target:
            return True
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return target in seen


def _bellman_ford(edges: list[Edge], source: str) -> tuple[dict[str, float], dict[str, Edge | None]]:
    """Bellman-Ford 单源最短路径，返回 (距离, 前驱边)。

    以无穷大初始化、仅 source 为 0，保证需要 n-1 轮传播的多跳路径
    能够收敛。调用方需先用 :func:`_negative_cycle` 确认无负环。
    """
    nodes: set[str] = set()
    for e in edges:
        nodes.add(e.u)
        nodes.add(e.v)
    inf = float("inf")
    dist = {n: inf for n in nodes}
    pred: dict[str, Edge | None] = {n: None for n in nodes}
    if source in dist:
        dist[source] = 0.0

    for _ in range(max(1, len(nodes) - 1)):
        updated = False
        for e in edges:
            if dist[e.u] == inf:
                continue
            cand = dist[e.u] + e.weight
            if cand < dist[e.v]:
                dist[e.v] = cand
                pred[e.v] = e
                updated = True
        if not updated:
            break
    return dist, pred


def _negative_cycle(edges: list[Edge]) -> Optional[list[Edge]]:
    """检测负权环并返回环上的边列表，无环返回 None。"""
    nodes: set[str] = set()
    for e in edges:
        nodes.add(e.u)
        nodes.add(e.v)
    dist = {n: 0.0 for n in nodes}
    pred: dict[str, Edge | None] = {n: None for n in nodes}

    n = len(nodes)
    for i in range(max(1, n - 1)):
        updated = False
        for e in edges:
            cand = dist[e.u] + e.weight
            if cand < dist[e.v]:
                dist[e.v] = cand
                pred[e.v] = e
                updated = True
        if not updated and i > 0:
            return None

    # 第 n 轮仍可松弛的节点一定在负环上或可由负环到达
    relaxed: Optional[str] = None
    for e in edges:
        if dist[e.u] + e.weight < dist[e.v]:
            relaxed = e.v
            pred[e.v] = e
    if relaxed is None:
        return None

    # 沿前驱回溯，必然回到环上某个节点
    seen: dict[str, int] = {}
    order: list[str] = []
    cur = relaxed
    while cur not in seen:
        seen[cur] = len(order)
        order.append(cur)
        pe = pred.get(cur)
        if pe is None:
            return None  # 理论上不会发生，防御性处理
        cur = pe.u
    cycle_start = seen[cur]
    cycle_nodes = order[cycle_start:]
    cycle_edges: list[Edge] = []
    for node in cycle_nodes:
        pe = pred[node]
        if pe is not None:
            cycle_edges.append(pe)
    return cycle_edges


def _reconstruct(pred: dict[str, Edge | None], source: str, target: str) -> list[Edge]:
    """根据前驱表重建 source -> target 的最短路径边序列。"""
    path: list[Edge] = []
    cur = target
    guard = 0
    max_steps = len(pred) + 1
    while cur != source:
        pe = pred.get(cur)
        if pe is None or guard > max_steps:
            return []  # 不可达或异常
        path.append(pe)
        cur = pe.u
        guard += 1
    path.reverse()
    return path


# --------------------------------------------------------------------------- #
# 判定结果
# --------------------------------------------------------------------------- #

VERDICT_RISE = "rise"            # 必涨
VERDICT_FALL = "fall"            # 必跌
VERDICT_UNKNOWN = "unknown"      # 无法证实涨跌
VERDICT_NO_BASIS = "no_basis"    # 两月缺少换算关系
VERDICT_CONFLICT = "conflict"    # 相关记录互相矛盾

VERDICT_LABELS = {
    VERDICT_RISE: "必涨",
    VERDICT_FALL: "必跌",
    VERDICT_UNKNOWN: "无法证实涨跌",
    VERDICT_NO_BASIS: "缺少换算依据",
    VERDICT_CONFLICT: "记录存在矛盾",
}


@dataclass
class PathStep:
    """证明链中的一步（对应一条源记录）。"""

    row_no: int
    version: str
    base_month: str
    stat_month: str
    published_index: int
    # 这一步比较的两个月份（按路径方向，from -> to）
    from_month: str
    to_month: str
    # F(to)/F(from) 的可行区间
    ratio_low: float
    ratio_high: float
    bound_kind: str  # "upper" 证明取上界；"lower" 证明取下界


def _step_from_edge(edge: Edge, invert: bool = False) -> PathStep:
    """由一条约束边构造证明步骤。

    正向（hi 边 base -> stat）：F(stat)/F(base) ∈ [low, high]，证明取上界；
    倒置（lo 边按 stat -> base 方向展示）：F(base)/F(stat) ∈ [1/high, 1/low]，
    若再沿反向使用（rise 链把 e -> s 的 lo 边翻回 s -> e 方向），
    区间需整体取倒数，证明取下界。
    """
    r = edge.record
    low, high = _bounds(r.published_index)
    if not invert:
        if edge.kind == "hi":
            return PathStep(
                r.row_no, r.version, r.base_month, r.stat_month, r.published_index,
                r.base_month, r.stat_month, low, high, "upper",
            )
        # lo 边：stat -> base，F(base)/F(stat) ∈ [1/high, 1/low]，取上界
        return PathStep(
            r.row_no, r.version, r.base_month, r.stat_month, r.published_index,
            r.stat_month, r.base_month, 1.0 / high, 1.0 / low, "upper",
        )
    # 倒置使用：边的展示方向与其物理方向相反，区间取倒数
    if edge.kind == "hi":
        # 物理方向 base -> stat，展示为 stat -> base
        return PathStep(
            r.row_no, r.version, r.base_month, r.stat_month, r.published_index,
            r.stat_month, r.base_month, 1.0 / high, 1.0 / low, "lower",
        )
    # lo 边物理方向 stat -> base，展示为 base -> stat
    return PathStep(
        r.row_no, r.version, r.base_month, r.stat_month, r.published_index,
        r.base_month, r.stat_month, low, high, "lower",
    )


@dataclass
class AnalysisResult:
    verdict: str
    start_month: str
    end_month: str
    selected_versions: list[str]
    message: str
    # 证明链（必涨/必跌时非空）
    chain: list[PathStep] = field(default_factory=list)
    # 链上各步区间连乘得到的总区间
    product_low: Optional[float] = None
    product_high: Optional[float] = None
    # 无法证实涨跌时给出可行区间
    ratio_low: Optional[float] = None
    ratio_high: Optional[float] = None
    # 矛盾时给出无法共存的记录（行号 + 摘要）
    conflicts: list[Record] = field(default_factory=list)
    used_records: list[Record] = field(default_factory=list)


def analyze(
    dataset: Dataset,
    selected_versions: list[str],
    start_month: str,
    end_month: str,
) -> AnalysisResult:
    """执行涨跌判定。

    仅使用勾选版次的记录，并要求这些记录与起、止两月在同一连通分量内
    （“与两月相连的全部换基记录”）。
    """
    selected = set(selected_versions)
    used = [r for r in dataset.records if r.version in selected]
    versions_sorted = sorted(selected)

    if start_month not in dataset.months:
        return AnalysisResult(
            VERDICT_NO_BASIS, start_month, end_month, versions_sorted,
            f"起月 {start_month} 不在已上传数据中，无法建立换算关系",
        )
    if end_month not in dataset.months:
        return AnalysisResult(
            VERDICT_NO_BASIS, start_month, end_month, versions_sorted,
            f"止月 {end_month} 不在已上传数据中，无法建立换算关系",
        )
    if not used:
        return AnalysisResult(
            VERDICT_NO_BASIS, start_month, end_month, versions_sorted,
            "未勾选任何版次，没有可用的换算记录",
        )

    all_edges = build_edges(used)

    if not _undirected_connected(all_edges, start_month, end_month):
        return AnalysisResult(
            VERDICT_NO_BASIS, start_month, end_month, versions_sorted,
            f"{start_month} 与 {end_month} 之间不存在任何相连的换基记录，"
            "缺少换算依据，无法比较客流涨跌",
        )

    # 仅保留与两月同属一个连通分量的记录（题目：与两月相连的全部换基记录）
    component_nodes = _component_nodes(all_edges, start_month)
    rel_edges = [e for e in all_edges if e.u in component_nodes]
    rel_records = []
    seen_rows: set[int] = set()
    for e in rel_edges:
        if e.record.row_no not in seen_rows:
            seen_rows.add(e.record.row_no)
            rel_records.append(e.record)

    cycle = _negative_cycle(rel_edges)
    if cycle is not None:
        conflict_rows: list[int] = []
        conflict_records: list[Record] = []
        for e in cycle:
            if e.record.row_no not in conflict_rows:
                conflict_rows.append(e.record.row_no)
                conflict_records.append(e.record)
        rows_text = "、".join(f"第 {r.row_no} 行" for r in conflict_records)
        return AnalysisResult(
            VERDICT_CONFLICT, start_month, end_month, versions_sorted,
            f"检测到负权环，下列记录的指数区间无法同时成立（{rows_text}），"
            "判定暂停，请先核对源数据",
            conflicts=conflict_records,
            used_records=rel_records,
        )

    dist_s, pred_s = _bellman_ford(rel_edges, start_month)
    dist_e, pred_e = _bellman_ford(rel_edges, end_month)

    upper = dist_s.get(end_month)      # d = x_e - x_s <= upper
    lower = -dist_e.get(start_month)   # -d <= dist_e[s]  => d >= -dist_e[s]

    eps = 1e-12
    if upper is not None and upper < -eps:
        # 必跌：取 s -> e 最短路，逐上界连乘，乘积 < 1
        path = _reconstruct(pred_s, start_month, end_month)
        chain = [_step_from_edge(e) for e in path]
        prod_hi = 1.0
        prod_lo = 1.0
        for st in chain:
            prod_hi *= st.ratio_high
            prod_lo *= st.ratio_low
        return AnalysisResult(
            VERDICT_FALL, start_month, end_month, versions_sorted,
            f"必跌：{end_month} 客流相对 {start_month} 的最大可能比值为 "
            f"{prod_hi:.6f}（< 1），任何满足公布数据的真值都意味着下跌",
            chain=chain, product_low=prod_lo, product_high=prod_hi,
            used_records=rel_records,
        )

    if lower is not None and lower > eps:
        # 必涨：取 e -> s 最短路，将每条边倒置回 s -> e 方向，
        # 各步取下界连乘，乘积 > 1
        path = _reconstruct(pred_e, end_month, start_month)
        chain_forward = [_step_from_edge(e, invert=True) for e in reversed(path)]
        prod_lo = 1.0
        prod_hi = 1.0
        for st in chain_forward:
            prod_lo *= st.ratio_low
            prod_hi *= st.ratio_high
        return AnalysisResult(
            VERDICT_RISE, start_month, end_month, versions_sorted,
            f"必涨：{end_month} 客流相对 {start_month} 的最小可能比值为 "
            f"{prod_lo:.6f}（> 1），任何满足公布数据的真值都意味着上涨",
            chain=chain_forward, product_low=prod_lo, product_high=prod_hi,
            used_records=rel_records,
        )

    return AnalysisResult(
        VERDICT_UNKNOWN, start_month, end_month, versions_sorted,
        f"无法证实涨跌：{end_month} 相对 {start_month} 的可行比值区间为 "
        f"[{math.exp(lower):.6f}, {math.exp(upper):.6f}]，区间包含 1，"
        "现有记录不足以确定方向",
        ratio_low=math.exp(lower) if lower is not None else None,
        ratio_high=math.exp(upper) if upper is not None else None,
        used_records=rel_records,
    )


def _component_nodes(edges: list[Edge], start: str) -> set[str]:
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e.u, []).append(e.v)
        adj.setdefault(e.v, []).append(e.u)
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen
