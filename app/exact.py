"""基于精确有理数的差分约束引擎（导出换基摘要 / 拼接查询专用）。

为什么需要精确算术
==================
原判定页用 ``math.log`` + 浮点 Bellman-Ford，足以回答“涨/跌/无法证实”。
但换基摘要要求“月对之间的**精确**比值闭区间”，且要求导出前后所选月份
可同时成立的比值组合**完全一致**。若用浮点数，路径连乘的舍入误差可能让
区间端点发生极微小变化，极端情况下会让本来相容的约束变成矛盾（或相反）。
因此这里把每条指数区间视为精确有理数：

* 公布指数 ``I`` 的真值区间为 ``(I-0.5)/100`` 到 ``(I+0.5)/100``，
  即 ``(2I-1)/200`` 到 ``(2I+1)/200``，完全是有理数；
* 差分约束的一条边写成 ``x_v - x_u <= ln(w)``，其中 ``w`` 是正有理数；
* 在“对数空间”里求最短路等价于在正有理数上用乘法求最小乘积。乘积仍是
  有理数，比较用 :class:`fractions.Fraction` 精确进行，因此区间端点是
  严格精确的。

符号约定
========
变量 ``x_m = ln F(m)``。一条边 ``u -> v, weight=w`` 表示

    x_v - x_u <= ln(w)

即 ``F(v)/F(u) <= w``。每条月报记录产生两条边：

* ``base -> stat``，权重 ``(2I+1)/200``（比值上界）；
* ``stat -> base``，权重 ``200/(2I-1)``（比值下界的反向约束）。

负权环（环上权重乘积 < 1）等价于记录互相矛盾；乘积恰好为 1 是相容的
边界，不会误报。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Iterable, Optional

from .logic import Record

#: 1/2，指数的绝对误差上限
HALF = Fraction(1, 2)
ONE = Fraction(1, 1)


def index_ratio_bounds(index: int) -> tuple[Fraction, Fraction]:
    """由公布指数 ``I`` 得到 ``F(stat)/F(base)`` 的精确闭区间。"""
    low = Fraction(2 * index - 1, 200)
    high = Fraction(2 * index + 1, 200)
    return low, high


@dataclass(frozen=True)
class EdgeMeta:
    """约束边的来源信息，用于矛盾回溯与前端展示。"""

    kind: str  # "record"（原始月报）或 "summary"（摘要月对）
    record: Optional[Record] = None
    summary_name: Optional[str] = None
    summary_id: Optional[str] = None
    pair: Optional[tuple[str, str]] = None


@dataclass(frozen=True)
class ConstraintEdge:
    """一条精确差分约束边：``x_v - x_u <= ln(weight)``。"""

    u: str
    v: str
    weight: Fraction
    meta: EdgeMeta


def record_edges(records: Iterable[Record]) -> list[ConstraintEdge]:
    """把月报记录转成精确约束边（自指记录不含比较信息，跳过）。"""
    edges: list[ConstraintEdge] = []
    for r in records:
        if r.base_month == r.stat_month:
            continue
        low, high = index_ratio_bounds(r.published_index)
        meta = EdgeMeta(kind="record", record=r)
        edges.append(ConstraintEdge(r.base_month, r.stat_month, high, meta))
        edges.append(ConstraintEdge(r.stat_month, r.base_month, ONE / low, meta))
    return edges


def interval_edges(
    a: str,
    b: str,
    low: Fraction,
    high: Fraction,
    *,
    summary_name: str,
    summary_id: str,
    pair: tuple[str, str],
) -> list[ConstraintEdge]:
    """把摘要中的月对区间 ``F(b)/F(a) ∈ [low, high]`` 转成两条精确边。"""
    meta = EdgeMeta(
        kind="summary",
        summary_name=summary_name,
        summary_id=summary_id,
        pair=pair,
    )
    return [
        ConstraintEdge(a, b, high, meta),  # x_b - x_a <= ln(high)
        ConstraintEdge(b, a, ONE / low, meta),  # x_a - x_b <= ln(1/low)
    ]


# --------------------------------------------------------------------------- #
# 图遍历 / 连通分量
# --------------------------------------------------------------------------- #

def nodes_of(edges: Iterable[ConstraintEdge]) -> set[str]:
    nodes: set[str] = set()
    for e in edges:
        nodes.add(e.u)
        nodes.add(e.v)
    return nodes


def neighbors(edges: Iterable[ConstraintEdge]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    for e in edges:
        adj.setdefault(e.u, set()).add(e.v)
        adj.setdefault(e.v, set()).add(e.u)
    return adj


def undirected_components(
    nodes: Iterable[str], edges: Iterable[ConstraintEdge]
) -> list[set[str]]:
    """返回无向连通分量（包含没有任何边的孤立月份）。"""
    node_set = set(nodes)
    adj = neighbors(edges)
    for n in node_set:
        adj.setdefault(n, set())
    seen: set[str] = set()
    comps: list[set[str]] = []
    for start in sorted(node_set):
        if start in seen:
            continue
        comp = {start}
        seen.add(start)
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):  # type: ignore[arg-type]
                if nxt in node_set and nxt not in seen:
                    seen.add(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def component_of(start: str, edges: Iterable[ConstraintEdge]) -> set[str]:
    adj = neighbors(edges)
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, ()):  # type: ignore[arg-type]
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def edges_within(
    edges: Iterable[ConstraintEdge], nodes: set[str]
) -> list[ConstraintEdge]:
    return [e for e in edges if e.u in nodes and e.v in nodes]


# --------------------------------------------------------------------------- #
# 乘法版 Bellman-Ford
# --------------------------------------------------------------------------- #

def min_products(
    edges: list[ConstraintEdge],
    source: str,
    nodes: Optional[Iterable[str]] = None,
) -> tuple[dict[str, Optional[Fraction]], dict[str, Optional[ConstraintEdge]]]:
    """单源最小乘积（等价于对数空间最短路）。

    返回 ``(dist, pred)``。``dist[t]`` 是 source 到 t 的最小权重乘积；
    不可达为 ``None``。调用方应先用 :func:`find_negative_cycle` 确认
    相关分量无负环。
    """
    node_set = set(nodes) if nodes is not None else nodes_of(edges)
    if source not in node_set:
        return {n: None for n in node_set}, {n: None for n in node_set}
    dist: dict[str, Optional[Fraction]] = {n: None for n in node_set}
    pred: dict[str, Optional[ConstraintEdge]] = {n: None for n in node_set}
    dist[source] = ONE

    relevant = [e for e in edges if e.u in node_set and e.v in node_set]
    for _ in range(max(0, len(node_set) - 1)):
        updated = False
        for e in relevant:
            du = dist[e.u]
            if du is None:
                continue
            cand = du * e.weight
            dv = dist[e.v]
            if dv is None or cand < dv:
                dist[e.v] = cand
                pred[e.v] = e
                updated = True
        if not updated:
            break
    return dist, pred


def _extract_cycle(
    pred: dict[str, Optional[ConstraintEdge]], start: str
) -> Optional[list[ConstraintEdge]]:
    """沿前驱回溯出负环上的边。"""
    seen: dict[str, int] = {}
    order: list[str] = []
    cur = start
    while cur not in seen:
        seen[cur] = len(order)
        order.append(cur)
        pe = pred.get(cur)
        if pe is None:
            return None
        cur = pe.u
    cycle_nodes = order[seen[cur]:]
    cycle_edges: list[ConstraintEdge] = []
    for node in cycle_nodes:
        pe = pred.get(node)
        if pe is None:
            return None
        cycle_edges.append(pe)
    return cycle_edges


def find_negative_cycle(
    edges: list[ConstraintEdge],
    nodes: Optional[Iterable[str]] = None,
) -> Optional[list[ConstraintEdge]]:
    """查找权重乘积 < 1 的环（即无法同时成立的约束），无则返回 None。

    所有节点以距离 1 作为超级源，能发现任意分量里的负环。严格小于 1 的
    比较保证了乘积恰为 1 的相容边界不被误判。
    """
    node_set = set(nodes) if nodes is not None else nodes_of(edges)
    if len(node_set) < 2:
        # 单节点不可能形成需要检查的环（也不应有自环记录）
        return None
    relevant = [e for e in edges if e.u in node_set and e.v in node_set]
    if not relevant:
        return None

    dist: dict[str, Fraction] = {n: ONE for n in node_set}
    pred: dict[str, Optional[ConstraintEdge]] = {n: None for n in node_set}

    for _ in range(max(0, len(node_set) - 1)):
        updated = False
        for e in relevant:
            cand = dist[e.u] * e.weight
            if cand < dist[e.v]:
                dist[e.v] = cand
                pred[e.v] = e
                updated = True
        if not updated:
            return None

    relaxed_node: Optional[str] = None
    for e in relevant:
        if dist[e.u] * e.weight < dist[e.v]:
            pred[e.v] = e
            relaxed_node = e.v
            break
    if relaxed_node is None:
        return None
    return _extract_cycle(pred, relaxed_node)


def pair_bounds(
    edges: list[ConstraintEdge],
    start: str,
    end: str,
    nodes: Optional[Iterable[str]] = None,
) -> Optional[tuple[Fraction, Fraction]]:
    """返回 ``F(end)/F(start)`` 的精确可行闭区间 ``(low, high)``。

    无向不连通时返回 ``None``。调用方需保证所在分量无负环。
    """
    node_set = set(nodes) if nodes is not None else nodes_of(edges)
    if start not in node_set or end not in node_set:
        return None
    dist_s, _ = min_products(edges, start, node_set)
    upper = dist_s.get(end)
    if upper is None:
        return None
    dist_e, _ = min_products(edges, end, node_set)
    rev = dist_e.get(start)
    if rev is None:
        return None
    return ONE / rev, upper


# --------------------------------------------------------------------------- #
# 展示辅助
# --------------------------------------------------------------------------- #

def fraction_text(value: Fraction) -> str:
    """精确有理数的可移交文本（整数不带分母）。"""
    return str(value)


def parse_fraction_text(text: str) -> Fraction:
    """解析摘要中的精确有理数字符串。

    只接受 ``123`` 或 ``123/456`` 形式，拒绝小数与浮点数，避免精度丢失。
    """
    if not isinstance(text, str):
        raise ValueError("精确比值必须使用字符串表示（如 \"399/200\"）")
    raw = text.strip()
    if not raw:
        raise ValueError("精确比值为空")
    if "/" in raw:
        num_text, den_text = raw.split("/", 1)
        if not num_text.isdigit() or not den_text.isdigit():
            raise ValueError(f"精确比值格式应为 整数/整数：{text}")
        den = int(den_text)
        if den == 0:
            raise ValueError(f"精确比值分母不能为 0：{text}")
        value = Fraction(int(num_text), den)
    else:
        if not raw.isdigit():
            raise ValueError(f"精确比值格式应为 整数 或 整数/整数：{text}")
        value = Fraction(int(raw), 1)
    if value <= 0:
        raise ValueError(f"精确比值必须为正数：{text}")
    return value


def fraction_decimal(value: Fraction, significant: int = 12) -> str:
    """给出精确有理数的高精度十进制近似，仅用于页面展示。"""
    with localcontext() as ctx:
        ctx.prec = max(20, significant + 8)
        dec = Decimal(value.numerator) / Decimal(value.denominator)
        return f"{dec:.{significant}g}"
