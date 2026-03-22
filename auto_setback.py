"""
自动建筑退线生成器 v8
==================================
核心算法:
  1. ezdxf 读取 DXF，提取所有闭合多段线（含弧线离散化）
  2. 射线法检测道路宽度（替代图层名判断）
  3. 逐边不同退距的偏移线 + 相邻偏移线交点计算
  4. shapely 修复自交/裁剪
  5. 退距+路宽标注写入新图层
"""

import math
import sys
import os
from datetime import datetime
from collections import Counter

import ezdxf
from ezdxf import recover
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid

# ======================== 配置 ========================

INPUT_FILE = "cadfile.dxf"
PARCEL_LAYER = "地块边界"

ARC_N = 32
RAY_LEN = 100.0
LABEL_H = 8.0
MIN_EDGE = 20.0
DEFAULT_SB = 5.0

ts = datetime.now().strftime("%H-%M-%S")
OUT_FILE = f"output_{ts}.dxf"
LOG_FILE = f"setback_log_{ts}.txt"

log_buf = []


def log(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('utf-8', errors='replace').decode('utf-8'))
    log_buf.append(msg)


def get_setback(road_width):
    if road_width == float('inf'):
        return 5    # 无交点（边界地块），默认退5m
    if road_width < 3:
        return 5    # 紧邻地块，不是道路
    elif road_width >= 35:
        return 15   # 主干道
    elif road_width >= 20:
        return 10   # 次干道
    else:
        return 5    # 支路


def road_type_label(road_width):
    if road_width == float('inf'):
        return "边界"
    if road_width < 3:
        return "邻地块"
    elif road_width >= 35:
        return "主干道"
    elif road_width >= 20:
        return "次干道"
    else:
        return "支路"


# ======================== 弧线处理 ========================

def bulge_arc(p1, p2, b, n=ARC_N):
    if abs(b) < 1e-6:
        return [p1]
    x1, y1 = p1
    x2, y2 = p2
    ch = math.hypot(x2 - x1, y2 - y1)
    if ch < 1e-10:
        return [p1]
    s = abs(b) * ch / 2
    r = (ch ** 2 / 4 + s ** 2) / (2 * s)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    nx, ny = -(y2 - y1) / ch, (x2 - x1) / ch
    d = r - s
    cx = mx + (nx * d if b > 0 else -nx * d)
    cy = my + (ny * d if b > 0 else -ny * d)
    a0 = math.atan2(y1 - cy, x1 - cx)
    a1 = math.atan2(y2 - cy, x2 - cx)
    if b > 0:
        if a1 <= a0:
            a1 += 2 * math.pi
    else:
        if a1 >= a0:
            a1 -= 2 * math.pi
    return [
        (cx + r * math.cos(a0 + (a1 - a0) * i / n),
         cy + r * math.sin(a0 + (a1 - a0) * i / n))
        for i in range(n)
    ]


def expand_polyline(verts, bulges, closed):
    n = len(verts)
    co = []
    for k in range(n):
        if closed:
            nk = (k + 1) % n
        else:
            if k == n - 1:
                co.append(verts[k])
                break
            nk = k + 1
        bg = bulges[k] if k < len(bulges) else 0
        co.extend(bulge_arc(verts[k], verts[nk], bg))
    if closed and len(co) >= 3 and co[0] != co[-1]:
        co.append(co[0])
    return co


# ======================== 几何工具 ========================

def signed_area(pts):
    n = len(pts)
    if n > 1 and pts[0] == pts[-1]:
        n -= 1
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return a / 2


def ray_seg_hit(ox, oy, dx, dy, ax, ay, bx, by):
    """射线(ox,oy)+t*(dx,dy) 与线段 (ax,ay)-(bx,by) 的交点参数 t"""
    ex, ey = bx - ax, by - ay
    den = dx * ey - dy * ex
    if abs(den) < 1e-12:
        return None
    t = ((ax - ox) * ey - (ay - oy) * ex) / den
    s = ((ax - ox) * dy - (ay - oy) * dx) / den
    if t > 0.1 and -0.001 <= s <= 1.001:
        return t
    return None


def line_intersect(p1, p2, p3, p4):
    """两条直线（非线段）的交点"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


# ======================== 读取 DXF ========================

def read_dxf(filepath):
    """用 ezdxf 读取 DXF，返回 (doc, polylines_list)"""
    try:
        doc = ezdxf.readfile(filepath)
    except Exception:
        log("  ezdxf.readfile 失败，尝试 recover 模式...")
        doc, _auditor = recover.readfile(filepath)

    msp = doc.modelspace()
    polylines = []

    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE':
            layer = entity.dxf.layer
            is_closed = entity.closed
            verts = []
            bulges = []
            for x, y, _sw, _ew, bulge in entity.get_points(format='xyseb'):
                verts.append((x, y))
                bulges.append(bulge)
            if len(verts) < 2:
                continue
            coords = expand_polyline(verts, bulges, is_closed)
            polylines.append({
                'layer': layer,
                'verts': verts,
                'bulges': bulges,
                'coords': coords,
                'closed': is_closed,
            })
        elif entity.dxftype() == 'LINE':
            layer = entity.dxf.layer
            s = (entity.dxf.start.x, entity.dxf.start.y)
            e = (entity.dxf.end.x, entity.dxf.end.y)
            polylines.append({
                'layer': layer,
                'verts': [s, e],
                'bulges': [0],
                'coords': [s, e],
                'closed': False,
            })

    return doc, polylines


# ======================== 射线检测道路宽度 ========================

def collect_segments(polylines, exclude_idx=None):
    """将多段线列表转成 (ax,ay,bx,by) 数组，跳过 exclude_idx"""
    segs = []
    for i, pl in enumerate(polylines):
        if i == exclude_idx:
            continue
        co = pl['coords']
        for j in range(len(co) - 1):
            segs.append((co[j][0], co[j][1], co[j + 1][0], co[j + 1][1]))
    return segs


def detect_road_width(mid, normal, segments):
    """
    从 mid 沿 normal 方向发射射线，返回首个交点距离（=道路宽度）。
    """
    ox, oy = mid
    dx, dy = normal
    best = float('inf')
    for ax, ay, bx, by in segments:
        t = ray_seg_hit(ox, oy, dx, dy, ax, ay, bx, by)
        if t is not None and t < best:
            best = t
    return best if best < RAY_LEN else float('inf')


def detect_edge_road_width(p1, p2, bulge, out_normal, segments):
    """
    在边的 0.25/0.5/0.75 处各采样一次射线，返回中位数道路宽度。
    弧线边取弧线上的采样点。
    """
    if abs(bulge) > 0.01:
        arc_pts = bulge_arc(p1, p2, bulge)
        arc_pts.append(p2)
        sample_indices = [
            len(arc_pts) // 4,
            len(arc_pts) // 2,
            3 * len(arc_pts) // 4,
        ]
        samples = []
        for idx in sample_indices:
            idx = max(1, min(idx, len(arc_pts) - 2))
            pt = arc_pts[idx]
            prev_pt = arc_pts[idx - 1]
            next_pt = arc_pts[idx + 1] if idx + 1 < len(arc_pts) else p2
            tdx = next_pt[0] - prev_pt[0]
            tdy = next_pt[1] - prev_pt[1]
            tL = math.hypot(tdx, tdy)
            if tL < 1e-10:
                samples.append(detect_road_width(pt, out_normal, segments))
            else:
                # 弧线处的外法线根据切线方向重新计算
                on = _outward_from_tangent(tdx, tdy, tL, out_normal)
                samples.append(detect_road_width(pt, on, segments))
    else:
        samples = []
        for t in [0.25, 0.5, 0.75]:
            pt = (p1[0] + t * (p2[0] - p1[0]),
                  p1[1] + t * (p2[1] - p1[1]))
            samples.append(detect_road_width(pt, out_normal, segments))

    samples.sort()
    return samples[1]  # 中位数


def _outward_from_tangent(tdx, tdy, tL, ref_normal):
    """
    根据切线方向计算外法线，选择与 ref_normal 大致同向的法线。
    """
    n1 = (tdy / tL, -tdx / tL)
    n2 = (-tdy / tL, tdx / tL)
    dot1 = n1[0] * ref_normal[0] + n1[1] * ref_normal[1]
    dot2 = n2[0] * ref_normal[0] + n2[1] * ref_normal[1]
    return n1 if dot1 > dot2 else n2


# ======================== 逐边偏移退线生成 ========================

def _verify_inward_direction(verts, full_poly, is_ccw):
    """
    验证法线方向是否正确：取最长边的中点，沿推测的内法线偏移2m，
    检查该点是否在地块内部。如果不在，说明方向反了。
    """
    n = len(verts)
    best_k = 0
    best_len = 0
    for k in range(n):
        nk = (k + 1) % n
        d = math.hypot(verts[nk][0] - verts[k][0], verts[nk][1] - verts[k][1])
        if d > best_len:
            best_len = d
            best_k = k

    p1 = verts[best_k]
    p2 = verts[(best_k + 1) % n]
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy)
    if L < 1e-10:
        return is_ccw

    # CCW 内法线 = (-dy/L, dx/L), CW 内法线 = (dy/L, -dx/L)
    if is_ccw:
        inx, iny = -dy / L, dx / L
    else:
        inx, iny = dy / L, -dx / L

    mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    test_pt = Point(mid[0] + inx * 2, mid[1] + iny * 2)

    if full_poly.contains(test_pt):
        return is_ccw
    else:
        return not is_ccw


def _do_offset(verts, edge_setbacks, is_ccw):
    """
    给定顶点、每条边退距、绕向，计算偏移多边形顶点。
    返回闭合坐标列表。
    """
    n = len(verts)

    def in_n(dx, dy):
        L = math.hypot(dx, dy)
        if L < 1e-10:
            return (0.0, 0.0)
        if is_ccw:
            return (-dy / L, dx / L)
        else:
            return (dy / L, -dx / L)

    offset_lines = []
    for k in range(n):
        nk = (k + 1) % n
        p1, p2 = verts[k], verts[nk]
        sb = edge_setbacks[k]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        inx, iny = in_n(dx, dy)
        s = (p1[0] + inx * sb, p1[1] + iny * sb)
        e = (p2[0] + inx * sb, p2[1] + iny * sb)
        offset_lines.append((s, e))

    setback_pts = []
    for k in range(n):
        nk = (k + 1) % n
        s1, e1 = offset_lines[k]
        s2, e2 = offset_lines[nk]
        pt = line_intersect(s1, e1, s2, e2)
        if pt is None:
            pt = ((e1[0] + s2[0]) / 2, (e1[1] + s2[1]) / 2)
        setback_pts.append(pt)

    setback_pts.append(setback_pts[0])
    return setback_pts


def _extract_polygon(geom):
    """从 shapely 几何体中提取最大面积的 Polygon，或返回 None"""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == 'Polygon':
        return geom
    if geom.geom_type == 'MultiPolygon':
        return max(geom.geoms, key=lambda p: p.area)
    if geom.geom_type == 'GeometryCollection':
        polys = [g for g in geom.geoms if g.geom_type == 'Polygon']
        if polys:
            return max(polys, key=lambda p: p.area)
    return None


def compute_setback_polygon(parcel, all_segments):
    """
    为一个地块生成退线多边形。
    返回: (coords_or_None, edge_info_list, orig_area)

    保障机制：
      1. 用展开弧线后的坐标判断绕向（而非原始顶点）
      2. 实际偏移2m测试点验证法线方向，如果错了自动翻转
      3. 偏移多边形与原始地块做 intersection 裁剪
      4. 裁剪后再次验证不超出地块
      5. 若上述均失败，回退到 shapely.buffer(-d)
    """
    verts = parcel['verts']
    bulges = parcel['bulges']
    n = len(verts)

    # ---- 构建完整多边形（含弧线展开）----
    full_co = expand_polyline(verts, bulges, True)
    try:
        full_poly = Polygon(full_co)
        if not full_poly.is_valid:
            full_poly = make_valid(full_poly)
        full_poly = _extract_polygon(full_poly)
        if full_poly is None:
            return None, [], 0
        orig_area = full_poly.area
    except Exception:
        return None, [], 0

    if orig_area < 100:
        return None, [], orig_area

    # ---- 确定绕向（使用展开坐标 + 实际偏移验证）----
    sa = signed_area(full_co)
    is_ccw = sa > 0
    is_ccw = _verify_inward_direction(verts, full_poly, is_ccw)

    def out_n(dx, dy):
        L = math.hypot(dx, dy)
        if L < 1e-10:
            return (0.0, 0.0)
        if is_ccw:
            return (dy / L, -dx / L)
        else:
            return (-dy / L, dx / L)

    # ---- 第一遍: 射线检测每条边的道路宽度和退距 ----
    edge_setbacks = []
    edge_info = []

    for k in range(n):
        nk = (k + 1) % n
        p1, p2 = verts[k], verts[nk]
        b = bulges[k] if k < len(bulges) else 0
        el = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]

        if el < MIN_EDGE:
            edge_setbacks.append(DEFAULT_SB)
            edge_info.append(
                f"    边{k + 1}: 长{el:.0f}m 倒角 → 退{DEFAULT_SB:.0f}m"
            )
            continue

        on = out_n(dx, dy)
        road_w = detect_edge_road_width(p1, p2, b, on, all_segments)
        sb = get_setback(road_w)
        edge_setbacks.append(sb)

        rtype = road_type_label(road_w)
        w_str = f"{road_w:.1f}" if road_w < float('inf') else "∞"
        edge_info.append(
            f"    边{k + 1}: 长{el:.0f}m "
            f"{'弧' if abs(b) > 0.01 else '直'} → "
            f"[{rtype} w={w_str}m] → 退{sb:.0f}m"
        )

    if not edge_setbacks:
        return None, edge_info, orig_area

    # ---- 第二遍: 逐边偏移 + 交点计算 ----
    setback_pts = _do_offset(verts, edge_setbacks, is_ccw)

    # ---- shapely 验证、裁剪、包含性检测 ----
    try:
        sb_poly = Polygon(setback_pts)
        if not sb_poly.is_valid:
            sb_poly = make_valid(sb_poly)

        sb_poly = _extract_polygon(sb_poly)
        if sb_poly is None or sb_poly.is_empty:
            return _fallback_buffer(full_poly, edge_setbacks, edge_info, orig_area)

        # 如果偏移后面积反而更大，说明法线方向反了 → 翻转重做
        if sb_poly.area > orig_area * 1.05:
            log(f"    偏移面积({sb_poly.area:.0f})>原始({orig_area:.0f}), 翻转法线")
            is_ccw = not is_ccw
            setback_pts = _do_offset(verts, edge_setbacks, is_ccw)
            sb_poly = Polygon(setback_pts)
            if not sb_poly.is_valid:
                sb_poly = make_valid(sb_poly)
            sb_poly = _extract_polygon(sb_poly)
            if sb_poly is None or sb_poly.is_empty:
                return _fallback_buffer(full_poly, edge_setbacks, edge_info, orig_area)

        # 关键安全步骤: 与原始地块取交集，绝对不超出地块
        sb_poly = sb_poly.intersection(full_poly)
        sb_poly = _extract_polygon(sb_poly)
        if sb_poly is None or sb_poly.is_empty:
            return _fallback_buffer(full_poly, edge_setbacks, edge_info, orig_area)

        # 面积过小 → 回退
        if sb_poly.area < orig_area * 0.05:
            return _fallback_buffer(full_poly, edge_setbacks, edge_info, orig_area)

        # ---- 最终验证: 退线是否完全在地块内 ----
        overflow = sb_poly.difference(full_poly)
        overflow_area = overflow.area if overflow is not None else 0
        if overflow_area > 0.01:
            log(f"    !! 发现超出地块 {overflow_area:.2f}m2, 强制裁剪")
            sb_poly = sb_poly.intersection(full_poly)
            sb_poly = _extract_polygon(sb_poly)
            if sb_poly is None or sb_poly.is_empty:
                return _fallback_buffer(full_poly, edge_setbacks, edge_info, orig_area)

        sb_summary = Counter(edge_setbacks)
        su = " + ".join(
            f"{v}边x{k:.0f}m"
            for k, v in sorted(sb_summary.items(), reverse=True)
        )
        log(f"    退距: {su}")
        log(f"    退线: {sb_poly.area:.0f}m2 ({sb_poly.area / orig_area * 100:.0f}%)")
        log(f"    验证: OK (退线完全在地块内)")

        return list(sb_poly.exterior.coords), edge_info, orig_area

    except Exception as e:
        log(f"    shapely错误: {e}")
        return _fallback_buffer(full_poly, edge_setbacks, edge_info, orig_area)


def _fallback_buffer(full_poly, edge_setbacks, edge_info, orig_area):
    """逐边偏移失败时，用最小退距做统一 buffer (shapely保证不超出)"""
    min_sb = min(edge_setbacks) if edge_setbacks else DEFAULT_SB
    log(f"    逐边偏移失败 -> 回退到 buffer(-{min_sb:.0f}m)")
    result = full_poly.buffer(-min_sb, join_style=2, mitre_limit=2)
    if result.is_empty:
        return None, edge_info, orig_area
    result = _extract_polygon(result)
    if result is None or result.is_empty:
        return None, edge_info, orig_area
    log(f"    退线(buffer): {result.area:.0f}m2 ({result.area / orig_area * 100:.0f}%)")
    log(f"    验证: OK (buffer天然在地块内)")
    return list(result.exterior.coords), edge_info, orig_area


# ======================== 主流程 ========================

def main():
    bar = "=" * 60
    log(bar)
    log("  自动建筑退线生成器 v8 (射线检测 + 逐边偏移)")
    log(f"  时间: {ts}")
    log(bar)

    if not os.path.exists(INPUT_FILE):
        log(f"\n找不到 '{INPUT_FILE}'")
        sys.exit(1)

    # === 第一步: 读取 DXF ===
    log(f"\n[1/5] 读取 DXF...")
    doc, polylines = read_dxf(INPUT_FILE)

    layer_counts = Counter(pl['layer'] for pl in polylines)
    log(f"  总实体: {len(polylines)}")
    for ly, cnt in sorted(layer_counts.items()):
        log(f"  [{ly}]: {cnt} 条")

    # === 第二步: 识别地块 ===
    log(f"\n[2/5] 识别地块...")
    parcel_indices = [
        i for i, pl in enumerate(polylines)
        if pl['layer'] == PARCEL_LAYER and pl['closed'] and len(pl['verts']) >= 3
    ]
    if not parcel_indices:
        log(f"  未找到 '{PARCEL_LAYER}' 图层，尝试按面积筛选闭合多段线...")
        for i, pl in enumerate(polylines):
            if pl['closed'] and len(pl['verts']) >= 3:
                try:
                    a = abs(Polygon(pl['coords']).area)
                    if a > 1000:
                        parcel_indices.append(i)
                except Exception:
                    pass
    log(f"  地块数: {len(parcel_indices)}")
    if not parcel_indices:
        log("  无地块可处理!")
        sys.exit(1)

    # === 第三步: 为每个地块生成退线 ===
    log(f"\n[3/5] 生成退线...\n")
    setback_polys = []
    labels = []
    ok_count = fail_count = 0

    for idx, pi in enumerate(parcel_indices):
        parcel = polylines[pi]
        n = len(parcel['verts'])

        try:
            area = abs(Polygon(parcel['coords']).area)
        except Exception:
            area = 0
        log(f"  地块{idx + 1:>3d}  {area:>8.0f}m2  {n}边")

        all_segs = collect_segments(polylines, exclude_idx=pi)
        result, edge_info, orig_area = compute_setback_polygon(parcel, all_segs)
        for info in edge_info:
            log(info)

        if result is None:
            log(f"    ↳ 失败")
            fail_count += 1
            continue

        setback_polys.append(result)
        ok_count += 1

        # 生成标注 (每条主边中点标注退距和路宽)
        verts = parcel['verts']
        bgs = parcel['bulges']
        sa = signed_area(verts)
        is_ccw = sa > 0
        cx = sum(v[0] for v in verts) / n
        cy = sum(v[1] for v in verts) / n

        for k in range(n):
            nk = (k + 1) % n
            p1, p2 = verts[k], verts[nk]
            b = bgs[k] if k < len(bgs) else 0
            el = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if el < MIN_EDGE:
                continue

            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            L = math.hypot(dx, dy)
            if is_ccw:
                on = (dy / L, -dx / L)
            else:
                on = (-dy / L, dx / L)

            road_w = detect_edge_road_width(p1, p2, b, on, all_segs)
            sb = get_setback(road_w)

            if abs(b) > 0.01:
                arc_pts = bulge_arc(p1, p2, b)
                mid_pt = arc_pts[len(arc_pts) // 2]
            else:
                mid_pt = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

            # 标注位置: 边中点向质心方向偏移
            dx_c = cx - mid_pt[0]
            dy_c = cy - mid_pt[1]
            dc = math.hypot(dx_c, dy_c)
            if dc > 0.01:
                lx = mid_pt[0] + dx_c / dc * sb * 0.4
                ly = mid_pt[1] + dy_c / dc * sb * 0.4
            else:
                lx, ly = mid_pt

            angle = math.degrees(math.atan2(dy, dx))
            if road_w < float('inf'):
                text = f"退{sb:.0f}m(路宽{road_w:.0f}m)"
            else:
                text = f"退{sb:.0f}m"
            labels.append((lx, ly, text, angle))

    # === 第四步: 写入 DXF ===
    log(f"\n[4/5] 写入 DXF...")
    try:
        doc.layers.add("建筑红线", color=1)
    except Exception:
        pass
    try:
        doc.layers.add("标注_参考", color=2)
    except Exception:
        pass

    msp = doc.modelspace()
    ent_count = 0

    for pc in setback_polys:
        pts = pc[:-1] if (len(pc) > 1 and pc[0] == pc[-1]) else pc
        if len(pts) < 3:
            continue
        pl = msp.add_lwpolyline(
            [(x, y) for x, y in pts],
            dxfattribs={'layer': '建筑红线'}
        )
        pl.close()
        ent_count += 1

    for x, y, text, angle in labels:
        msp.add_text(
            text,
            dxfattribs={
                'layer': '标注_参考',
                'height': LABEL_H,
                'rotation': angle,
                'insert': (x, y),
            }
        )
        ent_count += 1

    doc.saveas(OUT_FILE)
    log(f"  写入 {ent_count} 个实体 → {OUT_FILE}")

    # === 第五步: 日志 ===
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_buf))

    log(f"\n{bar}")
    log(f"  完成! 成功: {ok_count}  失败: {fail_count}")
    log(f"  输出: {OUT_FILE}")
    log(f"  日志: {LOG_FILE}")
    log(bar)


if __name__ == '__main__':
    main()
