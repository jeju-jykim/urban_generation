# generator.py
# -*- coding: utf-8 -*-
"""
Roadless apartment-style generator with ZONE-first placement.
Order:
  1) Build masks from zones (edge_ring no-build, row bands, pockets, scatter).
  2) Place rows (zone-limited) → blocks (pockets) → towers (leftovers) → (optional) arc rows.
  3) Stop early if coverage_target reached.

Inputs:
  --rules rules.json
  --library library.json
  --assets_dir <dir with *_footprint_polygons.json>
  [--out_geo layout.geojson] [--out_svg preview.svg]

rules.json (핵심 섹션: 예시)
{
  "boundary": { "width": 500, "height": 500, "outer_setback": 16 },
  "angles": { "jitter": 5.0 },

  "setback": { "internal": 10.0 },

  "zones": {
    "edge_ring": { "width_m": 16.0, "mode": "no_build" },   // "build"|"no_build" (default: no_build)
    "row_bands": [
      {
        "labels": ["Bar Slab","Long Slab"],
        "band_count": 7,
        "row_spacing_m": 41.5,
        "along_spacing_m": 43.3,
        "band_width_m": 16.2,
        "orientation_angle_deg": 17.4
      }
    ],
    "core_blocks": { "enabled": true, "labels": ["Irregular Block","Courtyard Block"], "min_pocket_area_m2": 900, "max_count": 2, "angle_mode":"pad_obb", "clearance": 10.0 },
    "tower_scatter": { "labels": ["Tower"], "across_m": 41.5, "along_m": 43.3, "max_points": 1200 }
  },

  "channels": {               // 백업 채널(존에 labels가 없으면 사용)
    "rows_labels":    ["Bar Slab","Long Slab"],
    "tower_labels":   ["Tower","Micro Tower"],
    "block_labels":   ["Irregular Block","Courtyard Block"]
  },

  "density": { "coverage_target": 0.17 },
  "arc_rows": { "enabled": false },   // (옵션) 필요 시 기존 규칙과 병행 배치 가능
  "seed": 42
}

library.json:
{
  "groups": {"cluster_0":{"ids":[...]}, ...},
  "labels": {"cluster_0":{"label":"Bar Slab"}, ...},
  "mix":    {"cluster_0":0.123, ...},
  "n_total": <int>
}
"""

import os, json, math, random, argparse
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, box, mapping, shape
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.prepared import prep

# ---------- I/O ----------
def load_json(p): return json.load(open(p, "r", encoding="utf-8"))

def load_library(path):
    lib = load_json(path)
    label_to_clusters = {}
    for ck, meta in lib.get("labels", {}).items():
        lbl = (meta.get("label") or "").strip()
        if lbl:
            label_to_clusters.setdefault(lbl, []).append(ck)
    return lib, label_to_clusters

def find_asset_json(assets_dir, asset_id):
    cands = [
        os.path.join(assets_dir, f"{asset_id}_footprint_polygons.json"),
        os.path.join(assets_dir, f"{asset_id}.json"),
    ]
    for p in cands:
        if os.path.exists(p): return p
    for name in os.listdir(assets_dir):  # fallback
        if name.startswith(asset_id) and name.endswith(".json"):
            return os.path.join(assets_dir, name)
    return None

def load_polygon_from_asset(path):
    obj = load_json(path)
    g = None
    if "footprint" in obj:
        g = shape(obj["footprint"])
    elif obj.get("type") == "Feature":
        g = shape(obj.get("geometry", {}))
    elif obj.get("type") == "FeatureCollection":
        polys = [shape(ft.get("geometry", {}))
                 for ft in obj.get("features", [])
                 if ft.get("geometry", {}).get("type") in ("Polygon","MultiPolygon")]
        if polys:
            g = max(polys, key=lambda gg: gg.area)
    if g is None: raise ValueError(f"No polygon geometry in {path}")
    if g.geom_type == "MultiPolygon":
        g = max(list(g.geoms), key=lambda gg: gg.area)
    return g.buffer(0)

# ---------- Geom helpers ----------
def rotate_world(g, theta_deg, origin=(0,0)): return rotate(g, theta_deg, origin=origin)

def place(local_poly, x, y, theta_deg):
    return translate(rotate(local_poly, theta_deg, origin=(0,0)), xoff=x, yoff=y)

def make_line_strips(mask_poly, axis_deg, row_count, row_spacing, band_width):
    """row_count개의 평행 띠를 mask 영역 안에서 생성(축 = axis_deg)"""
    if row_count <= 0: return []
    c = Point(0,0)
    P = rotate_world(mask_poly, -axis_deg, origin=c)
    if P.is_empty: return []
    minx, miny, maxx, maxy = P.bounds
    ymid = 0.5*(miny+maxy)
    strips=[]
    start = ymid - (row_count-1)*0.5*row_spacing
    for i in range(row_count):
        y0 = start + i*row_spacing - 0.5*band_width
        y1 = y0 + band_width
        band = box(minx-10_000, y0, maxx+10_000, y1).intersection(P)
        if not band.is_empty:
            strips.append(rotate_world(band, axis_deg, origin=c))
    return strips

def arc_points(center, r, th0_deg, th1_deg, spacing):
    th0 = math.radians(th0_deg); th1 = math.radians(th1_deg)
    arc_len = abs(th1 - th0) * r
    n = max(1, int(arc_len // max(1e-6, spacing)))
    thetas = [0.5*(th0+th1)] if n<=1 else [th0 + i*(th1-th0)/float(n-1) for i in range(n)]
    return [(center[0] + r*math.cos(t), center[1] + r*math.sin(t), math.degrees(t)) for t in thetas]

# ---------- SVG ----------
def poly_to_svg_path(poly, sx, sy, ox, oy):
    def ring(coords):
        if not coords: return ""
        d = f"M {sx*(coords[0][0]+ox):.2f} {sy*(oy-coords[0][1]):.2f} "
        for x,y in coords[1:]:
            d += f"L {sx*(x+ox):.2f} {sy*(oy-y):.2f} "
        return d + "Z "
    if poly.geom_type == "Polygon":
        d = ring(list(poly.exterior.coords))
        for r in poly.interiors: d += ring(list(r.coords))
        return d
    elif poly.geom_type == "MultiPolygon":
        return " ".join(poly_to_svg_path(p, sx, sy, ox, oy) for p in poly.geoms)
    return ""

def save_svg(site, pad, geoms, out_svg, px=900, road_mask=None):
    minx, miny, maxx, maxy = site.bounds
    W, H = maxx-minx, maxy-miny
    sx = sy = px / max(W, H)
    ox, oy = -minx, maxy
    header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{px}" height="{px}" viewBox="0 0 {px} {px}">\n'
    bg = '<rect width="100%" height="100%" fill="white"/>\n'
    paths = []
    # pad
    paths.append(f'<path d="{poly_to_svg_path(pad, sx, sy, ox, oy)}" fill="#f4f4f4" stroke="#ddd" stroke-width="1"/>')
    # road/no-build mask 시각화(있으면)
    if road_mask and not road_mask.is_empty:
        paths.append(f'<path d="{poly_to_svg_path(road_mask, sx, sy, ox, oy)}" fill="#e8e8e8" stroke="none" opacity="0.8"/>')
    # buildings
    for g in geoms:
        paths.append(f'<path d="{poly_to_svg_path(g, sx, sy, ox, oy)}" fill="black" stroke="none"/>')
    open(out_svg, "w", encoding="utf-8").write(header+bg+"\n".join(paths)+"\n</svg>")

# ---------- Main ----------
def run(args):
    rules = load_json(args.rules)
    lib, _ = load_library(args.library)
    rng = random.Random(rules.get("seed", 0))

    # boundary / pad
    W = float(rules["boundary"]["width"]); H = float(rules["boundary"]["height"])
    outer = float(rules["boundary"].get("outer_setback", 0.0))
    site = box(-W/2, -H/2, W/2, H/2)
    pad = site.buffer(-outer) if outer>0 else site
    PAD_PREP = prep(pad)

    jitter = float(rules.get("angles", {}).get("jitter", 2.0))
    internal = float(rules.get("setback", {}).get("internal", 2.0))
    cov_target = float(rules.get("density",{}).get("coverage_target", 0.0)) or None

    # --- zones & masks ---
    Z = rules.get("zones", {})
    # edge ring → road/no-build (default)
    road_mask = None
    if "edge_ring" in Z:
        er = Z["edge_ring"]; w = float(er.get("width_m",0.0))
        if w > 0.0:
            inner = pad.buffer(-w)
            ring = pad.difference(inner) if not inner.is_empty else pad
            mode = (er.get("mode") or "no_build").lower()
            if mode == "no_build":
                road_mask = ring
            elif mode == "build":
                # build 전용 링으로 쓸 경우, road_mask는 없음
                pass

    BUILD_MASK = pad if road_mask is None else pad.difference(road_mask)
    BUILD_PREP = prep(BUILD_MASK)

    # --- label → clusters → ids (by labels list) ---
    def clusters_for(labels_list):
        keys=[]
        for lab in labels_list:
            for ck, meta in lib.get("labels", {}).items():
                if (meta.get("label") or "") == lab and ck in lib.get("groups", {}):
                    keys.append(ck)
        ws = [max(1e-9, float(lib.get("mix", {}).get(ck, 0.0))) for ck in keys]
        s = sum(ws) or 1.0
        ws = [w/s for w in ws]
        return keys, ws

    # zone labels (fallback to channels.* if missing)
    rows_labels  = []
    if Z.get("row_bands"):
        for rb in Z["row_bands"]:
            rows_labels += rb.get("labels", [])
    if not rows_labels:
        rows_labels = rules.get("channels",{}).get("rows_labels", [])

    tower_labels = Z.get("tower_scatter",{}).get("labels") or rules.get("channels",{}).get("tower_labels", [])
    block_labels = Z.get("core_blocks",{}).get("labels") or rules.get("channels",{}).get("block_labels", [])

    rows_keys, rows_w   = clusters_for(rows_labels)
    tower_keys, tower_w = clusters_for(tower_labels)
    block_keys, block_w = clusters_for(block_labels)

    print("[pool] rows:", len(rows_keys), "towers:", len(tower_keys), "blocks:", len(block_keys))

    def ids_for(cluster_keys):
        ids=[]; G = lib.get("groups", {})
        for ck in cluster_keys: ids += G.get(ck, {}).get("ids", [])
        return ids

    def load_assets(cluster_keys):
        aset={}; miss=0
        for aid in ids_for(cluster_keys):
            p = find_asset_json(args.assets_dir, aid)
            if not p: miss += 1; continue
            try: aset[aid] = load_polygon_from_asset(p)
            except Exception: miss += 1
        print(f"[assets] loaded={len(aset)} missing={miss} for {len(cluster_keys)} clusters")
        return aset

    A_rows   = load_assets(rows_keys)
    A_towers = load_assets(tower_keys)
    A_blocks = load_assets(block_keys)

    def choose_asset(cluster_keys, weights, cache):
        if not cluster_keys: return None, None
        ck = rng.choices(cluster_keys, weights=weights, k=1)[0]
        idlist = lib["groups"][ck]["ids"]
        for _ in range(20):
            if not idlist: break
            aid = rng.choice(idlist)
            if aid in cache: return aid, cache[aid]
        return None, None

    placed=[]; occ=[]; tree=None; area_sum=0.0
    def refresh_tree():
        nonlocal tree
        tree = STRtree(occ) if occ else None
    def coverage_now():
        return area_sum / max(1e-9, pad.area)

    def can_place_world(g):
        gg = g if g.is_valid else g.buffer(0)
        if gg.is_empty: return False
        if not BUILD_PREP.covers(gg): return False
        gb = gg.buffer(internal) if internal>0 else gg
        if tree:
            try:
                hits = tree.query(gb, predicate="intersects")
                if (hasattr(hits,"size") and hits.size>0) or (hasattr(hits,"__len__") and len(hits)>0):
                    return False
            except TypeError:
                if any(gb.intersects(o) for o in tree.query(gb)): return False
        return True

    # ---------------- ROW BANDS (zone-first straight rows) ----------------
    strips_all=[]
    if Z.get("row_bands") and rows_keys:
        # 마스크: buildable 영역과 edge no-build 제외
        base_mask = BUILD_MASK
        for rb in Z["row_bands"]:
            axis = float(rb.get("orientation_angle_deg", 0.0))
            n    = int(rb.get("band_count", 0))
            rsp  = float(rb.get("row_spacing_m", 40.0))
            bw   = float(rb.get("band_width_m", 18.0))
            # 마스크 교차(존이 따로 지정되지 않았으면 BUILD_MASK 전체를 사용)
            # 필요 시 rb에 "mask":"edge" 등 추가 가능. 지금은 BUILD_MASK에 생성.
            strips = make_line_strips(base_mask, axis, n, rsp, bw)
            # edge no-build를 빼서 실제 배치 가능한 띠로 축소
            if road_mask:
                strips = [s.difference(road_mask) for s in strips if not s.is_empty]
            strips = [s for s in strips if s and not s.is_empty]
            strips_all += [(s, axis, float(rb.get("along_spacing_m", max(rsp, 10.0)))) for s in strips]

        rows_jitter = float(rules.get("angles", {}).get("rows_jitter", jitter))
        for band, main_axis, step in strips_all:
            # 회전 좌표계로 이동
            B = rotate(band, -main_axis, origin=(0,0))
            if B.is_empty: continue
            minx, miny, maxx, maxy = B.bounds
            ymid = 0.5*(miny+maxy)
            x = minx + 0.5*step
            guard=0
            while x < maxx - 0.5*step and guard < 6000:
                guard += 1
                aid, poly0 = choose_asset(rows_keys, rows_w, A_rows)
                if aid is None: break
                theta = main_axis + rng.uniform(-rows_jitter, rows_jitter)
                pR = place(poly0, x, ymid, theta - main_axis)   # rotated frame
                pW = rotate_world(pR, main_axis, origin=(0,0))
                # 밴드 범위 안 + 빌드마스크 커버 + 충돌/세트백
                if not band.covers(pW):
                    x += 0.6*step; continue
                if can_place_world(pW):
                    occ.append(pW); placed.append(("row", aid, pW, theta)); refresh_tree()
                    area_sum += pW.area
                    x += step
                    if cov_target and coverage_now() >= cov_target*1.02: break
                else:
                    x += 0.6*step
            if cov_target and coverage_now() >= cov_target*1.02: break

    # ---------------- BLOCKS (pockets in leftover) ----------------
    BK = Z.get("core_blocks", {})
    if block_keys and BK.get("enabled", False):
        used = unary_union([g.buffer(internal) for (_,_,g,_) in placed]) if placed else None
        leftover = pad.difference(used) if used else pad
        # edge no-build 제거
        if road_mask: leftover = leftover.difference(road_mask)

        # 면적 임계 필터
        pockets=[]
        if leftover.geom_type == "Polygon":
            if leftover.area >= float(BK.get("min_pocket_area_m2", 900.0)): pockets=[leftover]
        else:
            pockets = [pg for pg in leftover.geoms if pg.area >= float(BK.get("min_pocket_area_m2", 900.0))]
        pockets.sort(key=lambda p: p.area, reverse=True)

        def pad_obb_angle(g):
            r = g.minimum_rotated_rectangle
            xs, ys = r.exterior.coords.xy
            pts = list(zip(xs, ys))[:4]
            e = [(pts[(i+1)%4][0]-pts[i][0], pts[(i+1)%4][1]-pts[i][1]) for i in range(4)]
            L = [(a*a+b*b)**0.5 for a,b in e]
            ex = e[int(np.argmax(L))]
            return math.degrees(math.atan2(ex[1], ex[0]))

        theta_mode = (BK.get("angle_mode") or "pad_obb").lower()
        theta_blk = pad_obb_angle(pad) if theta_mode=="pad_obb" else float(rules.get("angles",{}).get("main_axis_angle",0.0))
        placed_blocks = 0
        for pocket in pockets:
            if placed_blocks >= int(BK.get("max_count", 0)): break
            cx, cy = pocket.representative_point().x, pocket.representative_point().y
            for _try in range(32):
                aid, poly0 = choose_asset(block_keys, block_w, A_blocks)
                if aid is None: break
                theta = theta_blk + rng.uniform(-3.0, 3.0)
                g = place(poly0, cx, cy, theta)
                clear = float(BK.get("clearance", internal))
                if pocket.covers(g.buffer(clear)) and can_place_world(g):
                    occ.append(g); placed.append(("block", aid, g, theta)); refresh_tree()
                    area_sum += g.area
                    placed_blocks += 1
                    if cov_target and coverage_now() >= cov_target*1.02: break
                    break
            if cov_target and coverage_now() >= cov_target*1.02: break

    # ---------------- TOWERS (scatter in leftovers) ----------------
    SC = Z.get("tower_scatter", {})
    if tower_keys and (SC.get("across_m") and SC.get("along_m")):
        used = unary_union([g.buffer(internal) for (_,_,g,_) in placed]) if placed else None
        leftover = pad.difference(used) if used else pad
        if road_mask: leftover = leftover.difference(road_mask)

        def blue_like(geom, across, along, nmax, rng):
            minx, miny, maxx, maxy = geom.bounds
            pts=[]; tries=0
            while len(pts)<nmax and tries<nmax*40:
                tries+=1
                x=rng.uniform(minx,maxx); y=rng.uniform(miny,maxy)
                p=Point(x,y)
                if not geom.contains(p): continue
                ok=True
                for q in pts:
                    dx,dy=x-q.x,y-q.y
                    if (dx/along)**2 + (dy/across)**2 < 1.0:
                        ok=False; break
                if ok: pts.append(p)
            return pts

        pts = blue_like(leftover,
                        float(SC["across_m"]), float(SC["along_m"]),
                        int(SC.get("max_points", 1200)), rng)

        for p in pts:
            if cov_target and coverage_now() >= cov_target*1.02: break
            aid, poly0 = choose_asset(tower_keys, tower_w, A_towers)
            if aid is None: continue
            theta = rng.uniform(-5, 5)
            g = place(poly0, p.x, p.y, theta)
            if can_place_world(g):
                occ.append(g); placed.append(("tower", aid, g, theta)); refresh_tree()
                area_sum += g.area

    # ---------------- (Optional) ARC ROWS (legacy/extra flavor) ----------------
    AR = rules.get("arc_rows", {"enabled":False})
    if AR.get("enabled", False) and rows_keys and (not Z.get("row_bands")):
        cx, cy = AR.get("center",[0,0])
        th0, th1 = AR.get("theta_deg",[210,330])
        r0, r1   = AR.get("radius_m",[140,260])
        ring_count = int(AR.get("ring_count",3))
        ring_spacing = float(AR.get("ring_spacing",40))
        along_spacing = float(AR.get("along_spacing",22))
        t_off = float(AR.get("tangent_offset",0.0))

        radii=[]
        if ring_count<=1: radii=[(r0+r1)*0.5]
        else:
            total_span = (ring_count-1)*ring_spacing
            base = (r0+r1-total_span)/2.0
            radii = [base + i*ring_spacing for i in range(ring_count)]

        for r in radii:
            pts = arc_points((cx,cy), r, th0, th1, along_spacing)
            for (x,y,th_deg) in pts:
                if cov_target and coverage_now() >= cov_target*1.02: break
                aid, poly0 = choose_asset(rows_keys, rows_w, A_rows)
                if aid is None: continue
                tangent = th_deg + 90.0 + rng.uniform(-jitter, jitter) + t_off
                g = place(poly0, x, y, tangent)
                if BUILD_PREP.covers(g) and can_place_world(g):
                    occ.append(g); placed.append(("arc_row", aid, g, tangent)); refresh_tree()
                    area_sum += g.area

    # ---------- OUTPUT ----------
    feats=[
        {"type":"Feature","properties":{"class":"boundary"},"geometry":mapping(site)},
        {"type":"Feature","properties":{"class":"pad"},"geometry":mapping(pad)}
    ]
    for cls, aid, g, th in placed:
        feats.append({"type":"Feature","properties":{"class":cls,"asset_id":aid,"theta":th},
                      "geometry":mapping(g)})
    fc={"type":"FeatureCollection","features":feats}
    if args.out_geo: open(args.out_geo,"w",encoding="utf-8").write(json.dumps(fc, ensure_ascii=False))
    if args.out_svg: save_svg(site, pad, [g for _,_,g,_ in placed], args.out_svg, road_mask=road_mask)

    print(f"[done] buildings={len(placed)}  coverage={coverage_now():.3f}  → {args.out_geo}, {args.out_svg}")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default='../result/procedural/library/rules_from_typed_v2.json')
    ap.add_argument("--library", default='../result/procedural/library/library_with_context.json')
    ap.add_argument("--assets_dir", default='../result/separate_buildings_footprint')
    ap.add_argument("--out_geo", default="../result/procedural/library/layout.geojson")
    ap.add_argument("--out_svg", default="../result/procedural/library/preview.svg")
    args = ap.parse_args()
    run(args)
