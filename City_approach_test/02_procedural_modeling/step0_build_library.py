# step0_build_library.py  ← 기존 파일 교체
# -*- coding: utf-8 -*-
"""
Footprint 라이브러리 구축 v2
- KMeans(k=5 기본) + 확장형상피처(Convexity/Complexity/Rectangularity/Compactness/Holes)
- QC: Silhouette/Davies–Bouldin + per-cluster IQR
- 썸네일: 클러스터별 무작위 5–10개, OBB 정렬로 회전 정규화하여 SVG 타일 생성

설치: pip install shapely scikit-learn numpy
입력: --assets_dir 아래 *.json (스키마: {meta?, footprint: <GeoJSON Polygon/MultiPolygon>} 또는 GeoJSON Feature/FC)
출력:
  - library_raw.json         : cluster_i → ids (LLM 네이밍 입력용)
  - library_report.json      : 클러스터 통계 + kmeans_centers_std
  - features.csv             : 자산별 피처 + 할당 클러스터
  - cluster_qc.json          : QC(실루엣/DB Index) + IQR 요약
  - (옵션) cluster_QA.svg     : 클러스터별 무작위 미리보기
"""

import os, glob, json, csv, math, random
from typing import List, Optional, Dict, Tuple
import numpy as np
from collections import defaultdict

from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.affinity import rotate, translate, scale
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ─────────────────────────────
# 0) 유틸
# ─────────────────────────────
def _obb_params(poly: Polygon) -> Tuple[float, float, float]:
    """OBB 폭/깊이/각도(도). 가장 긴 변 방향 각도(−180..180)."""
    r = poly.minimum_rotated_rectangle
    xs, ys = r.exterior.coords.xy
    pts = list(zip(xs, ys))[:4]
    e01 = (pts[1][0]-pts[0][0], pts[1][1]-pts[0][1])
    e12 = (pts[2][0]-pts[1][0], pts[2][1]-pts[1][1])
    L1 = (e01[0]**2 + e01[1]**2)**0.5
    L2 = (e12[0]**2 + e12[1]**2)**0.5
    if L1 >= L2:
        width, depth = L1, L2
        ang = math.degrees(math.atan2(e01[1], e01[0]))
    else:
        width, depth = L2, L1
        ang = math.degrees(math.atan2(e12[1], e12[0]))
    return width, depth, ang

def _iter_polys_from_file(fp) -> List[Polygon]:
    try:
        obj = json.load(open(fp, "r", encoding="utf-8"))
    except Exception:
        return []
    geoms: List[BaseGeometry] = []
    if isinstance(obj, dict) and "footprint" in obj:
        geoms.append(shape(obj["footprint"]))
    elif obj.get("type") == "Feature":
        geoms.append(shape(obj.get("geometry", {})))
    elif obj.get("type") == "FeatureCollection":
        for ft in obj.get("features", []):
            g = shape(ft.get("geometry", {}))
            if g.geom_type in ("Polygon", "MultiPolygon"):
                geoms.append(g)
    polys: List[Polygon] = []
    for g in geoms:
        if isinstance(g, Polygon):
            polys.append(g.buffer(0))
        elif isinstance(g, MultiPolygon) and len(g.geoms) > 0:
            polys.append(max(g.geoms, key=lambda p: p.area).buffer(0))
    return [p for p in polys if p.is_valid and p.area > 0]

# ─────────────────────────────
# 1) 피처 추출(확장판)
# ─────────────────────────────
def _features_from(poly: Polygon, meta: Optional[dict]) -> Dict[str, float]:
    # 면적
    if meta and isinstance(meta.get("area_m2", None), (int, float)):
        area = float(meta["area_m2"])
    else:
        area = float(poly.area)

    # OBB
    if meta and isinstance(meta.get("obb", {}).get("aspect_ratio", None), (int, float)):
        ar = float(meta["obb"]["aspect_ratio"])
        w = float(meta["obb"].get("width_m", 0.0)) or None
        d = float(meta["obb"].get("depth_m", 0.0)) or None
        if not (w and d):
            w, d, _ = _obb_params(poly)
    else:
        w, d, _ = _obb_params(poly)
        ar = w / max(d, 1e-9)

    # Convexity (1=볼록)
    convexity = area / max(poly.convex_hull.area, 1e-9)

    # Complexity (경계 복잡도: P/√A)
    complexity = poly.length / max(area**0.5, 1e-9)

    # Rectangularity (A / OBB_A) — 직사각형에 가까울수록 1
    rectangularity = area / max(w * d, 1e-9)

    # Compactness (P^2 / 4πA) — 원에 가까울수록 1 (여기선 ≥1, 1이면 원형)
    compactness = (poly.length**2) / max(4.0 * math.pi * area, 1e-9)

    # Holes: 개수 + 면적비
    ext_area = Polygon(poly.exterior).area
    holes_area_ratio = max((ext_area - area) / max(ext_area, 1e-9), 0.0)
    holes = max(len(poly.interiors), 0)

    return dict(
        area=area, ar=ar, convexity=convexity, complexity=complexity,
        rectangularity=rectangularity, compactness=compactness,
        holes=holes, holes_area_ratio=holes_area_ratio
    )

def _row_from_file(fp):
    obj = json.load(open(fp, "r", encoding="utf-8"))
    meta = obj.get("meta", None) if isinstance(obj, dict) else None
    asset_id = (meta.get("asset_id") if meta and meta.get("asset_id") else
                os.path.splitext(os.path.basename(fp))[0])
    polys = _iter_polys_from_file(fp)
    if not polys: return None
    p = max(polys, key=lambda x: x.area).buffer(0)
    feats = _features_from(p, meta)
    feats["id"] = str(asset_id)
    return feats

# ─────────────────────────────
# 2) 미리보기 SVG (OBB 정렬)
# ─────────────────────────────
def _poly_to_path(poly, sx, sy, ox, oy):
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
    return ""

def _render_cluster_svg(samples: Dict[str, List[Polygon]], out_svg: str, cols=5, cell=140):
    # samples: cluster_key -> list of Polygons (이미 OBB 정렬·정규화되어 있다고 가정)
    keys = sorted(samples.keys())
    n = sum(len(v) for v in samples.values())
    rows = max(1, (n + cols - 1)//cols)
    W = cols * cell; H = rows * cell
    sx = sy = 1.0; ox = oy = 0.0
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    i = 0
    for k in keys:
        for g in samples[k]:
            r, c = i//cols, i%cols
            cx, cy = c*cell + cell*0.5, r*cell + cell*0.5
            minx, miny, maxx, maxy = g.bounds
            w, h = maxx-minx, maxy-miny
            s = 0.8 * min(cell/w, cell/h)
            gg = translate(scale(g, xfact=s, yfact=s, origin=(0,0)), xoff=cx - s*(minx+maxx)/2.0, yoff=cy - s*(miny+maxy)/2.0)
            d = _poly_to_path(gg, 1.0, 1.0, 0.0, 0.0)
            lines.append(f'<path d="{d}" fill="black" stroke="none" opacity="0.95"/>')
            i += 1
    lines.append("</svg>")
    open(out_svg, "w", encoding="utf-8").write("\n".join(lines))

# ─────────────────────────────
# 3) 메인
# ─────────────────────────────
def build_library(assets_dir: str, k: int = 5, min_area: float = 1e-6,
                  out_dir: str = "./",
                  limit: Optional[int] = None,
                  random_state: int = 0):
    files = sorted(glob.glob(os.path.join(assets_dir, "**/*.json"), recursive=True))
    if not files:
        raise SystemExit(f"[ERR] No JSON under: {assets_dir}")

    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(random_state)

    rows = []
    geom_cache = {}  # 썸네일용 빠른 접근
    skipped = 0
    for fp in files:
        try:
            obj = json.load(open(fp, "r", encoding="utf-8"))
            meta = obj.get("meta", None) if isinstance(obj, dict) else None
            polys = _iter_polys_from_file(fp)
            if not polys: 
                skipped += 1; continue
            p = max(polys, key=lambda x: x.area).buffer(0)
            feats = _features_from(p, meta)
            feats["id"] = (meta.get("asset_id") if meta and meta.get("asset_id") else
                           os.path.splitext(os.path.basename(fp))[0])
            if feats["area"] < min_area:
                skipped += 1; continue
            rows.append(feats)
            geom_cache[feats["id"]] = p  # 썸네일용 저장
            if limit and len(rows) >= limit:
                break
        except Exception:
            skipped += 1
            continue

    if not rows:
        raise SystemExit("[ERR] Loaded 0 valid footprints. Check JSON schema or --min_area.")

    # 특징행렬 (Z공간) — 로그면적 + 6개 형상피처
    A = np.array([[ math.log(r["area"] + 1e-9),
                    r["ar"], 1.0 - r["convexity"], r["complexity"],
                    r["rectangularity"], r["compactness"],
                    r["holes"], r["holes_area_ratio"] ] for r in rows], dtype=np.float64)
    scaler = StandardScaler().fit(A)
    Z = scaler.transform(A)

    # KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Z)
    y = km.labels_
    centers = km.cluster_centers_

    # QC 지표
    sil = float(silhouette_score(Z, y)) if k > 1 and len(Z) > k else None
    dbi = float(davies_bouldin_score(Z, y)) if k > 1 else None

    # 그룹 묶기
    clusters = {f"cluster_{i}": [] for i in range(k)}
    for r, lab in zip(rows, y):
        clusters[f"cluster_{int(lab)}"].append(r)
    total = sum(len(v) for v in clusters.values())

    # 리포트 통계 + 예시 id
    out_clusters = {}
    for key, items in clusters.items():
        def arr(name): return np.array([it[name] for it in items], dtype=np.float64) if items else np.array([0.0])

        ids = [it["id"] for it in items]
        mu_area = float(arr("area").mean())
        exemplars = [it["id"] for it in sorted(items, key=lambda it: abs(it["area"]-mu_area))[:10]]

        def rng_pair(a): return [float(a.min()), float(a.max())]
        def iqr(a): 
            q1, q3 = np.quantile(a, [0.25, 0.75]) if len(a)>3 else (a.min(), a.max())
            return float(q3-q1)

        out_clusters[key] = {
            "count": len(items),
            "share": round(len(items)/total, 3),
            "means": {
                "area": mu_area,
                "ar": float(arr("ar").mean()),
                "convexity": float(arr("convexity").mean()),
                "complexity": float(arr("complexity").mean()),
                "rectangularity": float(arr("rectangularity").mean()),
                "compactness": float(arr("compactness").mean()),
                "holes": float(arr("holes").mean()),
                "holes_area_ratio": float(arr("holes_area_ratio").mean()),
            },
            "ranges": {
                "area": rng_pair(arr("area")),
                "ar": rng_pair(arr("ar")),
                "convexity": rng_pair(arr("convexity")),
                "complexity": rng_pair(arr("complexity")),
                "rectangularity": rng_pair(arr("rectangularity")),
                "compactness": rng_pair(arr("compactness")),
                "holes": rng_pair(arr("holes")),
                "holes_area_ratio": rng_pair(arr("holes_area_ratio")),
            },
            "iqr": {
                "ar": iqr(arr("ar")),
                "convexity": iqr(arr("convexity")),
                "complexity": iqr(arr("complexity")),
                "rectangularity": iqr(arr("rectangularity")),
                "compactness": iqr(arr("compactness")),
            },
            "exemplar_ids": exemplars
        }

    # raw + report
    library_raw = {
        "clusters": { key: {"ids": [it["id"] for it in items]} for key, items in clusters.items() },
        "ratios_raw": { key: round(len(items)/total, 3) for key, items in clusters.items() },
        "n_total": total
    }
    with open(os.path.join(out_dir, "library_raw.json"),"w",encoding="utf-8") as f:
        json.dump(library_raw, f, ensure_ascii=False, indent=2)

    report = {
        "clusters": out_clusters,
        "kmeans_centers_std": centers.tolist()
    }
    with open(os.path.join(out_dir, "library_report.json"),"w",encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # features.csv
    with open(os.path.join(out_dir, "features.csv"),"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","area","ar","convexity","complexity","rectangularity","compactness","holes","holes_area_ratio","group"])
        for r, lab in zip(rows, y):
            w.writerow([r["id"], r["area"], r["ar"], r["convexity"], r["complexity"], r["rectangularity"], r["compactness"], r["holes"], r["holes_area_ratio"], f"cluster_{int(lab)}"])

    # QC JSON
    qc = {"silhouette": sil, "davies_bouldin": dbi,
          "k": k, "n_total": total,
          "note": "silhouette 높을수록 좋고, DB index 낮을수록 좋습니다."}
    with open(os.path.join(out_dir, "cluster_qc.json"),"w",encoding="utf-8") as f:
        json.dump(qc, f, ensure_ascii=False, indent=2)

    # 썸네일 SVG (무작위 5–10개, OBB 정렬)
    qa_svg_path = os.path.join(out_dir, "cluster_QA.svg")
    samples = {}
    for key, items in clusters.items():
        ids = [it["id"] for it in items]
        rng.shuffle(ids)
        pick = ids[:max(5, min(10, len(ids)))]
        plist=[]
        for pid in pick:
            g = geom_cache.get(pid)
            if g is None: continue
            w, d, ang = _obb_params(g)
            gg = rotate(g, -ang, origin=(0,0))  # OBB 정렬(방향 정규화)
            # 크기 정규화(OBB 가로가 1이 되도록)
            if w > 0:
                gg = scale(gg, xfact=1.0/w, yfact=1.0/w, origin=(0,0))
            plist.append(gg)
        samples[key] = plist
    _render_cluster_svg(samples, qa_svg_path)

    print(f"[OK] assets={total}, skipped={skipped} → {os.path.join(out_dir,'library_raw.json')}, {os.path.join(out_dir,'library_report.json')}, features.csv, cluster_qc.json, cluster_QA.svg")
    return library_raw, report

# ─────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets_dir", default="../result/separate_buildings_footprint")
    ap.add_argument("--k", type=int, default=5)                  # ← 기본 5개
    ap.add_argument("--min_area", type=float, default=1e-6)
    ap.add_argument("--out_dir", default="../result/procedural/library")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    build_library(args.assets_dir, k=args.k, min_area=args.min_area,
                  out_dir=args.out_dir, limit=args.limit, random_state=args.seed)
