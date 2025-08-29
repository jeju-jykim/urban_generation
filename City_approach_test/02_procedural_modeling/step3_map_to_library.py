# map_to_library.py
# -*- coding: utf-8 -*-
"""
One-shot:
- (A) If inputs are images: binarize → footprints → label with kNN trained from features.csv + naming.json
- (B) If inputs are typed GeoJSONs: read as-is
- (C) Analyze typed maps: snaps/rows/spacing/coverage/use_probs...
- (D) Merge the context into an existing library.json and save to --out_library

Usage examples:

python map_to_library.py \
  --maps "../result/500m_map/*.png" \
  --size 500 500 \
  --features ../result/procedural/features.csv \
  --naming  ../result/procedural/naming.json \
  --in_library ../result/procedural/library_5/library.json \
  --out_library ../result/procedural/library_5/library_with_context.json \
  --write_typed_dir ../result/procedural/typed_maps

# If your inputs are already typed_map*.geojson:
python map_to_library.py \
  --maps "../result/procedural/typed_map*.geojson" \
  --in_library ../result/procedural/library_5/library.json \
  --out_library ../result/procedural/library_5/library_with_context.json

# Optionally constrain labels to a small set (e.g., 5 archetypes):
  --active_labels "Micro Tower,Tower,Irregular Block,Bar Slab,Long Slab"

# Optionally override library label_mix (and per-cluster mix) from the maps:
  --override_mix_from_maps
"""

import os, glob, json, math, csv, warnings
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

import numpy as np

# Optional backends (image → polys)
try:
    import cv2
except Exception:
    cv2 = None

from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────────
# Geometry & feature utils
# ─────────────────────────────────────────────────────────────
def obb_params(poly: Polygon) -> Tuple[float, float, float]:
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
    if ang < 0: ang += 180.0
    return width, depth, ang

def poly_features(p: Polygon) -> List[float]:
    area = float(p.area)
    w, d, _ = obb_params(p)
    ar = w / max(d, 1e-9)
    convexity = area / max(p.convex_hull.area, 1e-9)
    complexity = p.length / max(area**0.5, 1e-9)
    # Support both Polygon and MultiPolygon when counting holes
    if isinstance(p, MultiPolygon):
        holes = sum(len(g.interiors) for g in p.geoms)
    else:
        holes = len(getattr(p, "interiors", []))
    return [math.log(area+1e-9), ar, 1.0-convexity, complexity, float(holes)]

def image_to_polys(png_path: str, W: float, H: float, min_area: float=5.0) -> List[Polygon]:
    if cv2 is None:
        raise SystemExit("[ERR] OpenCV not available. Install opencv-python to read images.")
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"[ERR] cannot read image: {png_path}")
    h, w = img.shape
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sx, sy = W/float(w), H/float(h)
    polys=[]
    for c in cnts:
        if len(c) < 3: continue
        # origin at center
        pts = [(float(x)*sx - W/2.0, float(y)*sy - H/2.0) for [[x,y]] in c]
        p = Polygon(pts).buffer(0)
        if not p.is_valid or p.area < min_area: continue
        polys.append(p)
    return polys

# ─────────────────────────────────────────────────────────────
# Library features → scaler + kNN (labels are strings from naming.json)
# ─────────────────────────────────────────────────────────────
def load_library_feats(features_csv: str, naming_json: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(naming_json, "r", encoding="utf-8") as f:
        naming = json.load(f)
    cl2label = {k:(v.get("label") or k) for k,v in (naming.get("clusters") or {}).items()}

    X, y = [], []
    with open(features_csv, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                X.append([
                    math.log(float(row["area"])+1e-9),
                    float(row["ar"]),
                    1.0 - float(row["convexity"]),
                    float(row["complexity"]),
                    float(row["holes"])
                ])
                y.append(cl2label.get(row["group"], row["group"]))
            except Exception:
                continue
    if not X:
        raise SystemExit("[ERR] No rows loaded from features.csv")
    return np.array(X, dtype=np.float64), np.array(y, dtype=object)

def fit_label_knn(X: np.ndarray, y: np.ndarray, n_neighbors=7):
    scaler = StandardScaler().fit(X)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(scaler.transform(X), y)
    return scaler, knn

# ─────────────────────────────────────────────────────────────
# Typed features (labeling) for a single map input
# ─────────────────────────────────────────────────────────────
def typed_from_polys(polys: List[Polygon], scaler, knn, active_labels: Optional[List[str]]=None):
    feats = [poly_features(p) for p in polys]
    F = np.array(feats, dtype=np.float64)
    yhat = knn.predict(scaler.transform(F))

    # If active_labels is specified, remap labels not in the set to nearest allowed by neighbor vote
    if active_labels:
        active = set([s.strip() for s in active_labels if s.strip()])
        if active:
            dist, idx = knn.kneighbors(scaler.transform(F), n_neighbors=min(15, len(knn.classes_)))
            # idx: neighbor indices in training set; knn._y holds labels of training rows
            ref_labels = knn._y
            new_yhat = []
            for i, lbl in enumerate(yhat):
                if lbl in active:
                    new_yhat.append(lbl); continue
                cand_counts = Counter()
                for j in idx[i]:
                    lab = ref_labels[j]
                    if lab in active:
                        cand_counts[lab] += 1
                if cand_counts:
                    new_lbl = cand_counts.most_common(1)[0][0]
                else:
                    # fallback: keep original
                    new_lbl = lbl
                new_yhat.append(new_lbl)
            yhat = np.array(new_yhat, dtype=object)

    typed = []
    for p, lab in zip(polys, yhat):
        w,d,theta = obb_params(p)
        cx, cy = p.centroid.x, p.centroid.y
        typed.append({
            "type": "Feature",
            "properties": {
                "label": str(lab),
                "theta": float(theta),
                "centroid": [float(cx), float(cy)],
                "area": float(p.area),
                "obb_long": float(max(w,d)),
                "obb_short": float(min(w,d))
            },
            "geometry": mapping(p)
        })
    return {"type":"FeatureCollection", "features": typed}

def typed_from_geojson(path: str):
    gj = json.load(open(path, "r", encoding="utf-8"))
    # pass-through; ensure required props
    feats=[]
    for ft in gj.get("features", []):
        g = shape(ft["geometry"]).buffer(0)
        pr = ft.get("properties", {})
        # fill missing basics
        w,d,theta = obb_params(g)
        pr.setdefault("theta", theta)
        c = g.centroid; pr.setdefault("centroid", [float(c.x), float(c.y)])
        pr.setdefault("area", float(g.area))
        pr.setdefault("obb_long", float(max(w,d)))
        pr.setdefault("obb_short", float(min(w,d)))
        feats.append({"type":"Feature","properties":pr,"geometry":mapping(g)})
    return {"type":"FeatureCollection","features":feats}

# ─────────────────────────────────────────────────────────────
# Map analysis → context report (per map)
# ─────────────────────────────────────────────────────────────
def dominant_snaps_deg(thetas: List[float]):
    if len(thetas) < 2:
        if not thetas: return [0.0],[1.0]
        return [round(float(thetas[0]),1)],[1.0]
    ang = np.radians(np.array(thetas)*2.0)
    X = np.c_[np.cos(ang), np.sin(ang)]
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    snaps, shares = [], []
    for k in [0,1]:
        idx = (km.labels_ == k)
        if idx.sum() == 0: continue
        mean_angle = math.degrees(math.atan2(np.mean(np.sin(ang[idx])), np.mean(np.cos(ang[idx]))))/2.0
        if mean_angle < 0: mean_angle += 180.0
        snaps.append(round(mean_angle,1))
        shares.append(int(idx.sum()))
    s = sum(shares) or 1
    return snaps, [round(c/s,3) for c in shares]

def uv_axes(deg):
    th = math.radians(deg)
    return (math.cos(th), math.sin(th)), (-math.sin(th), math.cos(th))  # u, v

def project_uv(xy, u, v, c):
    x,y=xy; cx,cy=c
    return ((x-cx)*u[0] + (y-cy)*u[1], (x-cx)*v[0] + (y-cy)*v[1])

def analyze_typed_map(typed_fc: dict, fname: str):
    geoms=[]; labels=[]; thetas=[]; shorts=[]
    for ft in typed_fc.get("features", []):
        g = shape(ft["geometry"]).buffer(0)
        geoms.append(g)
        pr = ft.get("properties", {})
        labels.append(pr.get("label",""))
        thetas.append(pr.get("theta", 0.0))
        shorts.append(pr.get("obb_short", obb_params(g)[1]))
    if not geoms:
        return None

    U = unary_union(geoms)
    minx,miny,maxx,maxy = U.bounds
    W,H = maxx-minx, maxy-miny
    C=((minx+maxx)/2,(miny+maxy)/2)

    # slab-like labels
    slab_set = {"Bar Slab","Long Slab","Articulated Slab","Ribbon Slab","Landmark Slab"}
    slabs = [(shape(ft["geometry"]), ft["properties"]) for ft in typed_fc["features"]
             if ft["properties"].get("label") in slab_set]
    slab_thetas=[p[1].get("theta",0.0) for p in slabs]
    snaps, shares = dominant_snaps_deg(slab_thetas) if slab_thetas else ([0.0],[1.0])
    main = snaps[0]

    short_med = float(np.median(shorts)) if shorts else 20.0
    u,v = uv_axes(main)

    # row detection (simple 1D gap clustering on v)
    vvals=[]
    for g,pr in slabs:
        cx,cy = g.centroid.x, g.centroid.y
        uu,vv = project_uv((cx,cy), u, v, C)
        vvals.append(vv)
    vvals = sorted(vvals)
    rows_v = []
    if vvals:
        thresh = 1.25*short_med
        cur=[vvals[0]]
        for a,b in zip(vvals, vvals[1:]):
            if abs(b - a) <= thresh: cur.append(b)
            else: rows_v.append(float(np.median(cur))); cur=[b]
        if cur: rows_v.append(float(np.median(cur)))
    row_count=len(rows_v)

    # along gaps
    along_gaps=[]
    if row_count>0:
        assigns=[[] for _ in range(row_count)]
        for g,pr in slabs:
            cx,cy = g.centroid.x, g.centroid.y
            uu,vv = project_uv((cx,cy), u, v, C)
            k=int(np.argmin([abs(vv-r) for r in rows_v]))
            if abs(vv-rows_v[k])<=1.25*short_med:
                assigns[k].append(uu)
        for row in assigns:
            if len(row)>=2:
                s=sorted(row)
                along_gaps += [s[i+1]-s[i] for i in range(len(s)-1)]

    row_spacing = float(np.median(np.diff(sorted(rows_v)))) if len(rows_v)>=2 else None
    along_spacing = float(np.median(along_gaps)) if along_gaps else None
    band_width = float(1.2*short_med)

    # use tagging
    def dist_to_edge(pt):
        x,y=pt; dx=min(x-minx, maxx-x); dy=min(y-miny, maxy-y); return min(dx,dy)
    edge_T = max(5.0, 0.5*(row_spacing or 20.0))
    corner_T = max(8.0, 0.8*(row_spacing or 20.0))

    def tag_use(g, lbl):
        cx,cy = g.centroid.x, g.centroid.y
        uu,vv = project_uv((cx,cy), u, v, C)
        row_member=False
        if row_count>0:
            k=int(np.argmin([abs(vv-r) for r in rows_v]))
            row_member = (abs(vv-rows_v[k])<=1.25*short_med)
        d_edge = dist_to_edge((cx,cy))
        edge = d_edge < edge_T
        corner = min(
            ((cx-minx)**2+(cy-miny)**2)**0.5,
            ((cx-minx)**2+(cy-maxy)**2)**0.5,
            ((cx-maxx)**2+(cy-miny)**2)**0.5,
            ((cx-maxx)**2+(cy-maxy)**2)**0.5
        ) < corner_T
        if lbl in slab_set and row_member: use="row"
        elif "Block" in lbl: use="courtyard"
        else: use="isolated"
        return use, edge, corner

    cnt = Counter(labels)
    use_counts=defaultdict(Counter)
    edge_counts=Counter(); corner_counts=Counter()
    for ft in typed_fc["features"]:
        g = shape(ft["geometry"])
        lbl = ft["properties"].get("label","")
        use, edge, corner = tag_use(g, lbl)
        use_counts[lbl][use]+=1
        edge_counts[lbl]+=int(edge)
        corner_counts[lbl]+=int(corner)

    coverage = U.area/(W*H) if (W>0 and H>0) else None

    return {
        "file": os.path.basename(fname),
        "size_m": {"W": round(W,1), "H": round(H,1)},
        "counts": dict(cnt),
        "coverage": round(coverage,3) if coverage is not None else None,
        "snaps": snaps, "snap_shares": shares,
        "rows": {
            "row_count": row_count,
            "row_spacing": round(row_spacing,1) if row_spacing else None,
            "along_spacing": round(along_spacing,1) if along_spacing else None,
            "band_width": round(band_width,1)
        },
        "use_counts": {lab: dict(use_counts[lab]) for lab in use_counts},
        "edge_counts": dict(edge_counts),
        "corner_counts": dict(corner_counts)
    }

# ─────────────────────────────────────────────────────────────
# Aggregate context across maps
# ─────────────────────────────────────────────────────────────
def aggregate_context(reports: List[dict]) -> dict:
    if not reports: return {"global": {}, "labels": {}, "maps": []}
    all_labels=set()
    for r in reports:
        all_labels |= set(r["counts"].keys())
    ctx={"global": {}, "labels": {}, "maps": reports}

    sizes=[(r["size_m"]["W"], r["size_m"]["H"]) for r in reports]
    ctx["global"]["site_sizes"]=sizes
    snaps=[deg for r in reports for deg in r.get("snaps",[])]
    if snaps:
        ctx["global"]["snap_angles_deg"]=sorted(list({round(s,1) for s in snaps}))
    cov=[r["coverage"] for r in reports if r.get("coverage") is not None]
    if cov:
        ctx["global"]["coverage_mean"]=float(np.mean(cov))

    for lab in sorted(all_labels):
        uc=[]; ec=[]; cc=[]
        rows_sp=[]; along_sp=[]; band_w=[]
        lab_snaps=[]
        for r in reports:
            n=r["counts"].get(lab,0)
            if n==0: continue
            uc.append(r.get("use_counts",{}).get(lab,{}))
            ec.append((r.get("edge_counts",{}).get(lab,0), r["counts"].get(lab,0)))
            cc.append((r.get("corner_counts",{}).get(lab,0), r["counts"].get(lab,0)))
            if r["rows"]["row_spacing"]: rows_sp.append(r["rows"]["row_spacing"])
            if r["rows"]["along_spacing"]: along_sp.append(r["rows"]["along_spacing"])
            if r["rows"]["band_width"]: band_w.append(r["rows"]["band_width"])
            lab_snaps += r.get("snaps",[])
        total=Counter()
        for d in uc: total.update(d)
        total_n = sum(total.values()) or 1
        use_probs={k: round(v/total_n,3) for k,v in total.items()}
        edge_rate = sum(e for e,_ in ec)/max(1, sum(n for _,n in ec)) if ec else None
        corner_rate = sum(c for c,_ in cc)/max(1, sum(n for _,n in cc)) if cc else None
        ctx["labels"][lab]={
            "use_probs": use_probs,
            "row_stats": {
                "row_spacing_med": float(np.median(rows_sp)) if rows_sp else None,
                "along_spacing_med": float(np.median(along_sp)) if along_sp else None,
                "band_width_med": float(np.median(band_w)) if band_w else None
            },
            "angle_stats": {
                "snap_deg": sorted(list({round(a,1) for a in lab_snaps})) if lab_snaps else None
            },
            "edge_rate": round(edge_rate,3) if edge_rate is not None else None,
            "corner_rate": round(corner_rate,3) if corner_rate is not None else None
        }
    return ctx

def label_mix_from_reports(reports: List[dict]) -> Dict[str,float]:
    c = Counter()
    for r in reports:
        c.update(r.get("counts", {}))
    total = sum(c.values()) or 1
    return {lab: round(v/total,3) for lab,v in c.items()}

def override_mix_in_library(lib: dict, label_mix_new: dict) -> dict:
    """Optional: override label_mix and per-cluster mix proportionally within each label group (if by_label exists)."""
    lib = json.loads(json.dumps(lib))  # deep copy
    # Update label_mix
    lib.setdefault("label_mix", {})
    for lab, share in label_mix_new.items():
        lib["label_mix"][lab] = share

    # If by_label & mix present, distribute within clusters
    if "by_label" in lib and "mix" in lib:
        new_mix = {}
        for lab, obj in lib["by_label"].items():
            keys = obj.get("cluster_keys", [])
            if not keys: continue
            prev = np.array([lib["mix"].get(k, 0.0) for k in keys], dtype=np.float64)
            s = float(prev.sum())
            if s <= 1e-9:
                # equal distribution
                w = np.ones(len(keys), dtype=np.float64) / len(keys)
            else:
                w = prev / s
            target = label_mix_new.get(lab, s)  # if lab not present, keep previous sum
            for k, wk in zip(keys, w):
                new_mix[k] = round(float(target * wk), 3)
        # keep clusters not mapped to any label as-is
        for k,v in lib.get("mix",{}).items():
            if k not in new_mix:
                new_mix[k] = v
        lib["mix"] = new_mix
    return lib

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Maps → typed GeoJSON (if PNG) → context analysis → library merge")
    ap.add_argument("--maps", nargs="+", required=True, help="PNG or typed_map GeoJSON; glob patterns allowed")
    ap.add_argument("--size", nargs=2, type=float, metavar=("W","H"), help="Size in meters for PNG inputs (e.g., 500 500)")
    ap.add_argument("--min_area", type=float, default=5.0, help="Min polygon area when vectorizing PNG")
    ap.add_argument("--features", default='../result/procedural/library/features.csv', help="features.csv (required if any input is PNG)")
    ap.add_argument("--naming", default='../result/procedural/library/naming.json', help="naming.json (required if any input is PNG)")
    ap.add_argument("--k_neighbors", type=int, default=7)
    ap.add_argument("--in_library", default='../result/procedural/library/library.json', help="Existing library.json to merge context into")
    ap.add_argument("--out_library", default='../result/procedural/library/library_with_context.json', help="Path to write updated library with 'context'")
    ap.add_argument("--active_labels", default="", help="Optional comma-separated labels to restrict predictions to")
    ap.add_argument("--write_typed_dir", default="../result/procedural/library/typed_maps", help="Optional dir to save typed_map_<basename>.geojson")
    ap.add_argument("--override_mix_from_maps", action="store_true", help="Override library label_mix (and per-cluster mix proportionally) from maps")
    args = ap.parse_args()

    # python step3_map_to_library.py --maps "../result/500m_map/circle_tile.png" "../result/500m_map/circle_tile2.png" "../result/500m_map/circle_tile3.png" --size 500 500 --override_mix_from_maps

    # Expand globs
    paths=[]
    for p in args.maps:
        paths += glob.glob(p)
    if not paths:
        raise SystemExit("[ERR] No inputs matched with --maps")

    # Prepare classifier if there is any PNG
    has_png = any(p.lower().endswith((".png",".jpg",".jpeg","bmp","tif","tiff")) for p in paths)
    scaler = knn = None
    if has_png:
        if not args.size or not args.features or not args.naming:
            raise SystemExit("[ERR] PNG inputs require --size W H, --features, and --naming")
        Xlib, ylib = load_library_feats(args.features, args.naming)
        scaler, knn = fit_label_knn(Xlib, ylib, n_neighbors=args.k_neighbors)

    active_labels = [s.strip() for s in args.active_labels.split(",")] if args.active_labels else None

    # Process each map → typed features → analyze
    reports=[]
    os.makedirs(args.write_typed_dir, exist_ok=True) if args.write_typed_dir else None

    for p in paths:
        ext = os.path.splitext(p.lower())[1]
        if ext in (".png",".jpg",".jpeg",".bmp",".tif",".tiff"):
            W,H = args.size
            polys = image_to_polys(p, W, H, min_area=args.min_area)
            typed = typed_from_polys(polys, scaler, knn, active_labels=active_labels)
            if args.write_typed_dir:
                outp = os.path.join(args.write_typed_dir, f"typed_map_{os.path.splitext(os.path.basename(p))[0]}.geojson")
                json.dump(typed, open(outp,"w",encoding="utf-8"))
                print("[typed]", outp)
            typed_fc = typed
        else:
            typed_fc = typed_from_geojson(p)
            # optional filter to active labels
            if active_labels:
                for ft in typed_fc["features"]:
                    lbl = ft["properties"].get("label","")
                    if lbl not in active_labels:
                        # leave as-is or tag as 'Other' — here we keep as-is for transparency
                        pass

        rep = analyze_typed_map(typed_fc, p)
        if rep: reports.append(rep)

    # Aggregate context
    ctx = aggregate_context(reports)

    # Merge into library
    lib = json.load(open(args.in_library,"r",encoding="utf-8"))
    lib["context"] = ctx

    # Optional: override label_mix (and per-cluster mix proportionally) using maps
    if args.override_mix_from_maps:
        lm = label_mix_from_reports(reports)
        lib = override_mix_in_library(lib, lm)

    json.dump(lib, open(args.out_library,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[saved] {args.out_library}  (maps processed: {len(reports)})")

if __name__ == "__main__":
    main()
