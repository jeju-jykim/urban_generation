# merge_library_from_labels.py
# -*- coding: utf-8 -*-
import os, csv, json, argparse
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image

# optional backends (used if installed)
try:
    import cairosvg
except Exception:
    cairosvg = None

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
except Exception:
    svg2rlg = None
    renderPM = None


# ---------- I/O ----------
def load_cluster_to_ids(features_path):
    cl2ids = {}
    with open(features_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cl = row["group"].strip()
            _id = row["id"].strip()
            cl2ids.setdefault(cl, []).append(_id)
    # 중복 제거(안전)
    for k, v in cl2ids.items():
        cl2ids[k] = sorted(list(dict.fromkeys(v)))
    return cl2ids

def load_naming(naming_path):
    with open(naming_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    clusters = data.get("clusters", {}) or {}
    mix = data.get("mix", None)
    if mix is None:
        # fallback: per-cluster share 그대로
        mix = {k: round(float(v.get("share", 0.0)), 3) for k, v in clusters.items()}
    label_set = data.get("label_set", None)
    if label_set is None:
        label_set = sorted({(v.get("label") or "").strip() for v in clusters.values() if v.get("label")})
    return clusters, mix, label_set

def build_library_json(cl2ids, naming_clusters, mix, active_labels=None, strict=False):
    """
    출력:
      - groups: {cluster_k: {"ids":[...]}}
      - labels: {cluster_k: {"label":"...", "use":"...", "use_hint":"...", "use_confidence":"..."}}
      - mix:    {cluster_k: share}
      - by_label: {label: {"cluster_keys":[...], "ids":[...]} }
      - label_mix: {label: summed share}
    """
    groups, labels, missing = {}, {}, []
    blank = []
    for cl_key, ids in cl2ids.items():
        info = naming_clusters.get(cl_key)
        if not info:
            missing.append(cl_key)
            if strict:
                raise SystemExit(f"[ERR] missing cluster in naming.json: {cl_key}")
            groups[cl_key] = {"ids": ids}
            labels[cl_key] = {"label": "", "use": "", "use_hint": "", "use_confidence": ""}
            continue

        label = (info.get("label") or "").strip()
        use = (info.get("use") or "").strip()
        # NEW: support use_hint/use_confidence
        use_hint = (info.get("use_hint") or "").strip()
        use_conf = (info.get("use_confidence") or "").strip()

        if not label:
            blank.append(cl_key)
            if strict:
                raise SystemExit(f"[ERR] blank label for cluster: {cl_key}")

        groups[cl_key] = {"ids": ids}
        labels[cl_key] = {
            "label": label,
            "use": use,  # 유지(역호환)
            "use_hint": use_hint,
            "use_confidence": use_conf
        }

    # 액티브 라벨 필터(선택): cluster 단위로 거르지 않고, 라벨 기준으로 거른다.
    if active_labels:
        active_labels = {s.strip() for s in active_labels if s.strip()}
        groups = {k:v for k,v in groups.items() if labels.get(k,{}).get("label") in active_labels}
        labels = {k:v for k,v in labels.items() if v.get("label") in active_labels}
        mix = {k:mix[k] for k in mix.keys() if k in groups}

    out = {
        "groups": groups,
        "labels": labels,
        "mix": mix,
        "n_total": sum(len(v["ids"]) for v in groups.values())
    }

    # 인덱스: by_label / label_mix (NEW)
    by_label = {}
    label_mix = {}
    for cl_key, meta in labels.items():
        lab = meta.get("label","")
        if not lab: continue
        by_label.setdefault(lab, {"cluster_keys": [], "ids": []})
        by_label[lab]["cluster_keys"].append(cl_key)
        by_label[lab]["ids"].extend(groups.get(cl_key, {}).get("ids", []))
        label_mix[lab] = label_mix.get(lab, 0.0) + float(mix.get(cl_key, 0.0))
    # 정렬 & dedup
    for lab, obj in by_label.items():
        obj["cluster_keys"] = sorted(obj["cluster_keys"])
        obj["ids"] = sorted(list(dict.fromkeys(obj["ids"])))

    out["by_label"] = by_label
    out["label_mix"] = {k: round(v, 3) for k, v in label_mix.items()}

    return out, missing, blank


# ---------- 대표 샘플 선택 ----------
def find_representative_ids(cl2ids, features_path):
    feats = {}
    with open(features_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            feats[row["id"]] = {"area": float(row["area"]), "group": row["group"]}
    reps = {}
    for cl_key, ids in cl2ids.items():
        arr = [feats[i]["area"] for i in ids if i in feats]
        if not arr:
            continue
        mu = float(np.mean(arr))
        reps[cl_key] = min(ids, key=lambda i: abs(feats[i]["area"] - mu) if i in feats else 1e18)
    return reps


# ---------- 렌더링 유틸 ----------
def rasterize_svg_to_pil(svg_path):
    if cairosvg is not None:
        png_bytes = cairosvg.svg2png(url=svg_path)
        return Image.open(BytesIO(png_bytes))
    if svg2rlg is not None and renderPM is not None:
        drawing = svg2rlg(svg_path)
        png_data = renderPM.drawToString(drawing, fmt="PNG")
        return Image.open(BytesIO(png_data))
    raise RuntimeError("No SVG rasterizer available (install cairosvg OR svglib+reportlab).")

def draw_from_json(ax, json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        geom = obj.get("footprint") or obj.get("geometry")
        if obj.get("type") == "Feature":
            geom = obj.get("geometry")
        elif obj.get("type") == "FeatureCollection":
            for ft in obj.get("features", []):
                if ft.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon"):
                    geom = ft["geometry"]; break
        if not geom:
            raise ValueError("No footprint geometry in JSON")

        import matplotlib.path as mpath
        import matplotlib.patches as mpatches
        Path = mpath.Path

        def draw_polygon(coords, face="#111"):
            codes, verts = [], []
            for ring in coords:
                if len(ring) < 3: continue
                verts.extend(ring + [ring[0]])
                codes.extend([Path.MOVETO] + [Path.LINETO]*(len(ring)-1) + [Path.CLOSEPOLY])
            ax.add_patch(mpatches.PathPatch(Path(verts, codes), facecolor=face, edgecolor="none"))

        ax.set_axis_off()
        if geom["type"] == "Polygon":
            draw_polygon(geom["coordinates"])
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                draw_polygon(poly)
        else:
            raise ValueError(f"Unsupported geom type: {geom['type']}")
        ax.autoscale_view()
        return True
    except Exception as e:
        ax.text(0.5, 0.5, f"JSON Error\n{e}", ha="center", va="center", fontsize=9)
        return False


# ---------- Figure ----------
def show_label_report_figure(naming_clusters, cl2ids, features_path, asset_dir, mix=None):
    if not naming_clusters:
        print("경고: naming.json의 'clusters'가 비어 있음.")
        return

    reps = find_representative_ids(cl2ids, features_path)
    items = sorted(naming_clusters.items(), key=lambda kv: kv[0])
    n = len(items)
    COLS = 4
    rows = (n + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS*3.4, rows*3.8), facecolor="white")
    fig.suptitle("LLM Cluster Labels & Representative Shapes", fontsize=18, y=0.97)
    axes = axes.flatten()

    for i, (cl_key, info) in enumerate(items):
        ax = axes[i]; ax.set_axis_off()
        label = (info.get("label") or "(no-label)")
        share = None
        if mix and cl_key in mix:
            share = mix[cl_key]
        title = f"{cl_key}  →  {label}"
        if share is not None:
            title += f"  ({share:.3f})"
        ax.set_title(title, fontsize=10, y=-0.18)

        rep_id = reps.get(cl_key)
        if not rep_id:
            ax.text(0.5, 0.5, "No Rep. ID", ha="center", va="center", fontsize=9)
            continue

        svg_path = os.path.join(asset_dir, f"{rep_id}_footprint_polygons.svg")
        json_path = os.path.join(asset_dir, f"{rep_id}_footprint_polygons.json")

        if os.path.exists(svg_path):
            try:
                pil_img = rasterize_svg_to_pil(svg_path)
                ax.imshow(pil_img)
            except Exception as e:
                print(f"[warn] SVG rasterize failed for {cl_key} ({svg_path}): {e}")
                if os.path.exists(json_path):
                    draw_from_json(ax, json_path)
                else:
                    ax.text(0.5, 0.5, "SVG Error", ha="center", va="center", fontsize=9)
        elif os.path.exists(json_path):
            draw_from_json(ax, json_path)
        else:
            ax.text(0.5, 0.5, "Asset Not Found", ha="center", va="center", fontsize=9)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Merge features.csv + naming.json (labels) → final library.json (+ by_label)")
    ap.add_argument("--features", default="../result/procedural/library/features.csv",
                    help="Path to features.csv with columns: id, ..., group")
    ap.add_argument("--naming", default="../result/procedural/library/naming.json",
                    help="Path to labeling JSON from LLM (clusters.{cluster_i}.*)")
    ap.add_argument("--out_dir", default="../result/procedural/library",
                    help="Output directory for library.json")
    ap.add_argument("--asset_dir", default="../result/separate_buildings_footprint",
                    help="Dir containing per-building *_footprint_polygons.(svg|json)")
    ap.add_argument("--active_labels", default="",
                    help="Optional comma-separated labels to keep (others dropped).")
    ap.add_argument("--strict", action="store_true",
                    help="Error on missing clusters or blank labels.")
    ap.add_argument("--preview", action="store_true",
                    help="Show grid figure of cluster → label + representative shape")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load
    cl2ids = load_cluster_to_ids(args.features)
    naming_clusters, mix, label_set = load_naming(args.naming)

    active = [s.strip() for s in args.active_labels.split(",")] if args.active_labels else None

    # merge → library.json (+ by_label / label_mix)
    library, missing, blank = build_library_json(cl2ids, naming_clusters, mix, active_labels=active, strict=args.strict)
    out_path = os.path.join(args.out_dir, "library.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)
    print("[saved]", out_path)
    print(f"- clusters in features.csv : {len(cl2ids)}")
    print(f"- clusters in naming.json  : {len(naming_clusters)}")
    if missing:
        print("! warning: clusters present in features.csv but missing in naming.json:")
        for k in missing: print("  -", k)
    if blank:
        print("! warning: clusters with blank label:", ", ".join(blank))
    print("\nLabel set:", ", ".join(label_set) if label_set else "(none)")
    print("Mix (first 10):", dict(list(library["mix"].items())[:10]))
    if "label_mix" in library:
        print("Label mix:", library["label_mix"])

    if args.preview:
        show_label_report_figure(naming_clusters, cl2ids, args.features, args.asset_dir, mix=library.get("mix"))

if __name__ == "__main__":
    main()
