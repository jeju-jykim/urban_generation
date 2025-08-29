# -*- coding: utf-8 -*-
"""
단일 STL에서 투영-footprint(2D)와 메타데이터를 추출해
SVG/JSON으로 저장하는 최소 스크립트.

pip install trimesh shapely numpy
python stl_to_footprint.py --in path/to/model.stl --out_json out.json --out_svg out.svg \
    --method support --simplify 0.05
"""
import argparse, json, math, os, numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from shapely.affinity import translate

# -----------------------
# 유틸
# -----------------------
def _poly_from_xy(xy):
    if len(xy) < 3: 
        return None
    try:
        p = Polygon(xy)
        if p.is_valid and p.area > 0:
            return p
    except Exception:
        return None
    return None

def _triangles_to_union(tris_xy):
    polys=[]
    for tri in tris_xy:
        p = _poly_from_xy(tri)
        if p is not None:
            polys.append(p)
    if not polys:
        return None
    return unary_union(polys).buffer(0)

def _auto_unit_scale(mesh):
    # STL이 mm인 경우가 많음 → extents가 1000 이상이면 m로 스케일링
    ext = mesh.extents.max()
    if ext > 1000.0:
        mesh.apply_scale(1.0/1000.0)
        return "mm->m"
    return "as-is"

def _obb_params(poly):
    rect = poly.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    pts = list(zip(xs, ys))[:4]
    # 변 길이
    e01 = np.hypot(pts[1][0]-pts[0][0], pts[1][1]-pts[0][1])
    e12 = np.hypot(pts[2][0]-pts[1][0], pts[2][1]-pts[1][1])
    width, depth = (max(e01, e12), min(e01, e12))
    # 장변 방향 각도
    if e01 >= e12:
        dx, dy = pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]
    else:
        dx, dy = pts[2][0]-pts[1][0], pts[2][1]-pts[1][1]
    theta = math.degrees(math.atan2(dy, dx))
    return rect, width, depth, theta

def _poly_to_svg_path(poly):
    def ring_to_d(r):
        xs, ys = r.coords.xy
        coords = list(zip(xs, ys))
        return "M " + " L ".join([f"{x:.3f},{y:.3f}" for x,y in coords]) + " Z"
    d = ring_to_d(poly.exterior)
    for h in poly.interiors:
        d += " " + ring_to_d(h)
    return d

# -----------------------
# 핵심: footprint 계산
# -----------------------
def compute_footprint(mesh: trimesh.Trimesh, method="support", z_eps_ratio=0.05):
    """
    method:
      - 'shadow'  : 전체 메쉬를 수직투영한 그림자 합집합(간단, 오버슈트 가능)
      - 'support' : 바닥 근처의 수직벽/하부 지지부만 투영(빌딩 footprint에 근접)
    z_eps_ratio: (z_max - z_min) * ratio 를 바닥 근처 임계로 사용
    """
    V = mesh.vertices
    F = mesh.faces
    tris = V[F]  # (N, 3, 3)
    # 바닥 범위
    zmin = V[:,2].min(); zmax = V[:,2].max()
    z_eps = max(1e-4, (zmax - zmin) * z_eps_ratio)

    if method == "shadow":
        # 모든 삼각형을 XY로 투영
        tris_xy = [ [(p[0], p[1]) for p in tri] for tri in tris ]
        return _triangles_to_union(tris_xy)

    # support 모드: 바닥 근처 & 수직벽 위주
    # 1) 바닥 근접 삼각형(최소 z가 zmin+z_eps 이하)
    near_floor = np.min(tris[:,:,2], axis=1) <= (zmin + z_eps)
    # 2) 수직벽(법선의 z성분이 작음)
    norms = mesh.face_normals  # unit
    vertical = np.abs(norms[:,2]) < 0.5
    mask = near_floor | vertical
    sel = tris[mask]
    if sel.size == 0:
        sel = tris  # fallback

    tris_xy = [ [(p[0], p[1]) for p in tri] for tri in sel ]
    return _triangles_to_union(tris_xy)

# -----------------------
# 파이프라인
# -----------------------
def stl_to_footprint(stl_path, method="support", simplify_tol=0.05):
    mesh = trimesh.load_mesh(stl_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.dump().values()))
    scale_info = _auto_unit_scale(mesh)

    # 수치 안정화: 살짝 원점 이동
    mesh.apply_translation(-mesh.vertices.mean(axis=0))

    # footprint 계산
    poly = compute_footprint(mesh, method=method)

    if poly is None or poly.is_empty:
        raise RuntimeError("footprint를 만들 수 없습니다. 메쉬를 확인하세요.")

    # 멀티폴리곤이면 외곽이 큰 것 하나를 메인으로 사용(필요하면 Multi 그대로 저장)
    if isinstance(poly, MultiPolygon):
        parts = sorted(list(poly.geoms), key=lambda p: p.area, reverse=True)
        poly_main = parts[0]
        holes = parts[1:]
        poly = poly_main.difference(unary_union(holes)) if holes else poly_main

    # 단순화/clean
    if simplify_tol and simplify_tol > 0:
        poly = poly.simplify(simplify_tol, preserve_topology=True).buffer(0)

    # OBB/메타
    rect, w, d, theta = _obb_params(poly)
    area = poly.area
    perim = poly.length
    ar = (w / max(d, 1e-9))
    cx, cy = poly.centroid.x, poly.centroid.y

    meta = {
        "asset_id": os.path.splitext(os.path.basename(stl_path))[0],
        "units": "meters",
        "scale_info": scale_info,
        "method": method,
        "simplify_tol": simplify_tol,
        "centroid": [cx, cy],
        "area_m2": float(area),
        "perimeter_m": float(perim),
        "obb": {
            "width_m": float(w),
            "depth_m": float(d),
            "aspect_ratio": float(ar),
            "theta_deg": float(theta),
            "polygon": list(map(list, np.array(rect.exterior.coords)))
        },
        # front_vec을 알 수 없으면 OBB 장변 방향을 기본 전면으로 둔다
        "front_vec_hint": [math.cos(math.radians(theta)), math.sin(math.radians(theta))]
    }

    # GeoJSON 호환 좌표
    poly_geo = mapping(poly)

    return poly, meta, poly_geo

# -----------------------
# I/O
# -----------------------
def save_svg(poly: Polygon, svg_path, stroke=False):
    minx,miny,maxx,maxy = poly.bounds
    pad = 5.0
    w = maxx-minx+2*pad; h = maxy-miny+2*pad
    d = _poly_to_svg_path(translate(poly, xoff=pad-minx, yoff=pad-miny))
    stroke_style = 'fill="black"' if not stroke else 'fill="none" stroke="black" stroke-width="1"'
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.1f}" height="{h:.1f}" viewBox="0 0 {w:.1f} {h:.1f}">\n'
    svg += f'  <path d="{d}" {stroke_style}/>\n'
    svg += '</svg>\n'
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

def main():
    ap = argparse.ArgumentParser(description="폴더 내의 모든 STL 파일에 대한 2D footprint를 추출하고 JSON/SVG로 저장합니다.")
    ap.add_argument("--in_dir", default='./data/separate_buildings', help="입력 STL 파일들이 있는 폴더 경로")
    ap.add_argument("--out_dir", default='./result/separate_buildings_footprint', help="결과 JSON/SVG 파일들을 저장할 폴더 경로")
    ap.add_argument("--method", choices=["support","shadow"], default="shadow", help="Footprint 추출 방식 ('support' 권장)")
    ap.add_argument("--simplify", type=float, default=0.05, help="단순화 톨러런스(m), 0이면 비활성화")
    args = ap.parse_args()

    # 입력 폴더 유효성 검사
    if not os.path.isdir(args.in_dir):
        print(f"오류: 입력 폴더를 찾을 수 없습니다 -> {args.in_dir}")
        return

    # 출력 폴더 생성
    os.makedirs(args.out_dir, exist_ok=True)

    # STL 파일 목록 가져오기
    stl_files = [f for f in os.listdir(args.in_dir) if f.lower().endswith('.stl')]
    if not stl_files:
        print(f"경고: '{args.in_dir}' 폴더에 STL 파일이 없습니다.")
        return

    print(f"총 {len(stl_files)}개의 STL 파일을 처리합니다...")
    success_count = 0

    # 각 파일 처리
    for i, filename in enumerate(stl_files):
        in_path = os.path.join(args.in_dir, filename)
        base_name = os.path.splitext(filename)[0]
        
        # 출력 파일 경로 생성
        out_json_path = os.path.join(args.out_dir, f"{base_name}_footprint_polygons.json")
        out_svg_path = os.path.join(args.out_dir, f"{base_name}_footprint_polygons.svg")

        print(f"[{i+1}/{len(stl_files)}] 처리 중: {filename}")

        try:
            # 핵심 로직 호출
            poly, meta, poly_geo = stl_to_footprint(in_path, method=args.method, simplify_tol=args.simplify)

            # JSON 저장
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump({"meta": meta, "footprint": poly_geo}, f, ensure_ascii=False, indent=2)

            # SVG 저장
            save_svg(poly, out_svg_path)
            
            success_count += 1

        except Exception as e:
            print(f"  -> 오류 발생: {filename} 처리 실패. ({e})")

    print("\n[완료]")
    print(f"성공: {success_count} / {len(stl_files)}")
    print(f"결과가 '{args.out_dir}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()
