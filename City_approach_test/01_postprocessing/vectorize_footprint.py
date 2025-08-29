import cv2
import numpy as np
import json
import math

# --------- 경로 ----------
INPUT_IMAGE_PATH = r"D:\App_dev\City\result\500m_map\circle_tile3.png"
OUTPUT_SVG_PATH  = r"D:\App_dev\City\result\post_processing\footprint_vector3.svg"
OUTPUT_JSON_PATH = r"D:\App_dev\City\result\post_processing\footprint_polygons3.json"

# --------- 하이퍼파라미터 ----------
# 렌더를 크게 뽑았으면 여기서 축소해서 alias 줄이기 (예: 0.5)
DOWNSCALE = 0.5

# 형태학 커널(odd), 값 키우면 빈틈 메움/노이즈 제거 강해짐
K_CLOSE = 5   # closing: 구멍 메우기
K_OPEN  = 3   # opening: 노이즈 제거

# 너무 작은 폴리곤 제거
MIN_AREA_PX = 50

# Douglas-Peucker 비율(컨투어 둘레의 %) — 0.5~2% 사이를 많이 씀
APPROX_EPS_RATIO = 0.012

# (옵션) 직각/격자 스냅 보정용 그리드 간격(px). 0이면 사용 안함.
GRID_SNAP = 0  # 예: 1~2 픽셀

def snap_to_grid(p, g):
    if g <= 0: 
        return p
    return (int(round(p[0]/g)*g), int(round(p[1]/g)*g))

def polygon_to_svg_path(poly):
    # poly: (N,1,2) 또는 (N,2)
    pts = poly.reshape(-1,2)
    cmds = [f"M {pts[0,0]},{pts[0,1]}"]
    for i in range(1, len(pts)):
        cmds.append(f"L {pts[i,0]},{pts[i,1]}")
    cmds.append("Z")
    return " ".join(cmds)

# 1) 이미지 로드
img0 = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img0 is None:
    raise FileNotFoundError(INPUT_IMAGE_PATH)

# 2) 다운스케일(큰 해상도로 렌더했다면 0.5~0.75로 줄이기 권장)
if DOWNSCALE != 1.0:
    img0 = cv2.resize(img0, None, fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)

# 3) 이진화(건물이 검정이면 반전해서 흰색 객체로)
_, bin_inv = cv2.threshold(255-img0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4) 형태학(틈 메우고, 고립 픽셀 제거)
bin_clean = bin_inv.copy()
if K_CLOSE > 1:
    bin_clean = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, np.ones((K_CLOSE,K_CLOSE), np.uint8))
if K_OPEN > 1:
    bin_clean = cv2.morphologyEx(bin_clean, cv2.MORPH_OPEN,  np.ones((K_OPEN,K_OPEN),  np.uint8))

# 5) 컨투어 & 홀(내부) 계층 가져오기
contours, hierarchy = cv2.findContours(bin_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
if hierarchy is None:
    hierarchy = np.zeros((1,len(contours),4), dtype=np.int32)

H, W = bin_clean.shape
svg_paths = []
json_polygons = []

for i, cnt in enumerate(contours):
    # 외곽(Parent == -1)만 시작점
    if hierarchy[0][i][3] != -1:
        continue

    area = cv2.contourArea(cnt)
    if area < MIN_AREA_PX:
        continue

    # 매끈화(approx) — 비율 * 둘레
    eps = APPROX_EPS_RATIO * cv2.arcLength(cnt, True)
    outer = cv2.approxPolyDP(cnt, eps, True)

    # (옵션) 격자 스냅으로 계단 정리
    if GRID_SNAP > 0:
        outer = np.array([[[*snap_to_grid(p[0], GRID_SNAP)]] for p in outer], dtype=np.int32)

    # path 만들기
    path_d = polygon_to_svg_path(outer)

    building = {
        "outer": outer.reshape(-1,2).tolist(),
        "inner": []
    }

    # 자식(holes) 순회
    child = hierarchy[0][i][2]
    while child != -1:
        hole = contours[child]
        if cv2.contourArea(hole) >= MIN_AREA_PX:
            eps_in = APPROX_EPS_RATIO * cv2.arcLength(hole, True)
            inner = cv2.approxPolyDP(hole, eps_in, True)
            if GRID_SNAP > 0:
                inner = np.array([[[*snap_to_grid(p[0], GRID_SNAP)]] for p in inner], dtype=np.int32)

            path_d += " " + polygon_to_svg_path(inner)
            building["inner"].append(inner.reshape(-1,2).tolist())

        child = hierarchy[0][child][0]  # next sibling

    svg_paths.append(f'<path d="{path_d}" fill-rule="evenodd" fill="black"/>')
    json_polygons.append(building)

# 6) SVG 저장
with open(OUTPUT_SVG_PATH, "w", encoding="utf-8") as f:
    f.write(f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">\n')
    f.write('<rect width="100%" height="100%" fill="white"/>\n')
    for p in svg_paths:
        f.write(f"  {p}\n")
    f.write("</svg>\n")

# 7) JSON 저장
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(json_polygons, f, indent=2, ensure_ascii=False)

print("완료:", OUTPUT_SVG_PATH, OUTPUT_JSON_PATH)
