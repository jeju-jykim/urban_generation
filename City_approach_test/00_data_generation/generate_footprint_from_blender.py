import bpy
from mathutils import Vector

# ========= 사용자 설정 =========
OUT_PATH = r"D:\App_dev\City\circle_tile.png"  # 출력 경로
RES = 4096                                     # 해상도 (정사각형, 1024 -> 4096)
MARGIN = 1.02                                   # 원 지름 대비 여유 (2%)
USE_CIRCULAR_MASK = False                       # True면 원형 마스크로 원 밖을 흰색 처리
# =================================

# 1) 기준 원 오브젝트: 현재 '선택된' 오브젝트 사용
circle = bpy.context.active_object
if circle is None:
    raise RuntimeError("원 오브젝트를 선택한 상태에서 실행하세요.")
if circle.type not in {'MESH', 'CURVE', 'SURFACE'}:
    raise RuntimeError("선택된 오브젝트가 원형 가이드(메시/커브)가 아닙니다.")

# 원 중심/지름 계산 (Dimensions X/Y는 월드 스케일 반영됨)
cx, cy, cz = circle.location.x, circle.location.y, circle.location.z
diam_x = circle.dimensions.x
diam_y = circle.dimensions.y
diam = max(diam_x, diam_y)
if diam <= 0:
    raise RuntimeError("원 지름이 0 이하로 감지되었습니다.")

# 2) 카메라 준비 (정사영 Top-Down)
# 기존 카메라 제거
for o in list(bpy.data.objects):
    if o.type == 'CAMERA':
        bpy.data.objects.remove(o, do_unlink=True)

bpy.ops.object.camera_add(location=(cx, cy, cz + 100.0))
cam = bpy.context.active_object
cam.data.type = 'ORTHO'
cam.data.ortho_scale = diam * MARGIN
cam.rotation_euler = (0, 0, 0)  # X축 90도 회전 제거, Z축 아래를 보는 Top-Down 뷰로 수정
cam.data.clip_start = 0.01
cam.data.clip_end   = 100000.0
bpy.context.scene.camera = cam

# 3) 렌더/컬러 세팅 (바이너리 또렷하게)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.eevee.taa_render_samples = 1 # 안티에일리어싱 비활성화 (샘플 1)
scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.render.film_transparent = False

scene.display_settings.display_device = 'sRGB'
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

# 배경 흰색
scene.world.use_nodes = True
wn = scene.world.node_tree.nodes
bg = wn.get("Background")
if bg:
    bg.inputs[0].default_value = (1,1,1,1)

# 4) 모든 건물 메시를 '검정 Emission'으로 (광원/그림자 영향 제거)
mat = bpy.data.materials.get("__MaskBlack")
if mat is None:
    mat = bpy.data.materials.new("__MaskBlack")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()
out = nodes.new("ShaderNodeOutputMaterial")
em  = nodes.new("ShaderNodeEmission")
em.inputs["Color"].default_value = (0,0,0,1)
em.inputs["Strength"].default_value = 1.0
links.new(em.outputs["Emission"], out.inputs["Surface"])

# 렌더 대상 Mesh에 적용 (원 가이드는 제외하려면 컬렉션 분리/숨김)
for o in bpy.data.objects:
    if o.type == 'MESH' and o != circle:
        if not o.data.materials:
            o.data.materials.append(mat)
        else:
            o.data.materials[0] = mat

# 5) (옵션) 원형 마스크로 Outside 제거
# - Film Transparent 켜고, Compositor에서 Ellipse Mask로 원 밖을 흰색으로
if USE_CIRCULAR_MASK:
    scene.render.film_transparent = True
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    # Nodes
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    alpha_over = nt.nodes.new("CompositorNodeAlphaOver")
    alpha_over.inputs[1].default_value = (1,1,1,1)  # 배경 흰색
    ellipse = nt.nodes.new("CompositorNodeEllipseMask")
    scale  = nt.nodes.new("CompositorNodeScale")

    # Ellipse Mask 설정 (정사각 해상도 가정 → 원 반지름 = 0.5/MARGIN)
    ellipse.width = 1.0 / MARGIN
    ellipse.height = 1.0 / MARGIN
    ellipse.x = 0.5
    ellipse.y = 0.5

    # 연결: RenderLayer → AlphaOver(전경) / 배경 흰색 → Composite
    nt.links.new(rl.outputs["Image"], alpha_over.inputs[2])
    nt.links.new(ellipse.outputs["Mask"], alpha_over.inputs["Fac"])
    nt.links.new(alpha_over.outputs["Image"], comp.inputs["Image"])

# 6) 렌더
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = OUT_PATH
bpy.ops.render.render(write_still=True)
print(f"[OK] circle center=({cx:.2f},{cy:.2f}), diameter={diam:.2f}, ortho_scale={cam.data.ortho_scale:.2f}")
print(f"[OK] saved → {OUT_PATH}")
