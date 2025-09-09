import bpy

# 활성 오브젝트 확인
obj = bpy.context.active_object
if obj is None or obj.type != 'MESH':
    # Plane 자동 생성
    bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, 0))
    obj = bpy.context.active_object
    print("✅ Created 50x50 Plane")

NG_NAME = "Proc_Building_Layout"

# 기존 노드 그룹 제거 후 새로 생성
if NG_NAME in bpy.data.node_groups:
    bpy.data.node_groups.remove(bpy.data.node_groups[NG_NAME])

ng = bpy.data.node_groups.new(NG_NAME, 'GeometryNodeTree')

# 입출력 설정 - Building Count 제거
ng.inputs.clear()
ng.outputs.clear()
ng.inputs.new("NodeSocketGeometry", "Geometry")
ng.inputs.new("NodeSocketFloat", "Min Spacing").default_value = 2.5  # 빽빽하게 시작
ng.inputs.new("NodeSocketInt", "Seed").default_value = 0
ng.inputs.new("NodeSocketFloat", "Height Min").default_value = 5.0
ng.inputs.new("NodeSocketFloat", "Height Max").default_value = 15.0
ng.outputs.new("NodeSocketGeometry", "Geometry")

nodes = ng.nodes
links = ng.links
nodes.clear()

# IO 노드
group_in = nodes.new("NodeGroupInput")
group_in.location = (-800, 0)
group_out = nodes.new("NodeGroupOutput")
group_out.location = (800, 0)

# Poisson 배치 - Density 최대로 설정
dist = nodes.new("GeometryNodeDistributePointsOnFaces")
dist.location = (-400, 0)
dist.distribute_method = 'POISSON'
dist.inputs["Density"].default_value = 100.0  # 높은 밀도로 고정

links.new(group_in.outputs["Geometry"], dist.inputs["Mesh"])
links.new(group_in.outputs["Min Spacing"], dist.inputs["Distance Min"])
links.new(group_in.outputs["Seed"], dist.inputs["Seed"])

# Cube 인스턴스 - 2x2x1 고정 크기
cube = nodes.new("GeometryNodeMeshCube")
cube.location = (-400, -200)
cube.inputs["Size"].default_value = (2.0, 2.0, 1.0)  # 2x2x1 고정

inst = nodes.new("GeometryNodeInstanceOnPoints")
inst.location = (0, 0)
links.new(dist.outputs["Points"], inst.inputs["Points"])
links.new(cube.outputs["Mesh"], inst.inputs["Instance"])

# Random Height만 적용 (Z축 스케일)
rand_h = nodes.new("FunctionNodeRandomValue")
rand_h.data_type = 'FLOAT'
rand_h.location = (-200, -150)
links.new(group_in.outputs["Height Min"], rand_h.inputs[2])
links.new(group_in.outputs["Height Max"], rand_h.inputs[3])
links.new(group_in.outputs["Seed"], rand_h.inputs["Seed"])

# Scale - XY는 1.0 고정, Z만 높이
comb_scale = nodes.new("ShaderNodeCombineXYZ")
comb_scale.location = (0, -150)
comb_scale.inputs["X"].default_value = 1.0  # X 고정
comb_scale.inputs["Y"].default_value = 1.0  # Y 고정
links.new(rand_h.outputs[1], comb_scale.inputs["Z"])
links.new(comb_scale.outputs["Vector"], inst.inputs["Scale"])

# 바닥에 앉히기 - 간단한 Transform 사용
transform = nodes.new("GeometryNodeTransform")
transform.location = (200, 0)
# Z = (높이/2) 평균값으로 간단히
transform.inputs["Translation"].default_value = (0, 0, 5.0)  # 평균 높이의 절반
links.new(inst.outputs["Instances"], transform.inputs["Geometry"])

# Realize Instances
realz = nodes.new("GeometryNodeRealizeInstances")
realz.location = (400, 0)
links.new(transform.outputs["Geometry"], realz.inputs["Geometry"])

# Join with original plane
join = nodes.new("GeometryNodeJoinGeometry")
join.location = (600, 0)
links.new(group_in.outputs["Geometry"], join.inputs["Geometry"])
links.new(realz.outputs["Geometry"], join.inputs["Geometry"])
links.new(join.outputs["Geometry"], group_out.inputs["Geometry"])

# 모디파이어 적용
mod = obj.modifiers.get("BuildingLayout")
if mod:
    obj.modifiers.remove(mod)

mod = obj.modifiers.new("BuildingLayout", "NODES")
mod.node_group = ng
mod.show_viewport = True
mod.show_render = True

# 초기값 설정
print("\n📊 Initial Demo State:")
print("   - Min Spacing: 2.5m (very tight)")
print("   - Buildings: 2x2m fixed size")
print("   - Fill area based on spacing")
print("\n🎯 LLM can adjust:")
print("   - Increase Min Spacing → 5.0")
print("   - Result: Less crowded, better ventilation!")

bpy.context.view_layer.update()
print("\n✅ SUCCESS! Demo ready for physics-based adjustment!")