import mujoco
from mujoco import MjModel, MjData
import os

# 模型路径
model_path = "assets/urdf/scout_piper.xml"
assert os.path.exists(model_path), f"模型文件不存在: {model_path}"

# 加载模型
with open(model_path, 'r') as f:
    model_xml = f.read()
model = MjModel.from_xml_string(model_xml)
data = MjData(model)

# 初始化一次状态，确保 qvel 有效
mujoco.mj_forward(model, data)

print("关节名称 | qpos 索引 | dof 索引 | qvel 索引 | 关节范围 [min, max] | 当前速度")
print("=" * 90)

for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    qpos_index = model.jnt_qposadr[i]
    dof_index = model.jnt_dofadr[i]
    qvel_index = dof_index  # 对于 hinge/slide，一个 joint 对应一个 dof
    joint_range = model.jnt_range[i]
    current_vel = data.qvel[qvel_index]
    print(f"{joint_name:<20} qpos[{qpos_index}]  dof[{dof_index}]  qvel[{qvel_index}]  "
          f"range: [{joint_range[0]:.3f}, {joint_range[1]:.3f}]  vel: {current_vel:.5f}")

print(f"\n总的 qpos 长度为: {model.nq}")
print(f"总的 qvel 长度为: {model.nv}")

# 获取 floor 几何体的 ID
floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
print(f"\n'floor' 几何体的 ID 是: {floor_id}")
