import os

# 必须在导入任何 Mujoco 相关模块前设置
os.environ['MUJOCO_GL'] = 'glfw'
# os.environ["MUJOCO_GL"] = "egl"

import sys

sys.path.append(os.path.abspath("./gym_dcmm/"))

import argparse

print(os.getcwd())

from decorators import *

from gym_dcmm.envs import DcmmVecEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for DcmmVecEnv")
    parser.add_argument(
        "--viewer", action="store_true", help="open the mujoco.viewer or not"
    )
    parser.add_argument(
        "--imshow_cam", action="store_true", help="imshow the camera image or not"
    )
    args = parser.parse_args()
    print("args: ", args)

    env = DcmmVecEnv(
        task="Catching",
        object_name="object",
        render_per_step=False,
        print_reward=False,
        print_info=False,
        print_contacts=False,
        print_ctrl=True,
        print_obs=False,
        camera_name=["top"],
        render_mode="rgb_array",
        imshow_cam=args.imshow_cam,
        viewer=args.viewer,
        object_eval=False,
        env_time=2.5,
        steps_per_policy=20,
    )

    env.run_test()
