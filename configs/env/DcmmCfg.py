import os
from pathlib import Path

import numpy as np

## Define the model path
path = os.path.realpath(__file__)
root = str(Path(path).parent)
ASSET_PATH = os.path.join(root, "../../assets")
# print("ASSET_PATH: ", ASSET_PATH)
# Use Leap Hand
XML_DCMM_LEAP_OBJECT_PATH = "urdf/scout_piper.xml"
XML_DCMM_LEAP_UNSEEN_OBJECT_PATH = "urdf/scout_piper.xml"
XML_ARM_PATH = "urdf/xarm_piper.xml"
## Weight Saved Path
WEIGHT_PATH = os.path.join(ASSET_PATH, "weights")

## The distance threshold to change the stage from 'tracking' to 'grasping'
distance_thresh = 0.25

## Define the initial joint positions of the arm and the hand
arm_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

hand_joints = np.array([0.01, -0.01])

## Define the reward weights
reward_weights = {
    "r_base_pos": 0.0,
    "r_ee_pos": 10.0,
    "r_precision": 10.0,
    "r_orient": 1.0,
    "r_touch": {"Tracking": 5, "Catching": 0.1},
    "r_constraint": 1.0,
    "r_stability": 20.0,
    "r_ctrl": {
        "base": 0.2,
        "arm": 1.0,
        "hand": 0.2,
    },
    "r_collision": -10.0,
}

## Define the camera params for the MujocoRenderer.
cam_config = {
    "name": "top",
    "width": 640,
    "height": 480,
}

## Define the params of the Double Ackerman model.
RangerMiniV2Params = {
    "wheel_radius": 0.1,  # in meter //ranger-mini 0.1
    "steer_track": 0.364,  # in meter (left & right wheel distance) //ranger-mini 0.364
    "wheel_base": 0.494,  # in meter (front & rear wheel distance) //ranger-mini 0.494
    "max_linear_speed": 1.5,  # in m/s
    "max_angular_speed": 4.8,  # in rad/s
    "max_speed_cmd": 10.0,  # in rad/s
    "max_steer_angle_ackermann": 0.6981,  # 40 degree
    "max_steer_angle_parallel": 1.570,  # 180 degree
    "max_round_angle": 0.935671,
    "min_turn_radius": 0.47644,
    "Half of the wheelbase": 0.29153,
    "scout raduis": 0.16459,
}

## Define IK
ik_config = {
    "solver_type": "QP",
    "ps": 0.001,
    "λΣ": 12.5,
    "ilimit": 100,
    "ee_tol": 1e-4,
}

# Define the Randomization Params
## Wheel Drive
k_drive = np.array([0.75, 1.25])
## Wheel Steer
k_steer = np.array([0.75, 1.25])
## Arm Joints
k_arm = np.array([0.75, 1.25])
## Hand Joints
k_hand = np.array([0.75, 1.25])
## Object Shape and Size
object_shape = ["box", "cylinder", "sphere", "ellipsoid", "capsule"]
object_mesh = ["bottle_mesh", "bread_mesh", "bowl_mesh", "cup_mesh", "winnercup_mesh"]
object_size = {
    "sphere": np.array([[0.035, 0.045]]),
    "capsule": np.array([[0.025, 0.035], [0.025, 0.04]]),
    "cylinder": np.array([[0.025, 0.035], [0.025, 0.035]]),
    "box": np.array([[0.025, 0.035], [0.025, 0.035], [0.025, 0.035]]),
    "ellipsoid": np.array([[0.03, 0.03], [0.045, 0.045], [0.045, 0.045]]),
}
object_mass = np.array([0.035, 0.075])
object_damping = np.array([5e-3, 2e-2])
object_static = np.array([0.5, 0.75])
## Observation Noise
k_obs_base = 0
k_obs_arm = 0
k_obs_object = 0
k_obs_hand = 0
## Actions Noise
k_act = 0
## Action Delay
act_delay = {
    "base": [
        1,
    ],
    "arm": [
        1,
    ],
    "hand": [
        1,
    ],
}

## Define PID params for wheel drive and steering.
# driving
Kp_drive = 10
Ki_drive = 1e-3
Kd_drive = 1e-1
llim_drive = -20
ulim_drive = 20
# steering
Kp_steer = 50.0
Ki_steer = 2.5
Kd_steer = 7.5
llim_steer = -50
ulim_steer = 50

## Define PID params for the arm and hand.
Kp_arm = np.array([3.0, 4.0, 4.0, 5.0, 2.0, 2.0])
Ki_arm = np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3])
Kd_arm = np.array([0.0, 0.0, 0, 0, 0, 0])
llim_arm = np.array([-300.0, -300.0, -300.0, -50.0, -50.0, -20.0])
ulim_arm = np.array([300.0, 300.0, 300.0, 50.0, 50.0, 20.0])

Kp_hand = 100
Ki_hand = 1e-2
Kd_hand = 0.0
llim_hand = -10
ulim_hand = 10
hand_mask = np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1])
