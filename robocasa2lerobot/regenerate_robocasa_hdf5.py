import os
import sys 
sys.path.append("/home/lmaotan/workspace/robocasa_regenerate/robocasa")

import cv2
import json 
import h5py
import numpy as np
from dataclasses import asdict, dataclass, field
from tqdm import tqdm

import robosuite
import torch

from scipy.spatial.transform import Rotation as R
# from robocasa.utils.env_utils import create_env
# from robocasa.utils.robomimic.robomimic_env_utils import create_env_from_metadata
from robocasa.scripts.playback_dataset import reset_to
from robocasa.utils.camera_utils import CAM_CONFIGS
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_camera_extrinsic_matrix_rel
from robosuite.utils.transform_utils import make_pose, quat2mat
from renderers.common_utils import (
    inverse_tf,
    depth_to_pcd,
    transform_pcd,
    visualize_to_select_views
)
from renderers.rvt_renderer import RVTRenderer


ROBOCASA_DUMMY_ACTION = [0.0] * 6 + [-1.0] + [0.0] * 4 + [-1.0]
CAMERA_NAMES = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand"
]
# VIRTUAL_CAMERA_NAMES = [
#     "virtual_top",
#     "virtual_front",
#     "virtual_back",
#     "virtual_left",
#     "virtual_right"
# ]
H, W, C = 256, 256, 3


@dataclass
class RVTRendererConfig:
    camera_names: list[str] = field(
        default_factory=lambda: CAMERA_NAMES
    )
    renderer_device: str = "cuda:0"
    img_size: int = 256
    scene_bounds: list[float] = field(
        default_factory=lambda: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    )
    move_pc_in_bound:   bool = False
    place_pc_in_cube:   bool = False
    place_with_mean:    bool = False
    pers:               bool = True

    use_point_renderer: bool = True
    rend_three_views:   bool = False
    add_depth:          bool = True
    add_corr:           bool = True
    norm_corr:          bool = True
    add_pixel_loc:      bool = True


def get_camera_info(sim, camera_name, camera_height, camera_width):
    camera_intrinsics = get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width)
    camera_extrinsics = get_camera_extrinsic_matrix(sim, camera_name)

    return camera_intrinsics, camera_extrinsics 


def get_perturbed_transformation(sim, camera_name):
    rand_cam_2_base = get_camera_extrinsic_matrix_rel(sim, camera_name)

    org_cam_2_base_trans = np.asarray(CAM_CONFIGS["DEFAULT"][camera_name]["pos"]).copy()
    # org_cam_2_base_rot_mat = quat2mat(CAM_CONFIGS["DEFAULT"][camera_name]["quat"])
    org_cam_2_base_quat = np.asarray(CAM_CONFIGS["DEFAULT"][camera_name]["quat"]).copy()
    r = R.from_quat(org_cam_2_base_quat)
    org_cam_2_base_rot_mat = r.as_matrix()

    org_cam_2_base = make_pose(
        org_cam_2_base_trans,
        org_cam_2_base_rot_mat
    )

    rand_cam_2_org_cam = inverse_tf(org_cam_2_base, custom_op=True) @ rand_cam_2_base

    return rand_cam_2_org_cam


# obs_keys = [
#     "robot0_joint_pos",
#     "robot0_joint_pos_cos",
#     "robot0_joint_pos_sin",
#     "robot0_joint_vel",
#     "robot0_joint_acc",
#     "robot0_eef_pos",
#     "robot0_eef_quat",
#     "robot0_eef_quat_site",
#     "robot0_gripper_qpos",
#     "robot0_gripper_qvel",
#     "robot0_base_pos",
#     "robot0_base_quat",
#     "robot0_base_to_eef_pos",
#     "robot0_base_to_eef_quat",
#     "robot0_base_to_eef_quat_site",
#     "robot0_agentview_left_image",
#     "robot0_agentview_left_depth",
#     "robot0_agentview_left_segmentation_instance",
#     "robot0_agentview_right_image",
#     "robot0_agentview_right_depth",
#     "robot0_agentview_right_segmentation_instance",
#     "robot0_eye_in_hand_image",
#     "robot0_eye_in_hand_depth",
#     "robot0_eye_in_hand_segmentation_instance",
#     "obj_pos",
#     "obj_quat",
#     "obj_to_robot0_eef_pos",
#     "obj_to_robot0_eef_quat",
#     "distr_counter_pos",
#     "distr_counter_quat",
#     "distr_counter_to_robot0_eef_pos",
#     "distr_counter_to_robot0_eef_quat",
#     "distr_cab_pos",
#     "distr_cab_quat",
#     "distr_cab_to_robot0_eef_pos",
#     "distr_cab_to_robot0_eef_quat",
#     "robot0_proprio-state",
#     "object-state",
# ]

# act_keys = ['abs_pos', 'abs_rot_6d',
#             'abs_rot_axis_angle', 'gripper',
#             'rel_pos', 'rel_rot_6d', 'rel_rot_axis_angle']


def creat_env_from_hdf5(f):
    # dataset_path = '/home/binhng/Workspace/robocasa/robocasa/datasets/test/PnPCabToCounter.hdf5'
    # f = h5py.File(dataset_path, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    env_meta['env_kwargs']['camera_depths'] = True 
    env_meta['env_kwargs']['camera_heights'] = 256
    env_meta['env_kwargs']['camera_widths'] = 256
    env_meta['env_kwargs']['camera_segmentations'] = 'element' # element' #'instance'
    # f.close()

    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True #write_video
    env_kwargs["use_camera_obs"] = True
    env_kwargs["ignore_done"] = False

    env = robosuite.make(**env_kwargs)
    # env = create_env_from_metadata(env_meta=env_kwargs)
    # print("env ignore done:", env.ignore_done)
    # for _ in range(10):
    #     obs, reward, done, info = env.step(ROBOCASA_DUMMY_ACTION)
    
    return env, env_meta


def reset_each_demo(env, demo):
    # demo = f["data"]["demo_<idx>"]
    model_xml = demo.attrs["model_file"]
    init_state = demo['states'][()][0]
    ep_meta = demo.attrs["ep_meta"]
    
    state = {
        "states": init_state,
        "model": model_xml,
        "ep_meta": ep_meta
    }
    reset_to(env, state)
    

def process_1_demo(env, f, demo_id, grp, rvt_renderer, input_path):
    demo = f["data"][demo_id]
    reset_each_demo(env, demo)

    ep_meta = env.get_ep_meta()
    model_file = env.model.get_xml()
    task_name = os.path.basename(input_path)[:-5]
    demo_id_str = demo_id.replace("_", "")
    save_folder = f"/home/lmaotan/data/robocasa_obs_views/{task_name}/{demo_id_str}"
    os.makedirs(save_folder, exist_ok=True)
    
    for _ in range(10):
        obs, reward, done, info = env.step(ROBOCASA_DUMMY_ACTION)
    
    obs_keys = list(obs.keys())
    obs_keys += ["robot0_agentview_left_intrinsics", "robot0_agentview_right_intrinsics", "robot0_eye_in_hand_intrinsics"]
    obs_keys += ["robot0_agentview_left_extrinsics", "robot0_agentview_right_extrinsics", "robot0_eye_in_hand_extrinsics"]
    obs_keys += ["robot0_agentview_left_extrinsicsR", "robot0_agentview_right_extrinsicsR", "robot0_eye_in_hand_extrinsicsR"]
    obs_keys += ["robot0_agentview_left_depthW", "robot0_agentview_right_depthW", "robot0_eye_in_hand_depthW"]
    
    obs_dict = {key: [] for key in obs_keys}
    # action_dict = {key: [] for key in act_keys}
    actions = []
    actions_abs = []
    rewards = []
    dones = []
    states = [] # env state, not robot. The state for robot is included in obs
    
    # for key in obs_keys:
    #     obs_dict[key] = obs[key]
    orig_actions = demo['actions'][()]
    orig_actions_abs = demo['actions_abs'][()]
    # orig_action_dict = demo['action_dict']
    
    for i, action in enumerate(orig_actions):
    # for i, action in enumerate(orig_actions_abs):
        extent = env.sim.model.stat.extent
        far = env.sim.model.vis.map.zfar * extent
        near = env.sim.model.vis.map.znear * extent
        left_depth = obs["robot0_agentview_left_depth"].copy()
        right_depth = obs["robot0_agentview_right_depth"].copy()
        wrist_depth = obs["robot0_eye_in_hand_depth"].copy()
        left_depth = (near / (1.0 - left_depth * (1.0 - near / far)))[::-1]
        right_depth = (near / (1.0 - right_depth * (1.0 - near / far)))[::-1]
        wrist_depth = (near / (1.0 - wrist_depth * (1.0 - near / far)))[::-1]

        obs["robot0_agentview_left_depthW"] = left_depth
        obs["robot0_agentview_right_depthW"] = right_depth
        obs["robot0_eye_in_hand_depthW"] = wrist_depth

        left_intrinsics, left_extrinsics = get_camera_info(env.sim, "robot0_agentview_left", H, W)
        right_intrinsics, right_extrinsics = get_camera_info(env.sim, "robot0_agentview_right", H, W)
        wrist_intrinsics, wrist_extrinsics = get_camera_info(env.sim, "robot0_eye_in_hand", H, W)

        obs["robot0_agentview_left_intrinsics"] = left_intrinsics
        obs["robot0_agentview_right_intrinsics"] = right_intrinsics
        obs["robot0_eye_in_hand_intrinsics"] = wrist_intrinsics
        obs["robot0_agentview_left_extrinsics"] = left_extrinsics        
        obs["robot0_agentview_right_extrinsics"] = right_extrinsics
        obs["robot0_eye_in_hand_extrinsics"] = wrist_extrinsics

        left_extrinsics_rel = get_camera_extrinsic_matrix_rel(env.sim, "robot0_agentview_left")
        right_extrinsics_rel = get_camera_extrinsic_matrix_rel(env.sim, "robot0_agentview_right")
        wrist_extrinsics_rel = get_camera_extrinsic_matrix_rel(env.sim, "robot0_eye_in_hand")

        obs["robot0_agentview_left_extrinsicsR"] = left_extrinsics_rel
        obs["robot0_agentview_right_extrinsicsR"] = right_extrinsics_rel
        obs["robot0_eye_in_hand_extrinsicsR"] = wrist_extrinsics_rel

        left_pcd = depth_to_pcd(left_depth, left_intrinsics, depth_scale=1.0)
        right_pcd = depth_to_pcd(right_depth, right_intrinsics, depth_scale=1.0)
        wrist_pcd = depth_to_pcd(wrist_depth, wrist_intrinsics, depth_scale=1.0)

        left_pcd_aligned = transform_pcd(left_pcd, left_extrinsics)
        right_pcd_aligned = transform_pcd(right_pcd, right_extrinsics)
        wrist_pcd_aligned = transform_pcd(wrist_pcd, wrist_extrinsics)
        
        left_pcd_aligned_pt = torch.from_numpy(left_pcd_aligned).cuda().float()
        right_pcd_aligned_pt = torch.from_numpy(right_pcd_aligned).cuda().float()
        wrist_pcd_aligned_pt = torch.from_numpy(wrist_pcd_aligned).cuda().float()

        left_pcd_aligned_pt = left_pcd_aligned_pt.permute(1, 0).reshape(C, H, W).unsqueeze(0)
        right_pcd_aligned_pt = right_pcd_aligned_pt.permute(1, 0).reshape(C, H, W).unsqueeze(0)
        wrist_pcd_aligned_pt = wrist_pcd_aligned_pt.permute(1, 0).reshape(C, H, W).unsqueeze(0)

        left_image = obs["robot0_agentview_left_image"].copy()
        right_image = obs["robot0_agentview_right_image"].copy()
        wrist_image = obs["robot0_eye_in_hand_image"].copy()

        left_image_pt = torch.from_numpy(left_image).cuda().flip([0]).permute(2, 0, 1).unsqueeze(0)
        right_image_pt = torch.from_numpy(right_image).cuda().flip([0]).permute(2, 0, 1).unsqueeze(0)
        wrist_image_pt = torch.from_numpy(wrist_image).cuda().flip([0]).permute(2, 0, 1).unsqueeze(0)

        batch = {
            "robot0_agentview_left_pcd": left_pcd_aligned_pt,
            "robot0_agentview_right_pcd": right_pcd_aligned_pt,
            "robot0_eye_in_hand_pcd": wrist_pcd_aligned_pt,

            "robot0_agentview_left_rgb": left_image_pt,
            "robot0_agentview_right_rgb": right_image_pt,
            "robot0_eye_in_hand_rgb": wrist_image_pt
        }

        # left_rand_cam_2_left_org_cam = get_perturbed_transformation(env.sim, "robot0_agentview_left")
        base_2_world = left_extrinsics @ inverse_tf(left_extrinsics_rel, custom_op=True)
        
        selected_views_path = f"/home/lmaotan/workspace/robocasa_regenerate/robocasa2lerobot/selected_views/{task_name}/demo1161.npz"
        extrinsics = None
        if os.path.isfile(selected_views_path):
            extrinsics = []
            with np.load(selected_views_path) as data:
                for extr in data["camera_extrinsics"]:
                    extrinsics.append(
                        torch.from_numpy(base_2_world.copy() @ extr).cuda()
                    )
            extrinsics = torch.stack(extrinsics)
            extrinsics = (extrinsics,)
        
        rendered_images = rvt_renderer.render(
            batch,
            CAMERA_NAMES,
            extrinsics=extrinsics
        )

        _save_folder = os.path.join(save_folder, "robot0_agentview_left")
        os.makedirs(_save_folder, exist_ok=True)
        cv2.imwrite(os.path.join(_save_folder, f"step{i}.png"), cv2.cvtColor(left_image[::-1], cv2.COLOR_RGB2BGR))

        _save_folder = os.path.join(save_folder, "robot0_agentview_right")
        os.makedirs(_save_folder, exist_ok=True)
        cv2.imwrite(os.path.join(_save_folder, f"step{i}.png"), cv2.cvtColor(right_image[::-1], cv2.COLOR_RGB2BGR))

        _save_folder = os.path.join(save_folder, "robot0_eye_in_hand")
        os.makedirs(_save_folder, exist_ok=True)
        cv2.imwrite(os.path.join(_save_folder, f"step{i}.png"), cv2.cvtColor(wrist_image[::-1], cv2.COLOR_RGB2BGR))

        for j in range(getattr(rvt_renderer.renderer, "num_img", 0)):
            rendered_rgb = rendered_images[0, j, 3:6]
            rendered_rgb = rendered_rgb.permute(1, 2, 0)
            rendered_rgb = (rendered_rgb * 255).cpu().numpy().astype(np.uint8)
            
            _save_folder = os.path.join(save_folder, f"virtual_view{j}")
            os.makedirs(_save_folder, exist_ok=True)
            cv2.imwrite(os.path.join(_save_folder, f"step{i}.png"), cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))

        # TODO
        # env.close()
        # world_2_base = left_extrinsics_rel @ inverse_tf(left_extrinsics, custom_op=True)
        # visualize_to_select_views(
        #     rgbs=[left_image[::-1], right_image[::-1], wrist_image[::-1]],
        #     pcds=[left_pcd_aligned, right_pcd_aligned, wrist_pcd_aligned],
        #     camera_height=H,
        #     camera_width=W,
        #     camera_intrinsics=left_intrinsics,
        #     camera_extrinsics=inverse_tf(left_extrinsics, custom_op=True),
        #     zfar=far,
        #     znear=near,
        #     transformation=world_2_base,
        #     save_path=selected_views_path
        # )

        # append all keys
        for key in obs_keys:
            if ("eye_in_hand" in key or "agentview" in key) and "depthW" not in key and "intrinsics" not in key and "extrinsics" not in key:
                obs_dict[key].append(obs[key][::-1, :, :]) 
            else:
                obs_dict[key].append(obs[key])
        
        # for key in act_keys:
        #     action_dict[key].append(orig_action_dict[key][i])
        
        actions.append(action)
        actions_abs.append(orig_actions_abs[i])
        
        rewards.append(reward)
        dones.append(done)
        
        current_state = env.sim.get_state().flatten()
        states.append(current_state)
        
        # step env
        obs, reward, done, info = env.step(action.tolist())
        
        done = done or env._check_success()
        # if done:
        # print(f" Step {i} done: {done}")
        # print(f" Step {i} info: {info}")
        # print(f" Step {i} is_success: {env._check_success()}" )
    

    if done:
        print(f"Demo {demo_id} done after {i} actions!")
        
        # save to new hdf5 file here
        ep_data = grp.create_group(demo_id)
        # set attribute for ep_data here ...
        ep_data.attrs["model_file"] = model_file
        ep_data.attrs["ep_meta"] = json.dumps(ep_meta, indent=4)
        
        # obs group
        obs_grp = ep_data.create_group("obs")
        for key in obs_keys:
            obs_grp.create_dataset(key, data=np.stack(obs_dict[key], axis=0))
        
        # action_dict group
        # action_dict_grp = ep_data.create_group("action_dict")
        # for key in act_keys:
        #     action_dict_grp.create_dataset(key, data=np.stack(action_dict[key], axis=0))
        
        # actions dataset
        ep_data.create_dataset("actions", data=np.stack(actions, axis=0))
        
        ep_data.create_dataset("actions_abs", data=np.stack(actions_abs, axis=0))
        
        ep_data.create_dataset("dones", data=np.stack(dones, axis=0))
        
        ep_data.create_dataset("rewards", data=np.stack(rewards, axis=0))
        
        # state dataset
        ep_data.create_dataset("states", data=np.stack(states, axis=0))
    
    elif not done:
        print(f"Demo {demo_id} not done after all actions executed!")
        

def regenerate_hdf5_dataset(input_path, output_path, debug=False):
    rvt_renderer_config = RVTRendererConfig()
    rvt_renderer = RVTRenderer(**asdict(rvt_renderer_config))

    f = h5py.File(input_path, "r")
    env, env_meta = creat_env_from_hdf5(f)
    
    out_f = h5py.File(output_path, "w")
    out_f.attrs["env_args"] = json.dumps(env_meta)
    
    grp = out_f.create_group("data")
    
    all_demo_ids = list(f["data"].keys())
    if debug:
        all_demo_ids = all_demo_ids[:min(2, len(all_demo_ids))][1:]
    for demo_id in tqdm(all_demo_ids):
        print(f"Processing {demo_id} ...")
        process_1_demo(env, f, demo_id, grp, rvt_renderer, input_path)
    
    f.close()
    if len(out_f["data"].keys()) == 0:
        print("No demos were processed successfully. Deleting output file.")
        out_f.close()
        os.remove(output_path)
    else:
        print(f"Processed data saved {len(out_f['data'].keys())} demos to {output_path}")
        out_f.close()
        

if __name__ == "__main__":
    origin_dir = '/home/lmaotan/data/binh'
    regenerate_dir = '/home/binhng/Workspace/robocasa/robocasa/datasets/regenerate/robocasa-100demos-5chosen-tasks/'
    # os.makedirs(regenerate_dir, exist_ok=True)
    
    task_list = [
        # 'PnPCabToCounter',
        # 'PnPCounterToCab',
        # 'CoffeeSetupMug',
        # 'TurnOffStove',
        'TurnOnMicrowave'
        # ... add other tasks as needed
    ]
    
    for task in task_list:
        input_path = os.path.join(origin_dir, f'{task}.hdf5')
        # output_path = os.path.join(regenerate_dir, f'{task}.hdf5')
        output_path = "/home/lmaotan/data/binh/temp.hdf5"
        
        print(f"Regenerating dataset for task {task} ...")
        regenerate_hdf5_dataset(input_path, output_path, debug=False)