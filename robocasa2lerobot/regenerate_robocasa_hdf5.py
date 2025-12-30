import os
import sys 
sys.path.append("/home/binhng/Workspace/robocasa/robocasa")

import json 
import h5py
import numpy as np
from tqdm import tqdm

import robosuite

# from robocasa.utils.env_utils import create_env
# from robocasa.utils.robomimic.robomimic_env_utils import create_env_from_metadata
from robocasa.scripts.playback_dataset import reset_to  

ROBOCASA_DUMMY_ACTION = [0.0] * 6 + [-1.0] + [0.0] * 4 + [-1.0]

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
    env_meta['env_kwargs']['camera_segmentations'] = 'instance' # element' #'instance'
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
    
def process_1_demo(env, f, demo_id, grp):
    demo = f["data"][demo_id]
    reset_each_demo(env, demo) 
    
    for _ in range(10):
        obs, reward, done, info = env.step(ROBOCASA_DUMMY_ACTION)
    
    obs_keys = list(obs.keys())
    
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
        # append all keys
        for key in obs_keys:
            if "eye_in_hand" in key or "agentview" in key:
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
        ep_data.attrs["model_file"] = demo.attrs["model_file"]
        ep_data.attrs["ep_meta"] = demo.attrs["ep_meta"]
        
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
    f = h5py.File(input_path, "r")
    env, env_meta = creat_env_from_hdf5(f)
    
    out_f = h5py.File(output_path, "w")
    out_f.attrs["env_args"] = json.dumps(env_meta)
    
    grp = out_f.create_group("data")
    
    all_demo_ids = list(f["data"].keys())
    if debug:
        all_demo_ids = all_demo_ids[:min(2, len(all_demo_ids))]
    for demo_id in tqdm(all_demo_ids):
        print(f"Processing {demo_id} ...")
        process_1_demo(env, f, demo_id, grp)
    
    f.close()
    if len(out_f["data"].keys()) == 0:
        print("No demos were processed successfully. Deleting output file.")
        out_f.close()
        os.remove(output_path)
    else:
        print(f"Processed data saved {len(out_f['data'].keys())} demos to {output_path}")
        out_f.close()
        

if __name__ == "__main__":
    origin_dir = '/home/binhng/Workspace/robocasa/robocasa/datasets/origin/robocasa-100demos-5chosen-tasks/'
    regenerate_dir = '/home/binhng/Workspace/robocasa/robocasa/datasets/regenerate/robocasa-100demos-5chosen-tasks/'
    os.makedirs(regenerate_dir, exist_ok=True)
    
    task_list = [
        'PnPCabToCounter',
        # 'PnPCounterToCab',
        # 'CoffeeSetupMug',
        # 'TurnOffStove',
        # 'TurnOnMicrowave'
        # ... add other tasks as needed
    ]
    
    for task in task_list:
        input_path = os.path.join(origin_dir, f'{task}.hdf5')
        output_path = os.path.join(regenerate_dir, f'{task}.hdf5')
        
        print(f"Regenerating dataset for task {task} ...")
        regenerate_hdf5_dataset(input_path, output_path, debug=False)