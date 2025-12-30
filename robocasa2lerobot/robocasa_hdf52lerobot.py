"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

# from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# import tensorflow_datasets as tfds
import tyro
import h5py
import numpy as np
import json
import os
from tqdm import tqdm

os.environ['LEROBOT_HOME'] = "/home/binhng/Workspace/robocasa/robocasa/datasets/robocasa_lerobot/"

REPO_NAME = "binhng/robocasa_30_demos_debug_lerobot_v1_5_chosen_tasks"  # Name of the output dataset, also used for the Hugging Face Hub
# RAW_DATASET_NAME = "demo_gentex_im256_randcams.hdf5"  # For simplicity we will combine multiple Libero datasets into one training dataset
RAW_DATASET_NAME = [
    'PnPCabToCounter.hdf5',
    # 'PnPCounterToCab.hdf5',
    # 'CoffeeSetupMug.hdf5',
    # 'TurnOffStove.hdf5',
    # 'TurnOnMicrowave.hdf5'
]

OUT_DATA_DIR = "/home/binhng/Workspace/robocasa/robocasa/datasets/robocasa_lerobot/"
DATA_DIR = "/home/binhng/Workspace/robocasa/robocasa/datasets/regenerate/robocasa-30demos-5chosen-tasks/"

# def main(data_dir: str, *, push_to_hub: bool = False):
def main(data_dir: str = DATA_DIR, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    # output_path = LEROBOT_HOME / REPO_NAME
    output_path = os.path.join(OUT_DATA_DIR, REPO_NAME)
    # if output_path.exists():
    #     shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=20,
        features={
            "right_image": {
                "dtype": "videos",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "left_image": {
                "dtype": "videos",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "videos",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (12,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for dataset_name in RAW_DATASET_NAME:
        # raw_dataset = h5py.File(os.path.join(data_dir, RAW_DATASET_NAME), "r")
        # raw_dataset = h5py.File(os.path.join(data_dir, dataset_name), "r")
        raw_dataset = h5py.File(os.path.join(data_dir, dataset_name), "r")
        demos = raw_dataset["data"].keys()
        for demo in tqdm(demos):
            demo_length = len(raw_dataset["data"][demo]["actions"])
            demo_data = raw_dataset["data"][demo]

            left_images = demo_data["obs"]["robot0_agentview_left_image"][:]
            right_images = demo_data["obs"]["robot0_agentview_right_image"][:]
            wrist_images = demo_data["obs"]["robot0_eye_in_hand_image"][:]
            states = np.concatenate(
                (
                    demo_data["obs"]["robot0_base_to_eef_pos"][:],
                    demo_data["obs"]["robot0_base_to_eef_quat"][:],
                    demo_data["obs"]["robot0_gripper_qpos"][:],
                ),
                axis=1,
            )
            actions = demo_data["actions"][:]
            for i in range(demo_length):
                ep_meta = demo_data.attrs["ep_meta"]
                ep_meta = json.loads(ep_meta)
                lang = ep_meta["lang"]
                dataset.add_frame(
                    {
                        "right_image": right_images[i],
                        "left_image": left_images[i],
                        "wrist_image": wrist_images[i],
                        "state": states[i].astype(np.float32),
                        "actions": actions[i].astype(np.float32),
                        "task": lang,
                    }
                )
            # ep_meta = demo_data.attrs["ep_meta"]
            # ep_meta = json.loads(ep_meta)
            # lang = ep_meta["lang"]
            # dataset.save_episode(task=lang)
            dataset.save_episode()
    dataset.finalize() 

    # Consolidate the dataset, skip computing stats since we will do that later
    # dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    # tyro.cli(main)
    main()