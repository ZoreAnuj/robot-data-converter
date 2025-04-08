import os
import json
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# Configuration
SOURCE_ROOT = "path/to/source"
TARGET_ROOT = "path/to/first_data"
FPS = 30
CHUNK_SIZE = 1000
IMG_SIZE = (320, 240)  # Width x Height

# Utilities
def parse_transform(transform_str):
    """Extract translation and rotation from a transform string."""
    lines = transform_str.strip().split('\n')
    translation = [float(x.split('=')[1]) for x in lines[0].split()[1:]]
    rotation = [float(x.split('=')[1]) for x in lines[1].split()[1:]]
    return translation + rotation

def read_images_sorted(folder, suffix):
    files = sorted(glob(os.path.join(folder, f"*_{suffix}.jpg")), key=lambda x: int(os.path.basename(x).split("_")[0]))
    return [cv2.resize(cv2.imread(f), IMG_SIZE) for f in files]

def build_action_vector(act):
    """Map original actuator data to 12D action/state vector."""
    return [
        act['left_wheel_vel'], act['right_wheel_vel'], act['lift'],
        act['arm'], act['wrist_yaw'], act['wrist_pitch'],
        act['wrist_roll'], act['gripper'], act['head_pan'],
        act['head_tilt'], 0.0, 0.0  # Placeholder for unused secondary motors
    ]

# Conversion Logic
def convert_episode(source_episode_path, episode_index, output_data_dir, output_video_dir):
    # Load actuator
    with open(os.path.join(source_episode_path, "Actuator.json")) as f:
        actuator_data = json.load(f)["ActuatorData"]

    # Load transforms
    with open(os.path.join(source_episode_path, "ArmCam/ArmCamPos.json")) as f:
        arm_transforms = {t["index"]: t["transform"] for t in json.load(f)["transforms"]}
    with open(os.path.join(source_episode_path, "HeadCam/HeadCamPos.json")) as f:
        head_transforms = {t["index"]: t["transform"] for t in json.load(f)["transforms"]}

    # Read image sequences
    arm_images = read_images_sorted(os.path.join(source_episode_path, "ArmCam"), "RGB")
    head_images = read_images_sorted(os.path.join(source_episode_path, "HeadCam"), "RGB")

    records = []
    frame_count = len(actuator_data)
    for frame_index, act in enumerate(actuator_data):
        index = episode_index * 100000 + frame_index
        timestamp = frame_index / FPS
        action = build_action_vector(act)
        observation_state = action.copy()  # Mirror action as state

        records.append({
            "index": index,
            "timestamp": timestamp,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "task_index": 0,
            "action": np.array(action, dtype=np.float32),
            "observation.state": np.array(observation_state, dtype=np.float32)
        })

    # Save Parquet
    df = pd.DataFrame(records)
    df["action"] = df["action"].apply(lambda x: x.tolist())
    df["observation.state"] = df["observation.state"].apply(lambda x: x.tolist())
    parquet_path = os.path.join(output_data_dir, f"episode_{episode_index:06d}.parquet")
    df.to_parquet(parquet_path, index=False)

    # Save videos using OpenCV instead of MoviePy
    head_video_path = os.path.join(output_video_dir, "observation.images.main", f"episode_{episode_index:06d}.mp4")
    arm_video_path = os.path.join(output_video_dir, "observation.images.secondary_0", f"episode_{episode_index:06d}.mp4")
    os.makedirs(os.path.dirname(head_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(arm_video_path), exist_ok=True)

    # OpenCV video writer for head camera
    h, w = head_images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for h264 codec
    head_writer = cv2.VideoWriter(head_video_path, fourcc, FPS, (w, h))
    for frame in head_images:
        head_writer.write(frame)
    head_writer.release()
    
    # OpenCV video writer for arm camera
    arm_writer = cv2.VideoWriter(arm_video_path, fourcc, FPS, (w, h))
    for frame in arm_images:
        arm_writer.write(frame)
    arm_writer.release()

    return frame_count

# Batch Conversion
def convert_all(source_root, target_root):
    output_data_dir = os.path.join(target_root, "data/chunk-000")
    output_video_dir = os.path.join(target_root, "videos/chunk-000")
    meta_dir = os.path.join(target_root, "meta")
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    all_folders = sorted(glob(os.path.join(source_root, "[0-9]*")))
    total_frames = 0
    episode_meta = []

    for episode_index, folder in enumerate(tqdm(all_folders, desc="Converting episodes")):
        frame_count = convert_episode(folder, episode_index, output_data_dir, output_video_dir)
        episode_meta.append({
            "episode_index": episode_index,
            "length": frame_count,
            "chunk_index": 0,
            "task_id": 0
        })
        total_frames += frame_count

    # Save meta files
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump({
            "robot_type": "so-100, so-100",
            "codebase_version": "v2.0",
            "fps": FPS,
            "total_episodes": len(all_folders),
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_chunks": 1,
            "chunks_size": CHUNK_SIZE,
            "splits": {"train": f"0:{len(all_folders)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "action": {"dtype": "float32", "shape": [12], "names": [f"motor_{i+1}" for i in range(12)]},
                "observation.state": {"dtype": "float32", "shape": [12]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "observation.images.main": {
                    "dtype": "video", "shape": [240, 320, 3],
                    "info": {
                        "video_fps": FPS, "video_codec": "mp4v",
                        "video_pix_fmt": "yuv420p", "video_is_depth_map": False,
                        "has_audio": False
                    }
                },
                "observation.images.secondary_0": {
                    "dtype": "video", "shape": [240, 320, 3],
                    "info": {
                        "video_fps": FPS, "video_codec": "mp4v",
                        "video_pix_fmt": "yuv420p", "video_is_depth_map": False,
                        "has_audio": False
                    }
                }
            }
        }, f, indent=2)

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep in episode_meta:
            f.write(json.dumps(ep) + "\n")

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({
            "task_id": 0,
            "language_instruction": "Pick up the object",
            "goal_state": {},
            "scene_config": {}
        }) + "\n")

# Run the converter
SOURCE_ROOT="/home/zero/Lucky_robots/datacollection/test_data"
TARGET_ROOT="/home/zero/Lucky_robots/datacollection/converted_data"
convert_all(SOURCE_ROOT, TARGET_ROOT)  # Uncomment to run