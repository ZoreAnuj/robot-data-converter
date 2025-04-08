# Robot Data Converter

A Python script for converting robot data collection from source format to a standardized format with parquet files and MP4 videos.

## Features

- Converts actuator data to standardized parquet format
- Generates MP4 videos from image sequences
- Creates metadata files for dataset documentation
- Preserves camera transforms and sensor data

## Dependencies

- OpenCV (cv2)
- NumPy
- Pandas
- tqdm
- pyarrow (for parquet support)

## Usage

```bash
python transform.py
```

## Configuration

Edit the following variables in the script:
- SOURCE_ROOT: Path to source data
- TARGET_ROOT: Path to output directory
- FPS: Frames per second for video output
- CHUNK_SIZE: Number of episodes per chunk
- IMG_SIZE: Resolution of output videos

## Data Structure

### Source Structure

The source data is organized as follows:

```
[folder: 0]                          ← A numbered folder for one timestep or sample
├── Actuator.json                   ← Actuator command data
│   └── ActuatorData (Array)
│       └── Object {
│            index            : Integer      ← Frame number / timestep
│            left_wheel_vel   : Number       ← Left wheel velocity
│            right_wheel_vel  : Number       ← Right wheel velocity
│            lift             : Number       ← Lift control (e.g. up/down)
│            arm              : Number       ← Arm position
│            wrist_yaw        : Number       ← Wrist rotation around Y axis
│            wrist_pitch      : Number       ← Wrist rotation around X axis
│            wrist_roll       : Number       ← Wrist rotation around Z axis
│            gripper          : Number       ← Gripper open/close state
│            head_pan         : Number       ← Head horizontal rotation
│            head_tilt        : Number       ← Head vertical rotation
│         }
│
├── ArmCam/
│   ├── [n]_RGB.jpg                ← n RGB images from the arm-mounted camera
│   ├── [n]_Depth.jpg              ← Corresponding n depth images (same index as RGB)
│   └── ArmCamPos.json             ← Pose of arm camera for each frame
│
└── HeadCam/
    ├── [n]_RGB.jpg               ← n RGB images from the head-mounted camera
    ├── [n]_Depth.jpg             ← Corresponding n depth images (same index as RGB)
    └── HeadCamPos.json           ← Pose of head camera for each frame
```

### Target Structure

The converted data follows this standardized format:

```
first_data/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet         # Episodic data file (one episode = one file)
│       ├── episode_000001.parquet
│       └── ...                            # ⬅️ Up to N episodes
│
├── videos/
│   └── chunk-000/
│       ├── observation.images.main/
│       │   ├── episode_000000.mp4         # RGB video from main (head) camera
│       │   └── ...
│       └── observation.images.secondary_0/
│           ├── episode_000000.mp4         # RGB video from secondary (arm) camera
│           └── ...
│
└── meta/
    ├── info.json                          # Global config & feature schema
    ├── stats.json                         # Global feature stats per field
    ├── episodes.jsonl                     # Line-by-line episode metadata
    └── tasks.jsonl                        # Task definitions (1 line per task)
```

### Parquet Data Structure

Each parquet file contains the following columns:
- index: int64 - Global unique frame index
- timestamp: float32 - Seconds since start of episode
- episode_index: int64 - Episode ID
- frame_index: int64 - Frame number within episode
- task_index: int64 - Task ID
- action: float32[12] - 12D action vector (6 primary + 6 secondary motor commands)
- observation.state: float32[12] - Sensor state for all motors (same layout as action)

Videos are stored separately as MP4 files and referenced in the metadata.

## Notes

- The script converts raw JPG sequences to MP4 videos using OpenCV
- Actuator data is mapped to a standardized 12-dimension vector format
- Camera transforms are preserved in the conversion process
- The script handles both main (head) and secondary (arm) camera views

