# SMPLest-X

This repository is a fork of the original SMPLest-X project with additional functionality to convert model outputs to LHM (Lightweight Human Model) compatible SMPL format.

**Original Repository: https://github.com/SMPLCap/SMPLest-X**

## Overview

SMPLest-X is an SMPL-X based 3D human pose and mesh estimation model. This fork extends the original model with the capability to convert its outputs to a format compatible with LHM (Lightweight Human Model) by adjusting the SMPL format.

This project provides the following features:

- **3D Human Pose Estimation**: Estimate 3D human pose from images
- **SMPL-X Mesh Generation**: Generate detailed human mesh
- **LHM Compatibility**: Convert model outputs to SMPL format for use with LHM
- **JSON Output**: Save estimation results in JSON format

## Key Features

### SMPL-X to LHM Compatible Format Conversion

The model performs the following format adjustments to make outputs compatible with LHM (Lightweight Human Model):

- **Body Pose**: Convert from 63 parameters to 21 joints (3 parameters each)
- **Hand Poses**: Convert from 45 parameters to 15 joints (3 parameters each)
- **Eye Poses**: Set to default values (adjustable if needed)
- **Camera Parameters**: Include focal length, principal point, and other camera parameters

### Output Format

```json
{
  "betas": [shape_parameters],
  "root_pose": [root_rotation],
  "body_pose": [[joint1_x, joint1_y, joint1_z], ...],
  "jaw_pose": [jaw_rotation],
  "leye_pose": [left_eye_rotation],
  "reye_pose": [right_eye_rotation],
  "lhand_pose": [[joint1_x, joint1_y, joint1_z], ...],
  "rhand_pose": [[joint1_x, joint1_y, joint1_z], ...],
  "trans": [translation],
  "focal": [focal_length],
  "princpt": [principal_point],
  "img_size_wh": [width, height],
  "pad_ratio": 0.2
}
```

## Installation and Environment Setup

For installation and environment setup, refer to the original repository:
**Original: https://github.com/SMPLCap/SMPLest-X**

### Requirements

- Python 3.8+
- PyTorch
- CUDA (for GPU usage)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
SMPLest-X/
├── main/                    # Main execution files
│   ├── inference.py        # Inference script (with LHM compatible output)
│   └── ...
├── models/                  # Model definitions
├── utils/                   # Utility functions
├── human_models/           # SMPL-X model definitions
├── pretrained_models/      # Pre-trained models
├── demo/                   # Demo files
│   ├── input_frames/       # Input images
│   ├── output_frames/      # Output images
│   └── output_json/        # JSON output (LHM compatible)
└── requirements.txt        # Dependencies list
```

## Usage

### Run Inference

```bash
python main/inference.py \
    --file_name test \
    --ckpt_name model_dump \
    --start 1 \
    --end 10 \
    --save_json
```

### Parameters

- `--file_name`: Input image folder name
- `--ckpt_name`: Checkpoint model name
- `--start`: Starting frame number
- `--end`: Ending frame number
- `--save_json`: Save LHM compatible JSON output (newly added feature)

### Input Preparation

1. Place image files in `demo/input_frames/{file_name}/` folder with format `000001.jpg`, `000002.jpg`, etc.
2. Place model checkpoint in `pretrained_models/{ckpt_name}/` folder

### Output

- **Image Output**: `demo/output_frames/{file_name}/`
- **JSON Output**: `demo/output_json/{file_name}/` (LHM compatible format)

## LHM Compatibility

This project performs the following conversions to make SMPLest-X outputs compatible with LHM:

1. **Pose Parameter Conversion**: Convert SMPL-X's 72 pose parameters to LHM compatible format
2. **Joint Structure Adjustment**: Reorganize hand and body joints to match LHM format
3. **Camera Parameters**: Include focal length, principal point, and other camera information

## License

Follows the license of the original SMPLest-X project. 