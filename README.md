# Hand Pose Estimation Models

This repository contains two ONNX hand-pose estimation models.  Both models predict one `hand` class with 21 keypoints.

| Model | Input resolution | Files |
| --- | --- | --- |
| YOLO11n Hand Pose | Refer to the ONNX model metadata | `yolo11n-hand-pose/` |
| YOLOv8n Hand Pose | 512 × 288 | `yolov8n-hand-pose_512x288_split/` |

The YOLOv8n directory also includes `pose_split_kps.py`, which extracts the detector and keypoint head outputs for VectorBlox-compatible processing.

Dataset configuration is based on the [Ultralytics Hand Keypoints dataset](https://docs.ultralytics.com/datasets/pose/hand-keypoints/).
