# Hand Pose Estimation Models

This repository contains three ONNX hand-pose estimation models. All models predict one `hand` class with 21 keypoints.

| Model | Input resolution | Files |
| --- | --- | --- |
| YOLO11n Hand Pose | Refer to the ONNX model metadata | `yolo11n-hand-pose/` |
| YOLOv8n Hand Pose | 512 × 288 | `yolov8n-hand-pose_512x288_split/` |
| YOLOv8s Hand Pose, HaGRID fine-tuned | 640 × 640 | `yolov8s-hand-pose_512x288_split/` |

The YOLOv8n and YOLOv8s directories include `pose_split_kps.py`, which extracts the detector and keypoint head outputs for VectorBlox-compatible processing. The YOLOv8s directory contains the HaGRID fine-tuned PyTorch weights (`yolov8s-hand-pose.pt`), matching 640 × 640 ONNX exports, and `hand-keypoints-identity-flip-hagrid.yaml`. Intermediate checkpoints and training logs are excluded.

Dataset configuration is based on the [Ultralytics Hand Keypoints dataset](https://docs.ultralytics.com/datasets/pose/hand-keypoints/).
