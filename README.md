# Soybean V2 Prediction Package

SELECTED MODELS
Leaf: ConvNeXt-Tiny, epoch 14, input size 384 x 384.
Drone: MaxViT-Tiny, epoch 6, input size 224 x 224.
Classes: healthy, mosaic, rust, pest_attack.

The models were selected using validation macro-F1.
Each image is evaluated by its selected branch.
Paired leaf-drone fusion is not implemented.
This package performs inference, not training.

RUN IN KAGGLE
Extract the package into /kaggle/working.
Run these lines in a notebook code cell:

import sys
sys.path.insert(0, "/kaggle/working/soybean_v2_prediction_package")
from notebook_interface import launch
launch()

DIRECT PREDICTION
from inference import predict_crop
Pass an existing image path and mode='leaf' or mode='drone'.

ENVIRONMENT
Compatible existing Kaggle packages can be used without reinstalling.
environment_versions.json records the original environment.
requirements.txt records package versions without local CUDA suffixes.
On another machine, install a PyTorch build suitable for its hardware.
CPU inference is supported but may be slower.

INTERPRETATION
Outputs are four-class soybean predictions.
Softmax scores are uncalibrated and do not guarantee correctness.
Unsupported crops and unrelated images are not automatically rejected.
Drone test performance is exploratory, not proof of new-field accuracy.
Early-stage detection has not been validated.

CONTENTS
Selected model weights, inference code, and notebook interface.
Configuration, environment versions, and dependency list.
Prediction history is included if it was available during packaging.
History labels are user supplied, not independently verified.
Uploaded images are not included; historical image paths may be stale.
Full evaluation reports and other models remain in separate backups.
