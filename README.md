# ARRC-Net

**The code will be made publicly available upon acceptance of the paper.**

This repository accompanies the paper:

**Affinity-Guided Competitive Retrieval for Cross-Domain Few-Shot Hyperspectral Image Classification**

ARRC-Net integrates spectral pattern mapping, classification-calibrated spatial affinity refinement, and competitive local reference retrieval for cross-domain few-shot hyperspectral image classification.

![79302209240](C:\Users\29876\AppData\Local\Temp\1793022092406.png)

## Main Modules

- **SPM**: Spectral mapping based on local statistics and directional moments.
- **CP-SAR**: Training-time affinity calibration, feature refinement, and selective prediction distillation.
- **CLRR**: Uses the final V2 implementation to enhance classification scores through local reference retrieval.

## Environment and Data

Python, CUDA-enabled PyTorch, and the following packages are required:

```text
numpy==1.22.3
scipy==1.8.0
scikit-learn==1.0.2
matplotlib==3.5.1
tensorboard==2.8.0
protobuf==3.20.1
```

Prepare the datasets according to the paths specified in the configuration files. The default directory structure is:

```text
datasets/
├── Chikusei_imdb_128_7_7.pickle
├── IP/
│   ├── indian_pines_corrected.mat
│   └── indian_pines_gt.mat
└── Houston/
    ├── data.mat
    ├── mask_train.mat
    └── mask_test.mat
```

## Usage

Run the following commands from the project root:

```bash
# Indian Pines
python train.py --config config/Indian_pines_full.py

# Houston Full-GT Random
python train-HT.py --config config/houston_random_full.py
```

Periodic evaluation is performed during training. Logs, classification maps, and the `best.pt` and `final.pt` checkpoints are saved. Output locations are controlled by `log_dir` and the classification-map filenames specified in the configuration files. TensorBoard logs are saved to `runs/` by default.