> # Mitigating Upsampling Artifacts in Vision Transformers via Spatial-Spectral Decoupled Fine-Tuning
>
> > **Note:** This repository is the official anonymous PyTorch implementation for the double-blind review process.
>
> ## 📌 Introduction
>
> This repository provides the code for mitigating upsampling artifacts in Vision Transformers (ViTs) via a novel Spatial-Spectral Decoupled Fine-Tuning architecture. By synergistically decoupling the spatial and spectral domains using the **Spatial-Prior Injection (SPI)** and **Adaptive Frequency Mask (AFM)** modules, our method achieves Pareto optimality on Out-of-Distribution (OOD) generalization benchmarks such as CIFAR-100-C.
>
> ## 🛠️ Environment Setup
>
> Please ensure you have Python 3.8+ installed. You can set up the environment using the provided requirements file:
>
> ```bash
> conda create -n decoupled_ft python=3.9 -y
> conda activate decoupled_ft
> pip install -r requirements.txt

## 🚀 Quick Start & Reproducibility

### 1. Main Benchmark Training

To reproduce the main benchmark results comparing linear probing, VPT, LoRA, DoRA, and our proposed composite architecture on CIFAR-100, simply run the bash script:

Bash

```
bash run_week3_main.sh
```

*The script will sequentially train all variants and save the consolidated metrics to `./outputs/summary.csv`.*

### 2. OOD Robustness Evaluation (CIFAR-100-C)

To validate the Out-of-Distribution (OOD) generalization capabilities, ensure you have downloaded the CIFAR-100-C dataset and placed it in `./data/CIFAR-100-C`. Then execute the evaluation script:

Bash

```
python scripts/eval_cifar100c.py
```

*This script will load the best checkpoints from the `./outputs/` directory, evaluate them across all 15 corruption types, and generate `Table3_CIFAR100C_Robustness.csv`.*

## 📜 Citation & License
