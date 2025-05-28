# SMARTMAG-DL v1.1
*A MATLAB + PyTorch toolkit for physics‑based simulation **and** neural super‑resolution of indoor magnetic maps.*

---

## 1 What is SMARTMAG‑DL?

SMARTMAG‑DL (SMARTphone **MAG**netometry **D**eep‑Learning) generates paired *ground‑truth* and *measured* magnetic maps and upgrades the measured maps with a **SRResNet** super‑resolution network.  
The pipeline:

1. **Magnetostatics engine** (MATLAB) 👉 analytical field of finite cylinders.  
2. **Trajectory & noise model** 👉 synthetic smartphone readings.  
3. **Python SRResNet** 👉 learns to up‑sample noisy 1 m grids to high‑resolution (10 cm) maps.

![pipeline](docs/pipeline_overview.svg)

---

## 2 Key features

| Feature | File(s) | Notes |
|---------|---------|-------|
| **Analytic field of finite cylinders** | `ParallelBfield.m`, `ParallelDerby.m`, `ParallelBulirsch.m` | μT accuracy, kHz throughput. |
| **Workspace generator** | `buildWorkspace.m` | Flexible 2‑D/3‑D grid creation. |
| **Trajectory densification** | `densifyTrajectory.m` | Converts sparse GNSS/SLAM path to high‑rate track. |
| **Ground‑truth field builder** | `build_ground_truth_all.m` | Random magnet population per simulation. |
| **Synthetic smartphone loop** | `build_full_complete_database_HRDTraj.m` | Adds Earth field, gait kinematics, calibration errors, noise. |
| **Neural up‑sampler (training)** | `Train_SRResNet_smartmagdl.py` | Trains SRResNet on 1000 paired maps (takes ≈3 h on RTX 3080). |
| **Neural up‑sampler (inference)** | `Load_trained_model_SRResNet_smartmagdl.py` | Loads a checkpoint and predicts 100 test maps in <1 s each. |
| **Batch simulator** | see previous file | Generates hundreds of aligned datasets in one go. |

---

## 3 Requirements

| Component | Version / comment |
|-----------|-------------------|
| **MATLAB** | R2020b or newer |
| **Python** | 3.9 or newer |
| **PyTorch** | ≥2.0 (+ CUDA 11) |
| **Other Python libs** | numpy, scipy, scikit‑learn, ignite, matplotlib |
| **Hardware** | – CPU: any; – GPU: ≥8 GB VRAM recommended for training |
| **Toolboxes (MATLAB)** | none mandatory |

---

## 4 Quick start

<details><summary>MATLAB — generate synthetic dataset</summary>

```matlab
git clone https://github.com/your-org/smartmagdl.git
cd smartmagdl
addpath(genpath(pwd));

% Generate 1000 paired maps (ground truth + smartphone)
build_full_complete_database_HRDTraj;   % → ./data_dir_root
```
</details>

<details><summary>Python — train SRResNet</summary>

```bash
cd smartmagdl/nn
python Train_SRResNet_smartmagdl.py   # ~3 h on single GPU
```
</details>

<details><summary>Python — run inference demo</summary>

```bash
python Load_trained_model_SRResNet_smartmagdl.py   # loads .pth checkpoint
```
</details>

---

## 5 Folder structure

```
smartmagdl/
 ├─ core/                     % magnetostatics engine (MATLAB)
 ├─ utils/                    % densifyTrajectory, buildWorkspace
 ├─ builders/                 % synthetic dataset generators
 ├─ nn/                       % deep‑learning scripts (Python)
 │   ├─ Train_SRResNet_smartmagdl.py
 │   └─ Load_trained_model_SRResNet_smartmagdl.py
 ├─ examples/
 ├─ data/                     % (empty) – put your traj_data.mat here
 └─ README.md
```

---

## 6 Citing

```bibtex
@misc{smartmagdl2025,
  author    = {Fisher, Elad and others},
  title     = {{SMARTMAG-DL}: smartmagdl TBD},
  year      = {2025},
  url       = {https://github.com/your-org/smartmagdl},
  note      = {MIT License}
}
```

---

## 7 License

This project is released under the **MIT License** – free for academic and commercial use.

---

## 8 Contact / support

Open a GitHub Issue or contact **Elad Fisher** <elad.fisher@example.com>.

---

Happy mapping & super‑resolving!
