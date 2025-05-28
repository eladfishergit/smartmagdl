# SMARTMAG-DL v1.0
*A MATLAB toolkit for fast, physics‑based simulation of magnetic maps and synthetic smartphone magnetometer data.*

---

## 1 What is SMARTMAG-DL?

SMARTMAG-DL (SMARTphone **MAG**netometry **D**eep‑Learning) lets you create realistic training datasets for learning‑based indoor magnetic‑field mapping and localization.  
It couples an analytical magnetostatics engine (cylindrical dipoles solved with Bulirsch–Heuman elliptic integrals) to a kinematic/noise model of a walking smartphone. The result is a pair of perfectly aligned “ground‑truth” and “measured” magnetic maps—exactly what you need to benchmark interpolation networks, SRResNet‑type super‑resolution, or classical inverse methods.

![pipeline](docs/pipeline_overview.svg)

---

## 2 Key features

| Feature | File(s) | Notes |
|---------|---------|-------|
| **Analytic field of finite cylinders** | `ParallelBfield.m`, `ParallelDerby.m`, `ParallelBulirsch.m` | Vectorised Bulirsch algorithm gives μT accuracy at kHz throughput. |
| **Workspace generator** | `buildWorkspace.m` | Planar or cylindrical sensor grids with arbitrary spacing. |
| **Trajectory densification** | `densifyTrajectory.m` | Inserts virtual samples between GNSS/SLAM way‑points to emulate high‑rate IMU. |
| **Ground‑truth field builder** | `build_ground_truth_all.m` | Random magnet count, pose, size and magnetisation for each map. |
| **Synthetic smartphone loop** | `build_full_complete_database_HRDTraj.m` | Adds Earth field, gait kinematics, pitch/roll, soft‑ & hard‑iron, axis mis‑alignment, scale, Gaussian noise and bias. |
| **Batch simulator** | same file (`Nsim` loop) | Generates hundreds of aligned datasets in one run. |

---

## 3 Requirements

| Item | Version / comment |
|------|-------------------|
| **MATLAB** | R2020b or newer (uses implicit expansion & `griddata`) |
| **Toolboxes** | none mandatory; `Statistics and Machine Learning Toolbox` accelerates some steps |
| **Hardware** | any modern CPU; ≥8 GB RAM recommended for 300 × 300 × 100 grids |

---

## 4 Quick start

```matlab
% Clone and set path
git clone https://github.com/your-org/smartmagdl.git
cd smartmagdl
addpath(genpath(pwd));

% One‑shot verification
demo_flag_plot = true;          % show maps
P = load('traj_data.mat').P;    % your pre‑recorded XY trajectory
[bx,by,bz,~,~,~,~,~] = build_ground_truth_all(demo_flag_plot,P);

% Full synthetic dataset (100 realisations)
build_full_complete_database_HRDTraj;  % outputs to ./data_dir_root
```

### Output structure

```
data_dir_root/
 ├─ info_#.mat        % magnet poses, calibration matrices
 ├─ TheorA_#.mat      % Bx,By,Bz ground truth
 ├─ TheorB_#.mat      % ground‑truth + Earth field
 ├─ TheorE10_#.mat    % super‑resolved 10 cm map
 ├─ SimA_#.mat        % raw smartphone traces
 └─ SimB10_#.mat      % noisy 10 cm interpolation
```

---

## 5 Folder structure

```
smartmagdl/
 ├─ core/                 % magnetostatics engine
 │   ├─ ParallelBfield.m
 │   ├─ ParallelDerby.m
 │   └─ ParallelBulirsch.m
 ├─ utils/
 │   ├─ densifyTrajectory.m
 │   └─ buildWorkspace.m
 ├─ builders/
 │   ├─ build_ground_truth_all.m
 │   └─ build_full_complete_database_HRDTraj.m
 ├─ examples/             % usage demos & notebooks
 ├─ data/                 % (empty) place your traj_data.mat here
 └─ README.md
```

---

## 6 Citing

```bibtex
@misc{smartmagdl2025,
  author    = {Fisher, Elad and Masiero, Federico},
  title     = {{SMARTMAG-DL}: Synthetic Magnetic Dataset Generator},
  year      = {2025},
  url       = {https://github.com/your-org/smartmagdl},
  note      = {MIT Licence}
}
```

---

## 7 Licence

SMARTMAG-DL is released under the **MIT License** – free for academic and commercial use.  
See `LICENSE` for the full text.

---

## 8 Contact / support

Please open a GitHub Issue for bug reports or feature requests.  
For other questions contact **Elad Fisher** – <eladfisher.mail@gmail.com>
