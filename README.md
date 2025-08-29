# CDMP & CDMP-pen
<div align="left">

<img width="1536" height="680" alt="framework_00" src="https://github.com/user-attachments/assets/267d3cfa-1362-418b-9493-393ef9bd0745" />


This repository, corresponding to the paper https://arxiv.org/abs/2503.17693, presents the Conditional Diffusion Model Planner (**CDMP**) for high-dimensional offline resource allocation in clustered ad hoc networks. CDMP leverages the generative power of Diffusion Models (DMs) to accurately capture complex environmental dynamics and integrates an inverse dynamics model for effective policy planning. By incorporating a theoretically grounded, uncertainty-aware penalty metric, the approach is further extended to the **CDMP-pen** algorithm, which mitigates Out-of-Distribution (OOD) distributional shift and enhances robustness. Extensive experiments demonstrate that our model consistently outperforms Model-Free RL (MFRL) and other Model-Based RL (MBRL) algorithms in terms of average reward and QoS, validating its practicality and scalability for real-world network resource allocation tasks.

Note that this is a research project and by definition is unstable. Please write to us if you find something not correct or strange. We are sharing the codes under the condition that reproducing full or part of codes must cite the paper.

## Installation

```
conda env create -f environment.yml
conda activate cdmp
export PYTHONPATH=/path/to/cdmp/
```

## Dataset Preparation

To train and evaluate the CDMP & CDMP-pen, an **offline dataset** is required. The dataset should include the following NumPy files:

- `actions.npy` – action sequences
- `observations.npy` – state trajectories
- `rewards.npy` – reward signals
- `terminals.npy` – terminal flags indicating episode ends

All files must be placed inside the `dataset/` folder at the project root. The folder structure should look like:

```
project_root/
├── diffuser/
├── dataset/
│   ├── actions.npy
│   ├── observations.npy
│   ├── rewards.npy
│   └── terminals.npy
├── ...
├── train.py
├── eval.py
└── README.md
```

Make sure the `.npy` files are consistent in episode length and dimension across all modalities before running training scripts.

## Training

To start training **CDMP** or **CDMP-pen**, first set the training mode in `config/locomotion_config.py`:

- For **CDMP**: set

```
train_CDMP_pen = False
```

- For **CDMP-pen**: set

```
train_CDMP_pen = True
```

Then run the following command to begin training:

```bash
python train.py
```

You can modify the training and evaluation configuration by appropriately changing `analysis/default_inv.py`.

## Evaluation

We conduct simulation-based evaluation on the **OPNET platform**. During testing, the interaction between OPNET and CDMP/CDMP-pen is handled through intermediate files:

- The OPNET platform writes environment state information into `state_b.txt`.
- The trained CDMP or CDMP-pen model reads these states, outputs the corresponding actions, and writes them into `action_b.txt`.

The default interaction path is defined in `scripts/evaluate_inv_parallel.py` as:

```
INTERACT_DIR_PATH = './OPNET'
state_file = os.path.join(INTERACT_DIR_PATH, "state_b.txt")
action_file = os.path.join(INTERACT_DIR_PATH, "action_b.txt")
```

To evaluate the trained model, run the following command:

```bash
python eval.py
```

## Integrating CDMP & CDMP-pen with your own framework

Besides the **U-Net** architecture, we also provide a simplified **MLP** architecture for noise prediction, implemented in `diffuser/models/temporal.py`, to facilitate further experimentation. You can extend or modify this module to design your own denoising backbone. The framework is model-agnostic, allowing seamless integration of alternative architectures to explore diverse diffusion-based planning strategies.

## Reference

```bibtex
@article{meng2025conditional,
  title={Conditional Diffusion Model with OOD Mitigation as High-Dimensional Offline Resource Allocation Planner in Clustered Ad Hoc Networks},
  author={Kechen, Meng and Sinuo, Zhang and Rongpeng, Li and Chan, Wang and Ming, Lei and Zhifeng, Zhao},
  journal={IEEE Trans. Commun. },
  year={2025},
  month={Aug},
  note={accepted}
}
```
## Acknowledgements

The codebase is derived from [decision-diffuser repo](https://github.com/anuragajay/decision-diffuser/tree/main).
