# Dueling DQN for Pendulum Control Across Gravity Conditions

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rudhresh_R-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rudhresh-r/)
[![GitHub](https://img.shields.io/badge/GitHub-rudybrrr-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rudybrrr)

Can a discretised DQN controller reliably balance Pendulum, and how does a frozen control methodology behave when gravity changes? This project develops a Dueling DQN controller for the legacy Gym `Pendulum-v0` environment, then trains four separate fresh policies under fixed gravity conditions.

**Author:** Agne Rudhresh

![Representative greedy rollouts from one pre-specified held-out initial state across four gravity conditions](assets/gravity_comparison_preview.png)

## Overview

The study validates the environment and its physics, establishes random and untrained references, selects a seven-action torque representation, compares DQN variants and training choices, and retains Dueling DQN. The complete configuration is then frozen before four independent gravity-specific models are trained and evaluated on protected physical initial states.

This is **not** one gravity-conditioned universal policy. It contains four separately trained Dueling DQN policies, each using the same frozen recipe:

| Policy | Gravity |
| --- | ---: |
| Default | `g = 10` |
| Zero gravity | `g = 0` |
| Anti-gravity | `g = -10` |
| Supergravity | `g = 15` |

## Control Problem

Gym exposes the Pendulum state as raw `[cos(theta), sin(theta), theta_dot]` and accepts a continuous torque in `[-2, 2]`. Standard DQN selects from a finite action set, so the continuous torque interval is discretised. The retained policy selects one of seven uniformly spaced torques:

`[-2.0, -1.333333, -0.666667, 0.0, 0.666667, 1.333333, 2.0]`

The corrected smoothness audit distinguishes **action switching** from **torque-direction switching**; they are not the same metric.

| Actions | Action switching | Direction switching | Mean absolute torque change |
| ---: | ---: | ---: | ---: |
| 3 | 0.739322 | 0.739322 | 2.597990 |
| 5 | 0.780151 | 0.770519 | 2.338777 |
| 7 | 0.456658 | 0.451214 | 0.908710 |

## Model Development

The notebook preserves the selection path: random and untrained references; action-count investigation; Vanilla, Double and Dueling DQN; exploration, replay/target-network and optimisation comparisons; a state-scaling experiment; combined retained configuration; Dueling-head ablation; and a fresh-seed confirmation. It was a hypothesis-driven programme, not an exhaustive search.

### Final Dueling DQN

| Component | Frozen choice |
| --- | --- |
| Architecture | Dueling DQN |
| Hidden layers | 64, 64 ReLU |
| Trainable parameters | 4,936 |
| Exploration | epsilon `1.00 → 0.05` over 20,000 steps |
| Replay | capacity 10,000; warm-up 1,000; batch size 32 |
| Target network | hard copy every 250 optimiser updates |
| Optimisation | Adam, learning rate 0.001, MSE TD loss, no gradient clipping |
| Gamma | 0.99 |
| Training budget | 150 episodes / 30,000 environment steps |
| Development evaluations | episodes 0, 25, 50, 75, 100, 125, 150 |

In the final Dueling-head ablation, combined Dueling DQN returned `-135.942190` versus `-139.818601` for the standard-network ablation: a `-3.876411` difference in favour of Dueling. Dueling won most paired development states and added only 65 parameters, but the difference remained inside the project’s practical variation margin; this is not a statistical-significance claim.

Three predeclared fresh default-gravity seeds confirmed the frozen selection:

| Training seed | Selected development return |
| ---: | ---: |
| 16180 | -138.166628 |
| 27182 | -140.121455 |
| 31415 | -138.965574 |
| **Mean (SD)** | **-139.084552 (0.982829)** |

## Frozen Evaluation Protocol

A development suite was used for comparisons, checkpoint selection and tuning. A separate suite of 24 physical initial states stayed closed during training, model selection, checkpoint selection and gravity development, then was released only after the architecture, recipe, and all four gravity-specific policies were frozen. Comparisons are paired on the same physical states and use same-gravity references; raw returns across gravity settings are not direct cross-environment model-quality rankings.

The four-gravity runs use matched training seed `4242`.

## Four-Gravity Results

| Gravity | Selected episode | Development mean return | Upright fraction |
| --- | ---: | ---: | ---: |
| Default `g = 10` | 50 | -144.352853 | 0.864167 |
| Zero `g = 0` | 50 | -48.096170 | 0.910625 |
| Anti-gravity `g = -10` | 125 | -52.225482 | 0.902917 |
| Supergravity `g = 15` | 125 | -204.892825 | 0.827292 |

All trained policies outperformed their own same-gravity random and untrained references in the recorded evaluation. Checkpoint selection matters: supergravity fell from `-204.892825` at its selected episode 125 checkpoint to `-325.844027` at episode 150. That selection was frozen before held-out evaluation.

### Protected Held-Out Results

| Gravity | Mean return on 24 held-out states |
| --- | ---: |
| Default `g = 10` | -135.383892 |
| Zero `g = 0` | -53.772390 |
| Anti-gravity `g = -10` | -53.369357 |
| Supergravity `g = 15` | -203.310855 |

## Videos

The preview uses stills from the original supplied videos. Each video is a greedy 200-step rollout from the pre-specified held-out state **F08**, `(theta, theta_dot) = (1.35, 0.40)`—not a manually selected trajectory.

| Gravity | Representative return | Video |
| --- | ---: | --- |
| Default | -120.593603 | [MP4](videos/default_gravity.mp4) |
| Zero gravity | -22.733082 | [MP4](videos/zero_gravity.mp4) |
| Anti-gravity | -21.390217 | [MP4](videos/anti_gravity.mp4) |
| Supergravity | -120.119055 | [MP4](videos/supergravity.mp4) |

## Reproducibility

The published files are inference weights. Fresh Dueling reconstruction from each saved file produced seven outputs, 4,936 parameters, finite Q-values, and exactly reproduced the stored held-out metrics (absolute mean-return difference `0.0`). They do not preserve replay memory, optimiser state, or enough state to resume interrupted training exactly.

`best_dqn.weights.h5` is deliberately omitted because its SHA-256 hash is byte-identical to `models/final_dqn_default.weights.h5`.

## Running the Project

This public notebook is an executed research record. It preserves the code and experiment outputs but does not redistribute historical replay buffers, training histories, or temporary checkpoints used by the original formal-run workflow. Install the legacy-compatible dependencies, then open the notebook or rendered report.

```bash
python -m pip install -r requirements.txt
jupyter lab
```

The original work uses **Gym 0.17.3** and **`Pendulum-v0`**. It is intentionally not modernised to Gymnasium or a newer API.

## Limitations

- DQN requires a discretised torque approximation for this continuous-control task.
- Training is limited to 150 episodes / 30,000 steps, and checkpoint sensitivity was observed.
- The stored files contain inference weights only, not replay memory or optimiser state.
- Smoothness proxies do not directly measure physical jerk or actuator wear.
- The study does not establish that each gravity setting received independently optimal tuning.
- Different gravity environments have different dynamics, so their raw returns should not be read as direct quality rankings.

## Repository Structure

```text
notebooks/  Cleaned public experiment record
reports/    HTML export regenerated from the cleaned notebook
models/     Four distinct frozen gravity-specific inference weights
videos/     Fixed-state greedy rollout demonstrations
assets/     README comparison preview
```

## My Contribution

This project was completed collaboratively as a two-person Deep Learning coursework project.

My primary responsibilities focused on the evaluation and experimental methodology, including:

- fixed development and protected held-out state design
- matched training-seed policy and practical model-comparison criteria
- gravity hypotheses, physics interpretation and same-gravity comparison logic
- final four-gravity analysis, paired-comparison and stability review
- reproducibility evidence and saved-weight verification

The core DQN implementation, Gym/Pendulum setup, replay and target-network logic, action selection, training loop, formal runs, and notebook integration were primarily developed by my project partner. Final Dueling DQN selection and interpretation were reviewed jointly.

### Collaborator

**Mohamad Aniq Bin Mohamad Hisyam**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/xTurtleXP) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/mohamadaniq/)

## Coursework Context

This project originated as coursework for a Deep Learning module at Singapore Polytechnic and has been cleaned and reorganised for public presentation.
