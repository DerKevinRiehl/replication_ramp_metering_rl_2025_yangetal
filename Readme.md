# [RE] Reinforcement Learning–Based Ramp Metering Strategy Considering Queue Management

A replication of “Reinforcement Learning–Based Ramp Metering Strategy Considering Queue Management” by Yang et al. (2025), published in the *Journal of Advanced Transportation*. DOI: [10.1155/atr/2838943](https://doi.org/10.1155/atr/2838943)

This repository accompanies the replication study by Patrick Barry, Kevin Riehl, Qiaosen Li, Anastasios Kouvelas, and Michail Makridis from the Institute for Transportation Systems and Planning, Traffic Engineering Group, ETH Zürich.

## About

Yang et al. (2025) propose an action-replacement module for reinforcement learning–based ramp metering. The module uses a store-and-forward queue model to estimate a lower bound for the ramp metering rate. Actions below this bound are replaced before being applied, preventing the reinforcement learning agent from selecting actions that are expected to cause ramp queue spillback. A penalty associated with replaced actions is included during training to guide the policy toward feasible control decisions.

This repository reconstructs and evaluates the proposed method using the SUMO microscopic traffic simulator. It contains separate single-ramp and multi-ramp scenarios, PPO-based ramp metering agents with and without action replacement, calibration and evaluation scripts, reconstructed SUMO networks, and scripts for reproducing the figures and analyses reported in the replication study.

The implementation includes the following controller families:

- **No Control:** All ramps remain fully open, providing an uncontrolled baseline.
- **PI-ALINEA:** Local feedback ramp metering based on measured downstream occupancy.
- **HERO:** Coordinated ramp metering for the multi-ramp scenario using PI-ALINEA controllers and upstream–downstream queue coordination.
- **PPO Baseline:** A reinforcement learning controller trained without action replacement.
- **PPO with Action Replacement:** The same reinforcement learning architecture augmented with the store-and-forward lower-bound constraint and replacement penalty proposed by Yang et al. (2025).

The experiments examine whether action replacement can prevent ramp queue spillback during training and evaluation without degrading overall traffic efficiency. Performance is assessed using total time spent, ramp queue lengths, spillback occurrence, metering actions, and related traffic-state measurements.

## Repository Structure

The repository separates simulation and data production from visualization. The single-ramp and multi-ramp scenarios have independent SUMO networks, training workflows, calibration utilities, and figure-data generators, while the PPO implementation and action-replacement logic are shared.

```
.
├── 0_original_papers/
│   └── 2025_Yang_et_al.pdf             # Original paper
├── 0_original_repository/              # Material from the original study, where available
├── 1_code_produce/                     # Simulation, training, and evaluation
│   ├── action_replacement.py           # Lower-bound and replacement-penalty calculations
│   ├── model.py                        # Shared PPO actor–critic network
│   ├── ppo_loss.py                     # PPO and generalized advantage estimation
│   ├── stats.py                        # Running state statistics
│   ├── live_plot.py                    # Live training visualization
│   ├── single_ramp/
│   │   ├── config.py                   # Scenario and detector configuration
│   │   ├── env.py                      # Single-ramp SUMO environment
│   │   ├── controllers.py              # No-control, PI-ALINEA, and PPO controllers
│   │   ├── runner.py                   # Episode execution and data collection
│   │   ├── train.py                    # PPO training
│   │   ├── train_multiple.py           # Multi-seed training
│   │   ├── evaluate_model.py           # Controller evaluation
│   │   ├── calibration/                # Reward, normalization, queue, and ALINEA calibration
│   │   └── figures/                    # Figure 6 and 7 data generators
│   └── multi_ramp/
│       ├── config.py                   # Four-ramp scenario configuration
│       ├── env.py                      # Multi-ramp SUMO environment
│       ├── controllers.py              # No-control, PI-ALINEA, HERO, and PPO controllers
│       ├── runner.py                   # Episode execution and data collection
│       ├── train.py                    # PPO training
│       ├── train_multiple.py           # Multi-seed training
│       ├── evaluate_model.py           # Controller evaluation
│       ├── calibration/                # Reward, capacity, and queue calibration
│       ├── figures/                    # Figure 12, 13, and 15 data generators
│       └── network_tools/              # Spatial-detector generation
├── 1_data_source/
│   ├── single_ramp/sumo_network/       # Single-ramp SUMO network and demand files
│   └── multi_ramp/sumo_network/        # Multi-ramp SUMO network and demand files
├── 2_data_produced/                    # Generated histories and intermediate figure data
├── 3_code_visualization/
│   ├── single_ramp/                    # Single-ramp plotting scripts
│   └── multi_ramp/                     # Multi-ramp plotting scripts
├── 3_data_visualization/               # Generated publication figures
├── requirements.txt                    # Python dependencies
└── Readme.md
```

## Installation Instructions

### Prerequisites

- Python 3.9 or newer
- [SUMO](https://sumo.dlr.de/docs/Downloads.php), including its TraCI Python interface
- A Python environment supported by PyTorch

The experiments can run entirely on CPU; a CUDA-capable GPU is optional.

### Environment setup

Clone the repository and create a virtual environment:

```bash
git clone <repository-url>
cd replication_ramp_metering_rl_2025_yangetal
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Install the Python dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

SUMO must be installed separately and its `sumo` executable must be available on `PATH`.

Verify the installation:

```bash
sumo --version
python -c "import traci, numpy, torch, matplotlib, pandas, seaborn, tqdm; print('Installation OK')"
```

## Reproduction Workflow

Run simulation, training, and evaluation commands from the corresponding scenario directory. Generated models are written to its `models/` directory, while intermediate data and figures are written to `2_data_produced/` and `3_data_visualization/`.

### Single-ramp scenario

```bash
cd 1_code_produce/single_ramp
```

The calibration utilities reproduce the normalization statistics, reward constants, PI-ALINEA target occupancy, and ramp-capacity analysis:

```bash
python calibration/calibrate.py --seeds 3
python calibration/tune_alinea.py
python calibration/ramp_capacity_test.py
```

Train the PPO baseline and action-replacement variants:

```bash
python train.py --num_episodes 500
python train.py --use_replacement --num_episodes 500
```

For a quick installation check, reduce `--num_episodes` to 5. The 500-episode commands represent the full training horizon used in the replication. To run both variants over seeds 0–9, use `python train_multiple.py`; this is substantially more expensive than a single training run.

Each run saves a final model, state tracker, training history, and periodic checkpoints under `models/`. Evaluate the single-ramp controllers over the configured evaluation seeds:

```bash
python evaluate_model.py
```

Generate the intermediate data required for reproducing Figures 6 and 7:

```bash
python figures/generate_figure6_data.py
python figures/generate_figure7_data.py
```

### Multi-ramp scenario

From the single-ramp directory, change to the multi-ramp directory:

```bash
cd ../multi_ramp
```

Run the calibration utilities when reproducing the constants used by the multi-ramp configuration:

```bash
python calibration/calibrate.py --seeds 3
python calibration/estimate_ramp_capacity.py
python calibration/estimate_max_queue.py --seeds 10
```

Train both PPO variants:

```bash
python train.py --seed 42 --num_episodes 500
python train.py --use_replacement --seed 42 --num_episodes 500
```

As above, five episodes are sufficient only for a smoke test. Run `python train_multiple.py` for the full ten-seed baseline and replacement training experiment.

Evaluate No Control, PI-ALINEA, HERO, and both PPO variants:

```bash
python evaluate_model.py
```

Generate the intermediate data required for Figures 12, 13, and 15:

```bash
python figures/generate_figure12_data.py
python figures/generate_figure13_data.py
python figures/generate_figure15_data.py
```

The spatial detectors used for Figure 15 are already included in the SUMO detector file. If the network is modified, regenerate them with:

```bash
python network_tools/generate_spatial_detectors.py
```

This command updates `detectors.add.xml` in place.

### Generate figures

Return to the repository root:

```bash
cd ../..
```

Generate the single-ramp figures:

```bash
python 3_code_visualization/single_ramp/plot_figure6.py
python 3_code_visualization/single_ramp/plot_figure7.py
python 3_code_visualization/single_ramp/plot_figure8.py
python 3_code_visualization/single_ramp/plot_figure9.py
python 3_code_visualization/single_ramp/plot_figure10.py
python 3_code_visualization/single_ramp/plot_ramp_capacity.py
```

Generate the multi-ramp figures:

```bash
python 3_code_visualization/multi_ramp/plot_figure_12.py
python 3_code_visualization/multi_ramp/plot_figure_13.py
python 3_code_visualization/multi_ramp/plot_figure_14.py
python 3_code_visualization/multi_ramp/plot_figure_15.py
python 3_code_visualization/multi_ramp/plot_figure_16.py
python 3_code_visualization/multi_ramp/plot_figure_17.py
python 3_code_visualization/multi_ramp/plot_figure_18.py
python 3_code_visualization/multi_ramp/plot_action_comparison.py
python 3_code_visualization/multi_ramp/plot_ramp_queue.py
```

Training and calibration take substantially longer than plotting. Runtime depends on the number of episodes and random seeds, the available CPU cores, and the installed SUMO version.

## Outputs and File Formats

Outputs are generated rather than tracked as source code:

- `1_code_produce/<scenario>/models/model_<variant>_seed<N>.pth`: final seed-specific PyTorch `state_dict` checkpoints. Seed 42 is also saved without the seed suffix for use by the evaluation scripts; periodic checkpoints include `_seed<N>_ep<N>`.
- `1_code_produce/<scenario>/models/state_tracker_<variant>_seed<N>.pkl`: seed-specific state-normalization statistics saved with Python `pickle`. Seed 42 is also saved under the canonical filename expected by evaluation.
- `1_code_produce/<scenario>/models/training_history_<variant>_seed<N>.pkl`: episode rewards, interaction-step counts, episode lengths, TTS, replacement percentages, and lower bounds.
- `2_data_produced/action_comparison.pkl` and `multi_ramp_action_comparison.pkl`: evaluation actions and metadata for the single- and multi-ramp scenarios.
- `2_data_produced/figure_6_data.pkl`, `figure_7_matrix.npy`, `figure_12_data.pkl`, `figure_13_data.pkl`, and `fig_15_speed_grid_*.npy`: intermediate arrays used by the figure scripts.
- `2_data_produced/multi_ramp_queue_data.pkl` and `multi_ramp_max_queue_estimates.json`: ramp-capacity and storage-calibration outputs.
- `3_data_visualization/*.pdf`: generated replication figures. Diagnostic training plots may additionally be written as PNG files beside the training scripts.

## Replication Notes

The original networks, exact demand profiles, and several implementation parameters were unavailable. The SUMO environments and multi-ramp demand were therefore reconstructed from the paper and satellite imagery, while missing capacities, controller settings, and PPO hyperparameters were calibrated or selected using standard values.

These assumptions mean that this repository reproduces the published method but not the unavailable original implementation. Full details and justification are provided in the accompanying replication paper.

## Results and Limitations

The replication supports the central qualitative finding that action replacement reduces queue-spillback failures and improves training stability without materially slowing convergence.

However, the original study's absolute performance improvements could not be reproduced. Differences in the reconstructed network, demand profile, and ramp capacities prevent a direct numerical comparison, so the results validate the mechanism's functional behavior rather than the original travel-time claims.

## Citation
Replication Study:
```
Citation metadata will be added after publication.
```

Original Paper:
```
Yang, Y., Yu, S., Ding, F., & Han, Y. (2025). Reinforcement learning–based ramp metering strategy considering queue management. Journal of Advanced Transportation, 2025(1), 2838943. https://doi.org/10.1155/atr/2838943
```