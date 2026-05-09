# LED

Code for computing LED scores and selecting compact robot-learning datasets. This minimal release is organized around two basic workflows:

1. compute LED scores for a robot-learning dataset;
2. select a compact filtered subset and optionally export it back to an RLDS/TFDS dataset.

The included example uses LIBERO-10 as the test dataset. The example TFDS/RLDS data can be downloaded from the OpenVLA modified LIBERO RLDS dataset: https://huggingface.co/datasets/openvla/modified_libero_rlds. This release provides the extracted OpenVLA feature file used by the minimal reproduction.

## Installation

```bash
git clone https://github.com/bossxjh/LED.git
cd LED

conda create -n led python=3.10 -y
conda activate led

git lfs install
pip install -r requirements.txt
pip install -e .
```

Downloaded model weights are cached under `./checkpoint` by default.

## Data Format

This release provides a ready-to-use example feature file and a standard dataset location:

- example RLDS/TFDS dataset location: `./datasets/libero_10_no_noops`
- example extracted OpenVLA features: `./feature/rlds-libero-10_openvla_nf3_bs8.npz`

The feature `.npz` stores per-demo visual features, task ids, task descriptions, episode indices, and action statistics. The selection script uses the episode indices to map the selected demos back to the original RLDS/TFDS dataset.

## 1. Compute LED

Compute LED directly from the included example feature file:

```bash
python -m scripts.test_leanability_v5 \
  --npz_ten ./feature/rlds-libero-10_openvla_nf3_bs8.npz \
  --skip_benches goal,object,spatial \
  --transfer_mode harmonic \
  --task_knn 7 \
  --task_temp 0.07 \
  --alpha 0.35 \
  --beta 0.5 \
  --pi_scale 0.02 \
  --tau_floor 0.03 \
  --plot \
  --out_dir ./plots/libero10
```

The script prints the dataset LED score (`leanability_dataset`) and task-level scores. With `--plot`, it also saves a score-vs-ground-truth plot under `./plots/libero10`.

## 2. Select a Filtered Dataset

Run LED-based subset selection on the included example feature file:

```bash
python -m dataeval.metric.task_subset_select_v2 \
  --in_npz ./feature/rlds-libero-10_openvla_nf3_bs8.npz \
  --out_dir ./feature/selected_led \
  --ratios 0.8,0.6,0.4,0.2 \
  --best_restarts 10 \
  --random_max_samples 200000 \
  --random_max_patience 30000 \
  --seed 0 \
  --use_fixed_tau_for_search \
  --task_knn 7 \
  --task_temp 0.07 \
  --alpha 0.35 \
  --beta 0.5 \
  --pi_scale 0.02 \
  --tau_floor 0.03
```

Outputs are written to `--out_dir`:

- `filtered_r0.8.npz`, `filtered_r0.6.npz`, ...: LED-selected subsets;
- `selection_report.json`: selected local/global indices, episode indices, and per-task scores.

To materialize a selected subset back into an RLDS/TFDS dataset, use:

```bash
python -m scripts.make_filtered_rlds_tfds \
  --src_dataset_path ./datasets/libero_10_no_noops \
  --dst_dataset_path ./datasets/libero_10_no_noops_r0_6_led \
  --filtered_npz ./feature/selected_led/filtered_r0.6.npz \
  --split train \
  --max_examples_per_shard 256
```

The new dataset can then be loaded with `tensorflow_datasets` from the parent directory of `--dst_dataset_path`.

## Useful Files

- `scripts/get_feature_npz.py`: extract per-demo features and metadata from an RLDS/TFDS dataset.
- `scripts/test_leanability_v5.py`: compute LED scores and example benchmark correlations.
- `dataeval/metric/task_subset_select_v2.py`: select high-LED and low-LED subsets from a feature `.npz`.
- `scripts/make_filtered_rlds_tfds.py`: rewrite selected episode indices into a filtered RLDS/TFDS dataset.

## Notes

- The default hyperparameters above are the setting used for the included example dataset.
- The included `.npz` file lets you skip feature extraction for the two minimal experiments.
- If you want to re-extract features, use `scripts/get_feature_npz.py` with `--dataset_path ./datasets/libero_10_no_noops`.
