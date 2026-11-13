# PancreaAI

A student research project: deep learning models that find and outline the pancreas and pancreatic tumors in abdominal CT scans.

**This is an educational project. It is not a medical device and must not be used for diagnosis.**

## Status

| Stage | What | State |
|---|---|---|
| 1 | Foundations: environment, CT basics, how a model learns | Done |
| 2 | Data: download, patient-level split, dataset statistics | Done |
| 3 | Pancreas segmentation with a U-Net | In progress |
| 4 | Tumor detection | Planned |
| 5 | Tumor segmentation and volume | Planned |
| 6 | Research: small tumors, error analysis, comparison with papers | Planned |
| 7 | Streamlit app | Planned |

## Experiment log

All scores are on the 42 validation patients. The 42 test patients have not been used.

| # | Date | What was tried | Result | Conclusion |
|---|---|---|---|---|
| 1 | 2026-11-13 | Baseline: hand-written 2D U-Net (483,153 weights), 5 epochs, 256×256 slices | Average Dice per patient in 3D: 0.639. Best 0.876, worst 0.209. 7 of 42 patients below 0.5 | Works on most patients. The failures are misses: in 5 of the 7 failing patients the model coloured under 40% of the true pancreas |
| 2 | 2026-11-13 | Hypothesis: the model is too cautious. Lowered the decision bar from 0.5 down to 0.1 | Average Dice 0.640 at 0.1 against 0.639 at 0.5. Still 7 failing patients | Hypothesis rejected. The model is confident where it is wrong, so changing the bar cannot fix it. The model itself has to improve |
| 3 | 2026-11-13 | Hypothesis: the model is too small. Trained the same U-Net with twice the filters per level (1,927,841 weights, four times as many). Same data, 5 epochs, same settings | Average Dice 0.633 against 0.639. Middle 0.695 against 0.658. Worst 0.064 against 0.209. Still 7 failing patients, and 6 of them are the same patients | Hypothesis rejected. A bigger model does not help, and the same patients fail in both, so the problem is in those scans and in what a 2D slice model can see. One training run each, so a gap of 0.006 is too small to call either model the winner |
| 4 | 2026-11-13 | Error analysis: why do the same six patients fail with both models? Compared slice thickness, pancreas volume, tumor volume, pixel size and pancreas brightness, then drew the six scans (`outputs/failures.png`) | None of the five measurements separates the six failures from the other 36 patients. In the pictures the model outlines one piece of the pancreas and misses the rest of a long, stretched shape | The failure is "one piece only". New hypothesis: a model that reads one slice at a time cannot follow the pancreas along its length. To be tested with a 3D model. Based on one slice per patient, so this is an observation, not a measurement |

Files: `outputs/train_log_unet_small.csv`, `outputs/val_results_unet_small.csv`, `outputs/threshold_results.csv`, `outputs/train_log_unet_big.csv`, `outputs/val_results_unet_big.csv`.

## Folder layout

| Path | Contents |
|---|---|
| `scripts/` | All Python code |
| `outputs/` | Pictures and tables the scripts produce |
| `splits.json` | Which patient is in the training, validation, or test group. Fixed, never reshuffled |
| `data/` | The CT scans. Not uploaded to GitHub |
| `.venv/` | The Python environment. Not uploaded to GitHub |

## Scripts

Run every script from the project folder, not from inside `scripts/`:

```bash
cd ~/pancreaai
source .venv/bin/activate
python scripts/view_scan.py
```

| Script | What it does |
|---|---|
| `check_setup.py` | Confirms PyTorch and MONAI work and a 3D convolution runs on the Mac's GPU |
| `view_scan.py` | Opens one CT scan and draws the doctor's pancreas and tumor outlines. Saves `outputs/first_scan.png` |
| `learn_demo.py` | The smallest possible model: one weight learns the rule "multiply by 3" |
| `make_split.py` | Splits the 281 patients into training (197), validation (42), and test (42). Saves `splits.json` |
| `dataset_stats.py` | Measures pancreas and tumor volume for every patient. Saves `outputs/dataset_stats.csv` |
| `preprocess_demo.py` | Shows one slice before and after preparation: window, normalise, shrink |
| `make_slices.py` | Turns the 3D scans into a 2D slice dataset for training and validation. Saves `data/slices/` |
| `unet.py` | The 2D U-Net model, written by hand. `width` sets its size: 16 = small, 32 = big |
| `train.py` | Trains the U-Net. Run as `python scripts/train.py NAME WIDTH`. Saves the best model to `models/NAME.pt` and a log to `outputs/train_log_NAME.csv` |
| `show_predictions.py` | Draws the model's outline next to the doctor's on four validation patients |
| `evaluate.py` | Run as `python scripts/evaluate.py NAME WIDTH`. Scores every validation patient in full 3D. Saves `outputs/val_results_NAME.csv` |
| `threshold_test.py` | Experiment 2: scores the model at seven different decision bars |
| `show_failures.py` | Experiment 4: draws the six patients that fail with both models. Saves `outputs/failures.png` |

## Dataset

Medical Segmentation Decathlon, Task07 Pancreas: 420 portal-venous CT scans from Memorial Sloan Kettering Cancer Center, with expert masks for pancreas and tumor on 281 of them. Licensed CC-BY-SA 4.0.

Measured in this project (`outputs/dataset_stats.csv`):

- All 281 labelled patients have a tumor. There are no healthy cases.
- Tumor volume ranges from 413 mm³ to 732,388 mm³. The middle value is 5,778 mm³.
- 110 tumors are smaller than a 2 cm ball (4,189 mm³): 82 in training, 12 in validation, 16 in test.

Download (12.3 GB):

```bash
mkdir -p data
curl -L -C - -o data/Task07_Pancreas.tar https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar
tar -xf data/Task07_Pancreas.tar -C data
```

Citation: Simpson et al., "A large annotated medical image dataset for the development and evaluation of segmentation algorithms", arXiv:1902.09063.

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch monai numpy SimpleITK scikit-learn matplotlib
```
