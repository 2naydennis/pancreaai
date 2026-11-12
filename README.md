# PancreaAI

A student research project: deep learning models that find and outline the pancreas and pancreatic tumors in abdominal CT scans.

**This is an educational project. It is not a medical device and must not be used for diagnosis.**

## Status

| Stage | What | State |
|---|---|---|
| 1 | Foundations: environment, CT basics, how a model learns | Done |
| 2 | Data: download, patient-level split, dataset statistics | Done |
| 3 | Pancreas segmentation with a U-Net | Next |
| 4 | Tumor detection | Planned |
| 5 | Tumor segmentation and volume | Planned |
| 6 | Research: small tumors, error analysis, comparison with papers | Planned |
| 7 | Streamlit app | Planned |

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
