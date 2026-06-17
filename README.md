1. Local MLflow (First Step)
   What happens here?

MLflow stores experiments on your local machine and you can view them using the MLflow UI.

Run MLflow locally
Step 1: Run your training script
python example.py
Step 2: Start MLflow UI
mlflow ui
Step 3: Open in browser
http://127.0.0.1:5000
What you see in MLflow UI
Experiments
Parameters (alpha, l1_ratio)
Metrics (RMSE, MAE, R2)
Models (saved locally)

Limitation of local MLflow
Only works on your PC
Cannot share results
No collaboration
No cloud tracking

connect your repo of github to dagshub and then connection to experiment

2.  Move to DAGsHub (Remote MLflow Server)

Now we connect MLflow to cloud using DAGsHub.

🔗 DAGsHub MLflow Tracking URI
https://dagshub.com/Mzn310/Chest_project.mlflow
Setup DAGsHub integration
Step 1: Install dependencies
pip install mlflow dagshub
Step 2: Connect MLflow to DAGsHub
import dagshub

dagshub.init(
repo_owner='Mzn310',
repo_name='Chest_project',
mlflow=True
)
Step 3: Run your MLflow code normally
import mlflow

with mlflow.start_run():
mlflow.log_param("alpha", 0.5)
mlflow.log_metric("rmse", 0.7) 3. After DAGsHub Integration

Now your experiments are stored in the cloud.

View results online:
https://dagshub.com/Mzn310/Chest_project

Go to:
Experiments / MLflow tab

4.  Complete Workflow Summary
    Step 1 (Local testing)
    mlflow ui
    Step 2 (Experiment locally)
    python example.py
    Step 3 (Connect to cloud)
    dagshub.init(..., mlflow=True)
    Step 4 (Run again)
    python example.py

This setup deploys a self-hosted MLflow tracking server on an EC2 instance with artifact storage on S3. Requires an AWS account.

Step 1 — AWS Infrastructure Setup
Log in to the AWS Management Console
Create an IAM user with AdministratorAccess policy and download the access keys
Configure the AWS CLI locally by running:
aws configure
Provide your Access Key ID, Secret Access Key, and preferred region.
Create an S3 bucket (e.g. mlflow-tracking-buc25) — this will store all MLflow artifacts
Launch an EC2 instance (Ubuntu recommended) and open port 5000 in the Security Group inbound rules
Step 2 — EC2 Server Setup
SSH into your EC2 instance and run the following commands:

# Update packages and install dependencies

sudo apt update
sudo apt install python3-pip pipenv virtualenv -y

# Create project directory and install the MLflow stack

mkdir mlflow && cd mlflow
pipenv install mlflow awscli boto3
pipenv shell

# Configure AWS credentials on the EC2 instance

aws configure

# Start the MLflow tracking server

# Replace the S3 bucket name if you used a different one

mlflow server \
 -h 0.0.0.0 \
 --default-artifact-root s3://firstmlflow1 \
 --allowed-hosts \*
The server runs on port 5000. Open your instance's Public IPv4 DNS at http://<your-ec2-dns>:5000 in a browser to verify the server is running. with go to instance and security and rules and modify rules and add tcp and port 5000

# 🫁 Chest Cancer Classification using VGG16

A deep learning pipeline for classifying chest CT scan images using transfer learning with VGG16. The project is structured into 4 sequential stages: data ingestion, base model preparation, training, and evaluation.

---

## 📁 Project Structure

```
Chest_project/
├── artifacts/
│   ├── data_ingestion/
│   │   ├── data.zip
│   │   └── Chest-CT-Scan-data/
│   ├── prepare_base_model/
│   │   ├── base_model.h5
│   │   └── base_model_updated.h5
│   └── training/
│       └── model.keras
├── logs/
│   └── running_logs.log
├── src/
│   └── cnnClassifier/
│       ├── logger.py
│       └── pipeline/
│           ├── stage_01_data_ingestion.py
│           ├── stage_02_prepare_base_model.py
│           ├── stage_03_model_trainer.py
│           └── stage_04_model_evaluation.py
├── scores.json
└── README.md
```

---

## ⚙️ Requirements

```bash
pip install tensorflow gdown
```

---

## 🚀 Pipeline Stages

### Stage 1 — Data Ingestion

Downloads the Chest CT Scan dataset from Google Drive and extracts it locally.

```python
SOURCE_URL      = "https://drive.google.com/file/d/1z0mreUtRmR-P-magILsDR3T7M6IkGXtY/view?usp=sharing"
LOCAL_DATA_FILE = "artifacts/data_ingestion/data.zip"
UNZIP_DIR       = "artifacts/data_ingestion/"
```

**Run:**

```bash
python -m src.cnnClassifier.pipeline.stage_01_data_ingestion
```

---

### Stage 2 — Prepare Base Model

Loads VGG16 pretrained on ImageNet, freezes all layers, adds a custom classification head, and saves the updated model.

```python
IMAGE_SIZE    = [224, 224, 3]
LEARNING_RATE = 0.01
INCLUDE_TOP   = False
WEIGHTS       = "imagenet"
CLASSES       = 2
```

**Run:**

```bash
python -m src.cnnClassifier.pipeline.stage_02_prepare_base_model
```

**Architecture:**

```
VGG16 (frozen) → Flatten → Dense(2, softmax)
Total params:     14,764,866
Trainable params:     50,178   (custom head only)
```

---

### Stage 3 — Model Training

Loads the updated base model, applies data augmentation, and trains on the Chest CT Scan dataset.

```python
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 16
EPOCHS        = 10
AUGMENTATION  = True
LEARNING_RATE = 0.01
```

**Augmentation applied:**

- Rotation: 40°
- Horizontal flip
- Width & height shift: 0.2
- Shear & zoom: 0.2

**Run:**

```bash
python -m src.cnnClassifier.pipeline.stage_03_model_trainer
```

Model saved as `artifacts/training/model.keras`

---

### Stage 4 — Model Evaluation

Evaluates the trained model on the validation split and saves scores to `scores.json`.

```python
TRAINED_MODEL_PATH = "artifacts/training/model.keras"
BATCH_SIZE         = 16
```

**Run:**

```bash
python -m src.cnnClassifier.pipeline.stage_04_model_evaluation
```

**Output (`scores.json`):**

```json
{
  "loss": 2.6105,
  "accuracy": 0.8088
}
```

---

## 🏃 Run All Stages

```bash
python -m src.cnnClassifier.pipeline.stage_01_data_ingestion
python -m src.cnnClassifier.pipeline.stage_02_prepare_base_model
python -m src.cnnClassifier.pipeline.stage_03_model_trainer
python -m src.cnnClassifier.pipeline.stage_04_model_evaluation
```

---

## 📊 Results

| Metric   | Value  |
| -------- | ------ |
| Loss     | 2.6105 |
| Accuracy | 80.88% |

---

## 📝 Logs

All stage logs are saved to:

```
logs/running_logs.log
```

Format:

```
[TIMESTAMP: LEVEL: MODULE: MESSAGE]
```

---

# Chest CT Scan Classifier — DVC + DagsHub Setup

This document explains how data and model versioning is configured for this project using **DVC** (Data Version Control) with **DagsHub** as the remote storage backend.

## Why DVC + DagsHub?

Git is great for code but terrible for large binary files like datasets and trained models — it bloats the repo and makes every clone painfully slow. DVC solves this by tracking large files separately, storing only small metadata pointers (`.dvc` files) in git, while the actual data lives in a remote storage backend.

DagsHub provides that remote storage for free, plus a web UI for visualizing your pipeline DAG, data, and experiments, and it integrates directly with your GitHub repo.

This project previously downloaded its dataset from a public Google Drive link via `gdown`. That approach is fragile for automated environments: Google Drive throttles and rate-limits repeated automated downloads, and large files trigger a virus-scan confirmation page that frequently breaks non-browser clients (CI runners, scheduled jobs, etc.). Moving the dataset to a DVC remote on DagsHub removes that dependency entirely.

## Prerequisites

- DVC installed (`pip install dvc`)
- A DagsHub account and repository: [dagshub.com/Mzn310/DVC_Pipeline](https://dagshub.com/Mzn310/DVC_Pipeline)
- A DagsHub personal access token (generate one under **Profile → Settings → Tokens**)

## One-Time Setup

### 1. Add the DagsHub DVC remote

DagsHub exposes a DVC-compatible endpoint at `<repo-url>.dvc`:

```bash
dvc remote add origin https://dagshub.com/Mzn310/DVC_Pipeline.dvc
dvc remote default origin
```

### 2. Configure authentication

DagsHub requires basic auth (username + token) for both push and pull on private repos:

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user Mzn310
dvc remote modify origin --local password <your-dagshub-token>
```

> **Why `--local`?** Anything set _without_ `--local` is written to `.dvc/config`, which is tracked by git and would leak into your repository. The `--local` flag writes to `.dvc/config.local` instead, which is gitignored by default — keeping your token out of version control.

### 3. Verify the remote

```bash
dvc remote list
```

Expected output:

```
origin    https://dagshub.com/Mzn310/DVC_Pipeline.dvc
```

## Daily Usage

### Pushing data/models to DagsHub

After running the pipeline and producing new DVC-tracked outputs:

```bash
dvc push -r origin
```

### Pulling data/models from DagsHub

On a fresh clone, or to sync the latest tracked data:

```bash
dvc pull -r origin
```

> **Note:** the DVC remote and DagsHub's "Storage Bucket" feature are separate systems. Data pushed via `dvc push` will not appear under the Storage Buckets section of the DagsHub UI — it's tracked through `.dvc` pointer files alongside your git history instead.

## CI/CD Integration

In automated environments (GitHub Actions, etc.), store your DagsHub username and token as repository secrets, then authenticate before pulling/pushing:

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DAGSHUB_USERNAME"
dvc remote modify origin --local password "$DAGSHUB_TOKEN"
dvc pull -r origin
```

Example GitHub Actions step:

```yaml
- name: Configure DVC remote
  run: |
    dvc remote modify origin --local auth basic
    dvc remote modify origin --local user "${{ secrets.DAGSHUB_USERNAME }}"
    dvc remote modify origin --local password "${{ secrets.DAGSHUB_TOKEN }}"

- name: Pull data
  run: dvc pull -r origin
```

## Pipeline Overview

The full pipeline is defined in `dvc.yaml` and consists of four stages:

| Stage                | Description                          | Key Outputs                                             |
| -------------------- | ------------------------------------ | ------------------------------------------------------- |
| `data_ingestion`     | Pulls the Chest CT Scan dataset      | `artifacts/data_ingestion/Chest-CT-Scan-data`           |
| `prepare_base_model` | Builds and adapts the base CNN model | `artifacts/prepare_base_model/base_model_updated.keras` |
| `model_trainer`      | Trains the classifier                | `artifacts/training/model.keras`                        |
| `model_evaluation`   | Evaluates the trained model          | `scores.json`                                           |

Run the full pipeline with:

```bash
dvc repro
```

Run a specific stage (and anything downstream that depends on it):

```bash
dvc repro model_trainer
```

Force a rerun of a stage even if DVC thinks nothing changed:

```bash
dvc repro --force model_trainer model_evaluation
```

## Troubleshooting

**`dvc add` fails with "overlaps with an output of stage"**
This means the path is already declared as a pipeline output in `dvc.yaml`. Don't `dvc add` it manually — let `dvc repro` manage it instead.

**Push/pull fails with an authentication error**
Confirm your token hasn't expired and that `dvc remote modify origin --local ...` was run with `--local` (check `.dvc/config.local` exists and contains your credentials, not `.dvc/config`).

**Pipeline reruns produce identical results**
DVC skips stages whose dependencies haven't changed. If you need to force a genuine retrain (e.g., after an accuracy check fails), use `dvc repro --force <stage>`.
