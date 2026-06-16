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
