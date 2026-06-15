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
