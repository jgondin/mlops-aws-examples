# XGBoost MLOps Pipeline on AWS

A complete, production-ready MLOps pipeline for training and deploying XGBoost binary classification models on AWS using SageMaker, with GitHub Actions for CI/CD and Terraform for infrastructure as code.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Cost Estimation](#cost-estimation)
- [Contributing](#contributing)

## 🎯 Overview

This project demonstrates a complete MLOps pipeline for binary classification using:

- **Dataset**: Titanic survival prediction (from sklearn)
- **Model**: XGBoost using SageMaker built-in container
- **Training**: Offline training on SageMaker
- **Deployment**: Batch inference using SageMaker Batch Transform
- **Orchestration**: GitHub Actions for CI/CD
- **Scheduling**: AWS EventBridge for periodic retraining
- **Infrastructure**: Terraform for IaC
- **Region**: us-east-1

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Actions CI/CD                     │
│  ┌──────────────┐                   ┌──────────────┐        │
│  │ Train        │ ──── Success ───▶ │ Deploy       │        │
│  │ Workflow     │                   │ Workflow     │        │
│  └──────────────┘                   └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
           │                                    │
           ▼                                    ▼
┌─────────────────────┐              ┌─────────────────────┐
│  Feature Pipeline   │              │ Inference Pipeline  │
│  - Load Titanic     │              │ - Load Model        │
│  - Preprocess       │              │ - Batch Transform   │
│  - Save to S3       │              │ - Save Predictions  │
└─────────────────────┘              └─────────────────────┘
           │                                    │
           ▼                                    ▼
┌─────────────────────┐              ┌─────────────────────┐
│  Training Pipeline  │              │  S3 Predictions     │
│  - Train XGBoost    │              │  Bucket             │
│  - Evaluate F1      │              └─────────────────────┘
│  - Save Model       │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  CloudWatch         │
│  - Metrics          │
│  - Alarms           │
│  - Dashboard        │
└─────────────────────┘

Scheduled Retraining:
┌──────────────────┐
│  EventBridge     │ ──▶ SNS Topic ──▶ GitHub Actions
│  (Weekly)        │
└──────────────────┘
```

## ✨ Features

- ✅ **Automated Feature Engineering**: Load and preprocess Titanic dataset
- ✅ **SageMaker Training**: Distributed training with built-in XGBoost
- ✅ **Model Evaluation**: F1-score threshold validation (>0.75)
- ✅ **Batch Inference**: SageMaker Batch Transform for predictions
- ✅ **CI/CD Pipeline**: Automated training and deployment via GitHub Actions
- ✅ **Infrastructure as Code**: Complete Terraform configuration
- ✅ **Scheduled Retraining**: Weekly automated retraining with EventBridge
- ✅ **Monitoring**: CloudWatch metrics, alarms, and dashboard
- ✅ **OIDC Authentication**: Secure GitHub Actions → AWS authentication
- ✅ **Model Registry**: S3-based model versioning and metadata

## 📋 Prerequisites

### AWS Account Requirements

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Terraform >= 1.0 installed

### GitHub Requirements

- GitHub repository
- GitHub Actions enabled
- Permissions to configure repository secrets

### Local Development (Optional)

- Python 3.9+
- pip or conda for package management

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/mlops-aws-examples.git
cd mlops-aws-examples/xgboost-pipeline
```

### 2. Deploy Infrastructure

```bash
cd terraform

# Initialize Terraform
terraform init

# Review plan
terraform plan \
  -var="github_repository=YOUR-USERNAME/mlops-aws-examples" \
  -var="project_name=xgboost-mlops" \
  -var="environment=dev"

# Apply configuration
terraform apply \
  -var="github_repository=YOUR-USERNAME/mlops-aws-examples" \
  -var="project_name=xgboost-mlops" \
  -var="environment=dev"
```

### 3. Configure GitHub Secrets

After Terraform completes, configure the following GitHub repository secrets:

```bash
# Navigate to: Settings → Secrets and variables → Actions → New repository secret

AWS_REGION=us-east-1
AWS_ROLE_ARN=<from terraform output: github_actions_role_arn>
SAGEMAKER_ROLE_ARN=<from terraform output: sagemaker_execution_role_arn>
FEATURES_BUCKET=<from terraform output: processed_features_bucket>
MODELS_BUCKET=<from terraform output: model_artifacts_bucket>
PREDICTIONS_BUCKET=<from terraform output: predictions_bucket>
```

### 4. Trigger Training Pipeline

```bash
# Go to Actions tab in GitHub
# Select "Train XGBoost Model" workflow
# Click "Run workflow"
# Wait for completion (~15-20 minutes)
```

## 📚 Detailed Setup

### Step 1: AWS Infrastructure Setup

#### 1.1 Configure AWS CLI

```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter default region: us-east-1
# Enter default output format: json
```

#### 1.2 Create Terraform Backend (Optional)

For production use, configure remote state:

```bash
# Create S3 bucket for Terraform state
aws s3 mb s3://YOUR-TERRAFORM-STATE-BUCKET --region us-east-1

# Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# Uncomment backend configuration in terraform/main.tf
```

#### 1.3 Customize Terraform Variables

Create `terraform/terraform.tfvars`:

```hcl
aws_region         = "us-east-1"
project_name       = "xgboost-mlops"
environment        = "dev"
github_repository  = "YOUR-USERNAME/mlops-aws-examples"

training_instance_type  = "ml.m5.xlarge"
transform_instance_type = "ml.m5.large"

retraining_schedule        = "rate(7 days)"
enable_eventbridge_trigger = true
```

#### 1.4 Deploy Infrastructure

```bash
cd terraform

terraform init
terraform validate
terraform plan
terraform apply -auto-approve

# Save outputs
terraform output -json > ../terraform-outputs.json
```

### Step 2: GitHub Configuration

#### 2.1 Enable GitHub Actions

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Actions** → **General**
3. Ensure "Allow all actions and reusable workflows" is enabled

#### 2.2 Configure Repository Secrets

Navigate to **Settings** → **Secrets and variables** → **Actions**

Add the following secrets:

| Secret Name | Value | Source |
|-------------|-------|--------|
| `AWS_REGION` | `us-east-1` | Your AWS region |
| `AWS_ROLE_ARN` | `arn:aws:iam::...` | Terraform output: `github_actions_role_arn` |
| `SAGEMAKER_ROLE_ARN` | `arn:aws:iam::...` | Terraform output: `sagemaker_execution_role_arn` |
| `FEATURES_BUCKET` | `xgboost-mlops-dev-features-...` | Terraform output: `processed_features_bucket` |
| `MODELS_BUCKET` | `xgboost-mlops-dev-models-...` | Terraform output: `model_artifacts_bucket` |
| `PREDICTIONS_BUCKET` | `xgboost-mlops-dev-predictions-...` | Terraform output: `predictions_bucket` |

#### 2.3 Configure GitHub OIDC (Automatic)

The Terraform configuration automatically creates the OIDC provider and IAM role for GitHub Actions.

### Step 3: EventBridge Setup (Optional)

#### 3.1 Configure GitHub Webhook for EventBridge

To enable EventBridge to trigger GitHub Actions:

**Option A: Use Repository Dispatch** (Recommended)

1. Create a GitHub Personal Access Token (PAT)
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate new token with `repo` scope
   - Save token in AWS Secrets Manager

2. Store PAT in Secrets Manager:
```bash
aws secretsmanager create-secret \
  --name github-token \
  --secret-string '{"token":"YOUR-GITHUB-PAT"}' \
  --region us-east-1
```

3. The Lambda function will use this to trigger workflows

**Option B: Use GitHub Webhook** (Alternative)

Configure a webhook endpoint in `.github/workflows/` to receive SNS notifications.

## 💻 Usage

### Running the Pipeline

#### Manual Training

```bash
# Via GitHub Actions UI
1. Go to "Actions" tab
2. Select "Train XGBoost Model"
3. Click "Run workflow"
4. (Optional) Configure parameters
5. Click "Run workflow"
```

#### Automated Scheduled Training

The pipeline automatically runs weekly (configurable in Terraform):

- Default schedule: Every Monday at 2 AM UTC
- Configurable via EventBridge rule
- Triggers GitHub Actions workflow automatically

#### Manual Deployment

```bash
# Via GitHub Actions UI
1. Go to "Actions" tab
2. Select "Deploy XGBoost Model"
3. Click "Run workflow"
4. Enter model path and job name
5. Click "Run workflow"
```

### Local Development and Testing

#### Test Feature Pipeline Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run feature pipeline
python src/feature_pipeline.py \
  --s3-bucket YOUR-FEATURES-BUCKET \
  --s3-prefix processed-data
```

#### Test Training Pipeline Locally

```bash
python src/training_pipeline.py \
  --s3-bucket YOUR-MODELS-BUCKET \
  --role-arn YOUR-SAGEMAKER-ROLE-ARN \
  --train-data s3://YOUR-BUCKET/processed-data/train.csv \
  --test-data s3://YOUR-BUCKET/processed-data/test.csv \
  --region us-east-1
```

#### Test Inference Pipeline Locally

```bash
python src/inference_pipeline.py \
  --model-path s3://YOUR-BUCKET/model.tar.gz \
  --input-data s3://YOUR-BUCKET/test_xgb.csv \
  --output-path s3://YOUR-BUCKET/predictions/ \
  --role-arn YOUR-SAGEMAKER-ROLE-ARN \
  --image-uri <xgboost-container-uri> \
  --region us-east-1
```

## 📁 Project Structure

```
xgboost-pipeline/
├── src/
│   ├── feature_pipeline.py      # Data preprocessing and feature engineering
│   ├── training_pipeline.py     # XGBoost model training
│   └── inference_pipeline.py    # Batch inference
├── terraform/
│   ├── main.tf                  # Provider and backend configuration
│   ├── variables.tf             # Input variables
│   ├── outputs.tf               # Output values
│   ├── iam.tf                   # IAM roles and policies
│   ├── s3.tf                    # S3 buckets
│   ├── sagemaker.tf             # SageMaker configuration
│   └── eventbridge.tf           # EventBridge rules and targets
├── .github/
│   └── workflows/
│       ├── train.yml            # Training pipeline workflow
│       └── deploy.yml           # Deployment pipeline workflow
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## ⚙️ Configuration

### Hyperparameters

Default XGBoost hyperparameters (configurable in `train.yml`):

```yaml
max_depth: 5
eta: 0.2
num_round: 100
objective: binary:logistic
eval_metric: auc
```

### F1-Score Threshold

Models must achieve F1-score > 0.75 to be deployed. Configure in `train.yml`:

```yaml
THRESHOLD=0.75
```

### Instance Types

Configure in `terraform/variables.tf`:

```hcl
training_instance_type  = "ml.m5.xlarge"   # ~$0.23/hour
transform_instance_type = "ml.m5.large"    # ~$0.115/hour
```

### Retraining Schedule

Configure in `terraform/variables.tf`:

```hcl
retraining_schedule = "rate(7 days)"  # Weekly
# OR
retraining_schedule = "cron(0 2 * * MON)"  # Monday 2 AM UTC
```

## 📊 Monitoring

### CloudWatch Dashboard

View the MLOps dashboard:

```bash
# Get dashboard URL from Terraform output
terraform output cloudwatch_dashboard_name

# Or navigate to:
https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:
```

Dashboard includes:
- F1-Score trends
- Accuracy trends
- Training job failures
- Transform job failures

### CloudWatch Alarms

Configured alarms:
- Training job failures
- Low F1-score (<0.75)
- Batch transform failures

### Logs

View logs in CloudWatch:

```bash
# Training logs
/aws/sagemaker/TrainingJobs/xgboost-mlops-dev

# Transform logs
/aws/sagemaker/TransformJobs/xgboost-mlops-dev
```

### S3 Monitoring

Monitor S3 buckets:

```bash
# List processed data
aws s3 ls s3://YOUR-FEATURES-BUCKET/processed-data/

# List model artifacts
aws s3 ls s3://YOUR-MODELS-BUCKET/model-artifacts/

# List predictions
aws s3 ls s3://YOUR-PREDICTIONS-BUCKET/predictions/
```

## 🔧 Troubleshooting

### Common Issues

#### 1. GitHub Actions Authentication Fails

**Error**: `Error: Not authorized to perform sts:AssumeRoleWithWebIdentity`

**Solution**:
- Verify OIDC provider is created in IAM
- Check GitHub repository name matches exactly in IAM trust policy
- Ensure `id-token: write` permission is set in workflow

#### 2. SageMaker Training Job Fails

**Error**: `ResourceNotFound` or permission errors

**Solution**:
- Verify SageMaker execution role has S3 permissions
- Check S3 bucket names are correct
- Ensure training data exists in S3
- Review CloudWatch logs for detailed error

#### 3. F1-Score Below Threshold

**Error**: `F1-score below threshold (0.75)`

**Solution**:
- Tune hyperparameters in `train.yml`
- Increase `num_round` (more trees)
- Adjust `max_depth` and `eta`
- Check data quality and preprocessing

#### 4. Terraform Apply Fails

**Error**: Resource already exists

**Solution**:
```bash
# Import existing resources
terraform import aws_s3_bucket.raw_data YOUR-BUCKET-NAME

# Or destroy and recreate
terraform destroy
terraform apply
```

### Debug Mode

Enable verbose logging:

```bash
# In GitHub Actions workflow, add:
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true

# For local Python scripts, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

1. Check GitHub Actions logs
2. Review CloudWatch logs
3. Inspect S3 buckets for data
4. Verify IAM permissions
5. Test pipelines locally

## 💰 Cost Estimation

### Monthly Costs (Approximate)

Based on weekly retraining and moderate usage:

| Service | Usage | Cost/Month |
|---------|-------|------------|
| **SageMaker Training** | 4 jobs × 1 hour × ml.m5.xlarge | ~$4 |
| **SageMaker Batch Transform** | 4 jobs × 0.5 hour × ml.m5.large | ~$1 |
| **S3 Storage** | 5 GB | ~$0.12 |
| **CloudWatch** | Logs + Metrics | ~$1 |
| **EventBridge** | 4 invocations | Free |
| **SNS** | 4 notifications | Free |
| **Data Transfer** | Minimal | ~$0.50 |
| **Total** | | **~$6.62/month** |

### Cost Optimization Tips

1. **Use Spot Instances**: Configure managed spot training (60-90% savings)
2. **Reduce Frequency**: Change retraining from weekly to monthly
3. **Smaller Instances**: Use ml.t3.medium for small datasets
4. **S3 Lifecycle**: Archive old models to Glacier
5. **Log Retention**: Reduce CloudWatch log retention to 7 days

### Clean Up Resources

To avoid ongoing charges:

```bash
# Destroy all infrastructure
cd terraform
terraform destroy -auto-approve

# Or manually delete:
# - S3 buckets (empty first)
# - CloudWatch log groups
# - SageMaker models and endpoints
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- [AWS SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Titanic Dataset](https://www.openml.org/d/40945)

## 📞 Support

For issues and questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/YOUR-USERNAME/mlops-aws-examples/issues)
3. Open a new issue with detailed description

---

**Built with ❤️ for the MLOps community**
