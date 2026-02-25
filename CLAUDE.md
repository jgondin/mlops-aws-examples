# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Configure AWS credentials before running any pipelines:
```bash
aws configure
```

## Common Commands

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Linting and formatting
flake8 src/ examples/
black --check src/ examples/
mypy src/
isort <files>
```

## Infrastructure (Terraform)

```bash
cd xgboost-pipeline/terraform
terraform init
terraform plan -var="github_repository=YOUR-USERNAME/mlops-aws-examples" \
               -var="project_name=xgboost-mlops" \
               -var="environment=dev"
terraform apply -auto-approve
terraform destroy  # clean up
```

## Architecture

This repo provides end-to-end MLOps examples on AWS SageMaker. The primary example is the **XGBoost pipeline** (`xgboost-pipeline/`) which demonstrates a 3-stage pipeline for Titanic survival prediction (binary classification).

### Pipeline Stages (`xgboost-pipeline/src/`)

1. **`feature_pipeline.py`** — Loads Titanic dataset from OpenML, preprocesses features (imputation, label encoding), splits 80/20, and saves `train.csv`/`test.csv`/`metadata.json` to S3.
2. **`training_pipeline.py`** — Launches a SageMaker training job with XGBoost container, evaluates on test set, and logs metrics to CloudWatch. F1-score must exceed **0.75** to allow deployment.
3. **`inference_pipeline.py`** — Creates a SageMaker Batch Transform job for predictions and validates output.

### CI/CD (`xgboost-pipeline/.github/workflows/`)

- **`train.yml`**: Runs feature engineering → training → evaluation. Triggered manually, on a weekly schedule (Monday 2 AM UTC), or via EventBridge. Calls `deploy.yml` on success.
- **`deploy.yml`**: Runs batch inference and creates a release tag. Can also be triggered manually.

AWS authentication uses **OIDC** (no static credentials). GitHub Actions OIDC role and all infrastructure are provisioned via Terraform.

### Terraform (`xgboost-pipeline/terraform/`)

Provisions all AWS resources: 4 S3 buckets (raw_data, processed_features, model_artifacts, predictions), IAM roles (SageMaker, GitHub Actions OIDC, EventBridge), CloudWatch dashboard/alarms, EventBridge rule with Lambda for scheduled retraining, SNS topic, and SageMaker model package group.

### Secondary Example (`examples/sagemaker-pipeline/`)

A simpler SageMaker training + deployment example with endpoint auto-scaling, data capture, and monitoring config defined in `config.yaml`.

## Key Configuration

- **AWS Region**: `us-east-1` (default in `variables.tf` and workflows)
- **Training instance**: `ml.m5.xlarge`
- **Inference instance**: `ml.m5.large`
- **XGBoost hyperparameters**: `max_depth=5`, `eta=0.2`, `num_round=100`, `eval_metric=auc`
- **Retraining schedule**: Every 7 days via EventBridge
