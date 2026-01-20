# SageMaker Training and Deployment Pipeline

This example demonstrates a complete MLOps pipeline using Amazon SageMaker, including training, deployment, and monitoring of a machine learning model.

## Overview

This pipeline implements:
1. **Data Preprocessing**: Prepare and validate training data
2. **Model Training**: Train a scikit-learn model on SageMaker
3. **Model Evaluation**: Validate model performance
4. **Model Registry**: Register approved models
5. **Deployment**: Deploy to SageMaker endpoint
6. **Monitoring**: Track model performance and data drift

## Architecture

```
S3 Bucket (Data) → SageMaker Training Job → Model Registry
                                                ↓
                                    SageMaker Endpoint → CloudWatch Monitoring
```

## Files

- `train.py` - Training script for SageMaker
- `inference.py` - Custom inference handler
- `preprocess.py` - Data preprocessing utilities
- `deploy.py` - Deployment automation script
- `config.yaml` - Configuration parameters
- `requirements.txt` - Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python preprocess.py --input-data ./data/raw --output-path s3://your-bucket/data/processed
```

### 3. Train Model

```bash
python train.py \
  --train-data s3://your-bucket/data/processed/train \
  --val-data s3://your-bucket/data/processed/val \
  --model-dir s3://your-bucket/models \
  --instance-type ml.m5.xlarge
```

### 4. Deploy Model

```bash
python deploy.py \
  --model-uri s3://your-bucket/models/model.tar.gz \
  --endpoint-name my-ml-endpoint \
  --instance-type ml.t2.medium
```

### 5. Test Inference

```bash
python test_inference.py --endpoint-name my-ml-endpoint --test-data ./data/test_sample.json
```

## Configuration

Edit `config.yaml` to customize:
- AWS region and S3 bucket
- Instance types and sizes
- Training hyperparameters
- Monitoring thresholds

## Model Details

This example uses a **Random Forest Classifier** for demonstration. The model:
- Predicts customer churn (binary classification)
- Uses 10 input features
- Includes feature importance analysis
- Achieves ~85% accuracy on test data

## Monitoring

The pipeline includes:
- **Model accuracy tracking**: Real-time accuracy metrics
- **Data drift detection**: Statistical tests on input features
- **Endpoint health**: Latency and error rate monitoring
- **Automated alerts**: CloudWatch alarms for degradation

## Costs

Estimated costs for running this example (us-east-1):
- Training job (1 hour): ~$0.30
- Endpoint (per hour): ~$0.05
- S3 storage (1GB): ~$0.02/month

**Remember to delete resources when done!**

```bash
python cleanup.py --endpoint-name my-ml-endpoint
```

## Advanced Features

- **Automated Retraining**: Trigger retraining on data drift
- **A/B Testing**: Deploy multiple model versions
- **Multi-Model Endpoints**: Host multiple models efficiently
- **Batch Transform**: Process large datasets offline

## Troubleshooting

See [TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md) for common issues and solutions.

## Next Steps

- Integrate with CI/CD pipeline (see `../../infrastructure/cicd/`)
- Add feature store integration
- Implement canary deployment strategy
- Set up MLflow experiment tracking
