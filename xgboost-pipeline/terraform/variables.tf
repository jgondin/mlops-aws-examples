# Terraform variables for XGBoost MLOps pipeline

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "xgboost-mlops"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "github_repository" {
  description = "GitHub repository in format 'owner/repo'"
  type        = string
  # Example: "yourusername/mlops-aws-examples"
}

variable "github_oidc_thumbprint" {
  description = "GitHub OIDC provider thumbprint"
  type        = string
  default     = "6938fd4d98bab03faadb97b34396831e3780aea1"
}

variable "training_instance_type" {
  description = "SageMaker instance type for training"
  type        = string
  default     = "ml.m5.xlarge"
}

variable "transform_instance_type" {
  description = "SageMaker instance type for batch transform"
  type        = string
  default     = "ml.m5.large"
}

variable "retraining_schedule" {
  description = "EventBridge schedule expression for retraining (e.g., rate(7 days))"
  type        = string
  default     = "rate(7 days)"
}

variable "enable_eventbridge_trigger" {
  description = "Enable EventBridge scheduled retraining"
  type        = bool
  default     = true
}
