# Terraform outputs for XGBoost MLOps pipeline

# S3 Buckets
output "raw_data_bucket" {
  description = "S3 bucket name for raw data"
  value       = aws_s3_bucket.raw_data.id
}

output "processed_features_bucket" {
  description = "S3 bucket name for processed features"
  value       = aws_s3_bucket.processed_features.id
}

output "model_artifacts_bucket" {
  description = "S3 bucket name for model artifacts"
  value       = aws_s3_bucket.model_artifacts.id
}

output "predictions_bucket" {
  description = "S3 bucket name for predictions"
  value       = aws_s3_bucket.predictions.id
}

# IAM Roles
output "sagemaker_execution_role_arn" {
  description = "ARN of SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution.arn
}

output "github_actions_role_arn" {
  description = "ARN of GitHub Actions OIDC role"
  value       = aws_iam_role.github_actions.arn
}

output "eventbridge_role_arn" {
  description = "ARN of EventBridge execution role"
  value       = aws_iam_role.eventbridge.arn
}

# SNS Topics
output "training_trigger_topic_arn" {
  description = "ARN of SNS topic for training triggers"
  value       = aws_sns_topic.training_trigger.arn
}

# CloudWatch
output "cloudwatch_dashboard_name" {
  description = "Name of CloudWatch dashboard"
  value       = aws_cloudwatch_dashboard.mlops.dashboard_name
}

# SSM Parameters (for easy access in scripts)
output "ssm_parameter_prefix" {
  description = "Prefix for SSM parameters"
  value       = "/${var.project_name}/${var.environment}"
}

# EventBridge
output "retraining_schedule_rule" {
  description = "EventBridge rule name for scheduled retraining"
  value       = var.enable_eventbridge_trigger ? aws_cloudwatch_event_rule.scheduled_retraining[0].name : "disabled"
}

# Configuration summary for GitHub Actions
output "github_actions_config" {
  description = "Configuration values for GitHub Actions workflows"
  value = {
    aws_region                 = var.aws_region
    github_actions_role_arn    = aws_iam_role.github_actions.arn
    sagemaker_execution_role   = aws_iam_role.sagemaker_execution.arn
    features_bucket            = aws_s3_bucket.processed_features.id
    models_bucket              = aws_s3_bucket.model_artifacts.id
    predictions_bucket         = aws_s3_bucket.predictions.id
    training_instance_type     = var.training_instance_type
    transform_instance_type    = var.transform_instance_type
  }
  sensitive = false
}

# Instructions
output "setup_instructions" {
  description = "Next steps for setup"
  value = <<-EOT

    ========================================
    XGBoost MLOps Pipeline - Setup Complete
    ========================================

    Next Steps:

    1. Configure GitHub Actions Secrets:
       - AWS_REGION: ${var.aws_region}
       - AWS_ROLE_ARN: ${aws_iam_role.github_actions.arn}
       - SAGEMAKER_ROLE_ARN: ${aws_iam_role.sagemaker_execution.arn}
       - FEATURES_BUCKET: ${aws_s3_bucket.processed_features.id}
       - MODELS_BUCKET: ${aws_s3_bucket.model_artifacts.id}
       - PREDICTIONS_BUCKET: ${aws_s3_bucket.predictions.id}

    2. View CloudWatch Dashboard:
       https://console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.mlops.dashboard_name}

    3. Monitor S3 Buckets:
       - Raw Data: s3://${aws_s3_bucket.raw_data.id}
       - Features: s3://${aws_s3_bucket.processed_features.id}
       - Models: s3://${aws_s3_bucket.model_artifacts.id}
       - Predictions: s3://${aws_s3_bucket.predictions.id}

    4. Test the pipeline:
       - Manually trigger GitHub Actions workflow 'train.yml'
       - Monitor progress in CloudWatch Logs and Dashboard

    ========================================

  EOT
}
