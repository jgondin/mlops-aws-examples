# SageMaker configuration for XGBoost MLOps pipeline

# CloudWatch Log Group for SageMaker
resource "aws_cloudwatch_log_group" "sagemaker_training" {
  name              = "/aws/sagemaker/TrainingJobs/${var.project_name}-${var.environment}"
  retention_in_days = 30

  tags = {
    Name = "SageMaker Training Logs"
  }
}

resource "aws_cloudwatch_log_group" "sagemaker_transform" {
  name              = "/aws/sagemaker/TransformJobs/${var.project_name}-${var.environment}"
  retention_in_days = 30

  tags = {
    Name = "SageMaker Transform Logs"
  }
}

# CloudWatch metric alarms for monitoring

# Alarm for training job failures
resource "aws_cloudwatch_metric_alarm" "training_job_failed" {
  alarm_name          = "${var.project_name}-${var.environment}-training-job-failed"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "TrainingJobsFailed"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Alert when SageMaker training job fails"
  treat_missing_data  = "notBreaching"

  dimensions = {
    TrainingJobName = "${var.project_name}-${var.environment}"
  }

  tags = {
    Name = "Training Job Failure Alarm"
  }
}

# Alarm for low F1-score
resource "aws_cloudwatch_metric_alarm" "low_f1_score" {
  alarm_name          = "${var.project_name}-${var.environment}-low-f1-score"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "F1Score"
  namespace           = "MLOps/XGBoost"
  period              = 300
  statistic           = "Average"
  threshold           = 0.75
  alarm_description   = "Alert when model F1-score is below threshold"
  treat_missing_data  = "notBreaching"

  tags = {
    Name = "Low F1-Score Alarm"
  }
}

# Alarm for batch transform failures
resource "aws_cloudwatch_metric_alarm" "transform_job_failed" {
  alarm_name          = "${var.project_name}-${var.environment}-transform-job-failed"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "TransformJobsFailed"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Alert when SageMaker batch transform job fails"
  treat_missing_data  = "notBreaching"

  tags = {
    Name = "Transform Job Failure Alarm"
  }
}

# SSM Parameters for pipeline configuration
resource "aws_ssm_parameter" "sagemaker_role_arn" {
  name        = "/${var.project_name}/${var.environment}/sagemaker/role-arn"
  description = "SageMaker execution role ARN"
  type        = "String"
  value       = aws_iam_role.sagemaker_execution.arn

  tags = {
    Name = "SageMaker Role ARN"
  }
}

resource "aws_ssm_parameter" "features_bucket" {
  name        = "/${var.project_name}/${var.environment}/s3/features-bucket"
  description = "S3 bucket for processed features"
  type        = "String"
  value       = aws_s3_bucket.processed_features.id

  tags = {
    Name = "Features Bucket Name"
  }
}

resource "aws_ssm_parameter" "models_bucket" {
  name        = "/${var.project_name}/${var.environment}/s3/models-bucket"
  description = "S3 bucket for model artifacts"
  type        = "String"
  value       = aws_s3_bucket.model_artifacts.id

  tags = {
    Name = "Models Bucket Name"
  }
}

resource "aws_ssm_parameter" "predictions_bucket" {
  name        = "/${var.project_name}/${var.environment}/s3/predictions-bucket"
  description = "S3 bucket for predictions"
  type        = "String"
  value       = aws_s3_bucket.predictions.id

  tags = {
    Name = "Predictions Bucket Name"
  }
}

resource "aws_ssm_parameter" "training_instance_type" {
  name        = "/${var.project_name}/${var.environment}/sagemaker/training-instance-type"
  description = "Instance type for SageMaker training"
  type        = "String"
  value       = var.training_instance_type

  tags = {
    Name = "Training Instance Type"
  }
}

resource "aws_ssm_parameter" "transform_instance_type" {
  name        = "/${var.project_name}/${var.environment}/sagemaker/transform-instance-type"
  description = "Instance type for SageMaker batch transform"
  type        = "String"
  value       = var.transform_instance_type

  tags = {
    Name = "Transform Instance Type"
  }
}

# Dashboard for monitoring (optional)
resource "aws_cloudwatch_dashboard" "mlops" {
  dashboard_name = "${var.project_name}-${var.environment}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["MLOps/XGBoost", "F1Score", { stat = "Average" }],
            [".", "Accuracy", { stat = "Average" }]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "Model Performance Metrics"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/SageMaker", "TrainingJobsFailed", { stat = "Sum" }],
            [".", "TransformJobsFailed", { stat = "Sum" }]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          title  = "Job Failures"
        }
      }
    ]
  })
}
