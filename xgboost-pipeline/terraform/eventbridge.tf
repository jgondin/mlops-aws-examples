# EventBridge configuration for scheduled retraining

# SNS Topic for training triggers
# This topic will be used to trigger GitHub Actions workflow
resource "aws_sns_topic" "training_trigger" {
  name = "${var.project_name}-${var.environment}-training-trigger"

  tags = {
    Name        = "Training Trigger Topic"
    Description = "SNS topic for triggering model retraining"
  }
}

# SNS Topic subscription (webhook endpoint)
# Note: You'll need to configure a webhook receiver in your GitHub Actions workflow
# or use AWS Lambda to trigger GitHub Actions via API
resource "aws_sns_topic_subscription" "training_trigger_webhook" {
  topic_arn = aws_sns_topic.training_trigger.arn
  protocol  = "https"
  endpoint  = "https://PLACEHOLDER-WEBHOOK-URL"  # Replace with actual webhook URL

  # Set to false initially - confirm subscription manually
  endpoint_auto_confirms = false

  # Add this lifecycle block to prevent creation until endpoint is configured
  lifecycle {
    ignore_changes = [endpoint]
  }
}

# EventBridge rule for scheduled retraining
resource "aws_cloudwatch_event_rule" "scheduled_retraining" {
  count = var.enable_eventbridge_trigger ? 1 : 0

  name                = "${var.project_name}-${var.environment}-scheduled-retraining"
  description         = "Trigger model retraining on schedule"
  schedule_expression = var.retraining_schedule

  tags = {
    Name = "Scheduled Retraining Rule"
  }
}

# EventBridge target: SNS topic
resource "aws_cloudwatch_event_target" "sns_target" {
  count = var.enable_eventbridge_trigger ? 1 : 0

  rule      = aws_cloudwatch_event_rule.scheduled_retraining[0].name
  target_id = "TriggerTraining"
  arn       = aws_sns_topic.training_trigger.arn

  input = jsonencode({
    trigger_type = "scheduled_retraining"
    timestamp    = "$${time}"
    project      = var.project_name
    environment  = var.environment
  })
}

# SNS Topic Policy to allow EventBridge to publish
resource "aws_sns_topic_policy" "training_trigger" {
  arn = aws_sns_topic.training_trigger.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
        Action   = "SNS:Publish"
        Resource = aws_sns_topic.training_trigger.arn
        Condition = {
          ArnEquals = {
            "aws:SourceArn" = var.enable_eventbridge_trigger ? aws_cloudwatch_event_rule.scheduled_retraining[0].arn : "*"
          }
        }
      }
    ]
  })
}

# Alternative: Lambda function to trigger GitHub Actions
# This Lambda function can be invoked by EventBridge to trigger GitHub workflow

resource "aws_lambda_function" "trigger_github_workflow" {
  count = var.enable_eventbridge_trigger ? 1 : 0

  filename      = "${path.module}/lambda_trigger.zip"  # You'll need to create this
  function_name = "${var.project_name}-${var.environment}-trigger-workflow"
  role          = aws_iam_role.lambda_trigger[0].arn
  handler       = "index.handler"
  runtime       = "python3.9"
  timeout       = 30

  environment {
    variables = {
      GITHUB_REPOSITORY = var.github_repository
      GITHUB_WORKFLOW   = "train.yml"
    }
  }

  # Create dummy zip file if it doesn't exist
  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }

  tags = {
    Name = "GitHub Workflow Trigger"
  }
}

# IAM role for Lambda function
resource "aws_iam_role" "lambda_trigger" {
  count = var.enable_eventbridge_trigger ? 1 : 0

  name = "${var.project_name}-${var.environment}-lambda-trigger-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "Lambda Trigger Role"
  }
}

# Lambda basic execution policy
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  count = var.enable_eventbridge_trigger ? 1 : 0

  role       = aws_iam_role.lambda_trigger[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Lambda policy for Secrets Manager (to access GitHub token)
resource "aws_iam_role_policy" "lambda_secrets" {
  count = var.enable_eventbridge_trigger ? 1 : 0

  name = "secrets-access"
  role = aws_iam_role.lambda_trigger[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:github-token-*"
      }
    ]
  })
}

# EventBridge target: Lambda function
resource "aws_cloudwatch_event_target" "lambda_target" {
  count = var.enable_eventbridge_trigger ? 1 : 0

  rule      = aws_cloudwatch_event_rule.scheduled_retraining[0].name
  target_id = "TriggerGitHubWorkflow"
  arn       = aws_lambda_function.trigger_github_workflow[0].arn
}

# Lambda permission for EventBridge
resource "aws_lambda_permission" "allow_eventbridge" {
  count = var.enable_eventbridge_trigger ? 1 : 0

  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.trigger_github_workflow[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.scheduled_retraining[0].arn
}
