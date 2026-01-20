# S3 buckets for MLOps pipeline

# Bucket for raw data
resource "aws_s3_bucket" "raw_data" {
  bucket = "${var.project_name}-${var.environment}-raw-data-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "Raw Data Bucket"
    Description = "Storage for raw input data"
  }
}

# Enable versioning for raw data
resource "aws_s3_bucket_versioning" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Block public access for raw data
resource "aws_s3_bucket_public_access_block" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bucket for processed features
resource "aws_s3_bucket" "processed_features" {
  bucket = "${var.project_name}-${var.environment}-features-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "Processed Features Bucket"
    Description = "Storage for processed training/test data"
  }
}

resource "aws_s3_bucket_versioning" "processed_features" {
  bucket = aws_s3_bucket.processed_features.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "processed_features" {
  bucket = aws_s3_bucket.processed_features.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bucket for model artifacts
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.project_name}-${var.environment}-models-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "Model Artifacts Bucket"
    Description = "Storage for trained models"
  }
}

resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle policy for model artifacts (optional)
resource "aws_s3_bucket_lifecycle_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  rule {
    id     = "archive-old-models"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 180
      storage_class = "GLACIER"
    }
  }
}

# Bucket for predictions output
resource "aws_s3_bucket" "predictions" {
  bucket = "${var.project_name}-${var.environment}-predictions-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "Predictions Output Bucket"
    Description = "Storage for model predictions"
  }
}

resource "aws_s3_bucket_versioning" "predictions" {
  bucket = aws_s3_bucket.predictions.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "predictions" {
  bucket = aws_s3_bucket.predictions.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle policy for predictions (clean up old predictions)
resource "aws_s3_bucket_lifecycle_configuration" "predictions" {
  bucket = aws_s3_bucket.predictions.id

  rule {
    id     = "delete-old-predictions"
    status = "Enabled"

    expiration {
      days = 30  # Delete predictions after 30 days
    }
  }
}
