#!/usr/bin/env python3
"""
Inference Pipeline: Run batch predictions using SageMaker Batch Transform.

This script loads a trained model and runs batch inference on new data.
"""
import argparse
import json
import logging
import time
from datetime import datetime

import boto3
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InferencePipeline:
    """Handle batch inference using SageMaker Batch Transform."""

    def __init__(self, region='us-east-1'):
        """
        Initialize the inference pipeline.

        Args:
            region: AWS region
        """
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)

        logger.info(f"Initialized InferencePipeline in region: {region}")

    def create_model(self, model_name, model_s3_path, role_arn, image_uri):
        """
        Create a SageMaker model from trained artifacts.

        Args:
            model_name: Name for the model
            model_s3_path: S3 path to model.tar.gz
            role_arn: SageMaker execution role ARN
            image_uri: Container image URI

        Returns:
            Model ARN
        """
        logger.info(f"Creating SageMaker model: {model_name}")

        try:
            response = self.sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': image_uri,
                    'ModelDataUrl': model_s3_path,
                    'Mode': 'SingleModel'
                },
                ExecutionRoleArn=role_arn
            )

            model_arn = response['ModelArn']
            logger.info(f"Model created: {model_arn}")
            return model_arn

        except self.sagemaker_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                logger.info(f"Model {model_name} already exists, using existing model")
                return f"arn:aws:sagemaker:{self.region}::model/{model_name}"
            raise

    def create_batch_transform_job(
        self,
        job_name,
        model_name,
        input_s3_path,
        output_s3_path,
        instance_type='ml.m5.large',
        instance_count=1
    ):
        """
        Create and run a SageMaker Batch Transform job.

        Args:
            job_name: Unique name for the transform job
            model_name: Name of the model to use
            input_s3_path: S3 path to input data
            output_s3_path: S3 path for predictions output
            instance_type: EC2 instance type
            instance_count: Number of instances

        Returns:
            Transform job name
        """
        logger.info(f"Creating batch transform job: {job_name}")

        # Parse input path to get bucket and prefix
        input_path_parts = input_s3_path.replace('s3://', '').split('/')
        input_bucket = input_path_parts[0]
        input_key = '/'.join(input_path_parts[1:])

        # Determine if input is a file or directory
        # For batch transform, we need to specify the S3 URI correctly
        if input_key.endswith('.csv'):
            # Single file
            data_source = input_s3_path
        else:
            # Directory
            data_source = input_s3_path

        response = self.sagemaker_client.create_transform_job(
            TransformJobName=job_name,
            ModelName=model_name,
            MaxConcurrentTransforms=1,
            MaxPayloadInMB=6,
            BatchStrategy='MultiRecord',
            TransformInput={
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': data_source
                    }
                },
                'ContentType': 'text/csv',
                'SplitType': 'Line',
                'CompressionType': 'None'
            },
            TransformOutput={
                'S3OutputPath': output_s3_path,
                'AssembleWith': 'Line',
                'Accept': 'text/csv'
            },
            TransformResources={
                'InstanceType': instance_type,
                'InstanceCount': instance_count
            }
        )

        logger.info(f"Batch transform job started: {job_name}")
        return job_name

    def wait_for_transform_job(self, job_name, poll_interval=30):
        """
        Wait for batch transform job to complete.

        Args:
            job_name: Transform job name
            poll_interval: Seconds between status checks

        Returns:
            Transform job details
        """
        logger.info(f"Waiting for transform job to complete: {job_name}")

        while True:
            response = self.sagemaker_client.describe_transform_job(
                TransformJobName=job_name
            )
            status = response['TransformJobStatus']

            if status in ['Completed', 'Failed', 'Stopped']:
                logger.info(f"Transform job finished with status: {status}")

                if status == 'Failed':
                    failure_reason = response.get('FailureReason', 'Unknown')
                    logger.error(f"Transform job failed: {failure_reason}")
                    raise Exception(f"Transform job failed: {failure_reason}")

                return response

            logger.info(f"Transform job status: {status}. Waiting {poll_interval}s...")
            time.sleep(poll_interval)

    def validate_predictions(self, output_s3_path):
        """
        Validate that predictions were generated successfully.

        Args:
            output_s3_path: S3 path to predictions output

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating predictions output...")

        # Parse S3 path
        path_parts = output_s3_path.replace('s3://', '').split('/')
        bucket = path_parts[0]
        prefix = '/'.join(path_parts[1:])

        # List objects in output path
        response = self.s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )

        if 'Contents' not in response or len(response['Contents']) == 0:
            raise Exception(f"No prediction files found at {output_s3_path}")

        # Get first prediction file
        prediction_files = [obj['Key'] for obj in response['Contents']]
        logger.info(f"Found {len(prediction_files)} prediction file(s)")

        # Load and validate first file
        first_file_key = prediction_files[0]
        obj = self.s3_client.get_object(Bucket=bucket, Key=first_file_key)

        try:
            # Try to read predictions
            predictions_df = pd.read_csv(obj['Body'], header=None)
            num_predictions = len(predictions_df)

            logger.info(f"Successfully loaded {num_predictions} predictions")
            logger.info(f"Sample predictions:\n{predictions_df.head()}")

            validation_results = {
                'status': 'success',
                'num_files': len(prediction_files),
                'num_predictions': num_predictions,
                'output_path': output_s3_path,
                'sample_predictions': predictions_df.head(5).values.tolist()
            }

            return validation_results

        except Exception as e:
            logger.error(f"Failed to read predictions: {e}")
            raise

    def run(
        self,
        model_s3_path,
        input_s3_path,
        output_s3_path,
        role_arn,
        image_uri
    ):
        """
        Execute the complete inference pipeline.

        Args:
            model_s3_path: S3 path to trained model
            input_s3_path: S3 path to input data for predictions
            output_s3_path: S3 path for predictions output
            role_arn: SageMaker execution role ARN
            image_uri: XGBoost container image URI

        Returns:
            Dictionary with inference results
        """
        logger.info("=" * 60)
        logger.info("Starting Inference Pipeline")
        logger.info("=" * 60)

        # Create unique names
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_name = f"xgboost-titanic-model-{timestamp}"
        job_name = f"xgboost-batch-transform-{timestamp}"

        # Step 1: Create model
        model_arn = self.create_model(
            model_name=model_name,
            model_s3_path=model_s3_path,
            role_arn=role_arn,
            image_uri=image_uri
        )

        # Step 2: Create batch transform job
        transform_job_name = self.create_batch_transform_job(
            job_name=job_name,
            model_name=model_name,
            input_s3_path=input_s3_path,
            output_s3_path=output_s3_path
        )

        # Step 3: Wait for completion
        transform_info = self.wait_for_transform_job(transform_job_name)

        # Step 4: Validate predictions
        validation_results = self.validate_predictions(output_s3_path)

        results = {
            'model_name': model_name,
            'transform_job_name': transform_job_name,
            'output_path': output_s3_path,
            'validation': validation_results,
            'status': 'completed'
        }

        logger.info("=" * 60)
        logger.info("Inference Pipeline Completed Successfully!")
        logger.info("=" * 60)

        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="XGBoost Inference Pipeline")

    parser.add_argument(
        '--model-path',
        required=True,
        help='S3 path to trained model (model.tar.gz)'
    )
    parser.add_argument(
        '--input-data',
        required=True,
        help='S3 path to input data for predictions'
    )
    parser.add_argument(
        '--output-path',
        required=True,
        help='S3 path for predictions output'
    )
    parser.add_argument(
        '--role-arn',
        required=True,
        help='SageMaker execution role ARN'
    )
    parser.add_argument(
        '--image-uri',
        required=True,
        help='XGBoost container image URI'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--instance-type',
        default='ml.m5.large',
        help='Instance type for batch transform (default: ml.m5.large)'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Initialize and run pipeline
    pipeline = InferencePipeline(region=args.region)

    results = pipeline.run(
        model_s3_path=args.model_path,
        input_s3_path=args.input_data,
        output_s3_path=args.output_path,
        role_arn=args.role_arn,
        image_uri=args.image_uri
    )

    # Print results
    print("\n" + "=" * 60)
    print("INFERENCE PIPELINE RESULTS")
    print("=" * 60)
    print(f"Model Name:     {results['model_name']}")
    print(f"Transform Job:  {results['transform_job_name']}")
    print(f"Output Path:    {results['output_path']}")
    print(f"\nValidation:")
    print(f"  Files:        {results['validation']['num_files']}")
    print(f"  Predictions:  {results['validation']['num_predictions']}")
    print("=" * 60)
    print("\n✅ Predictions generated successfully!")


if __name__ == '__main__':
    main()
