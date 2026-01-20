#!/usr/bin/env python3
"""
Deploy trained model to SageMaker endpoint with monitoring.
"""
import argparse
import json
import logging
import time
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerDeployer:
    """Handle SageMaker model deployment and monitoring setup."""

    def __init__(self, region_name="us-east-1"):
        """Initialize AWS clients."""
        self.sagemaker = boto3.client("sagemaker", region_name=region_name)
        self.cloudwatch = boto3.client("cloudwatch", region_name=region_name)
        self.region = region_name

    def create_model(self, model_name, model_data, role, image_uri=None):
        """
        Create a SageMaker model.

        Args:
            model_name: Name for the model
            model_data: S3 URI to model.tar.gz
            role: IAM role ARN for SageMaker
            image_uri: Optional custom inference image

        Returns:
            Model ARN
        """
        logger.info(f"Creating model: {model_name}")

        # Use default scikit-learn container if not specified
        if not image_uri:
            account_id = boto3.client("sts").get_caller_identity()["Account"]
            image_uri = f"{account_id}.dkr.ecr.{self.region}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"

        try:
            response = self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    "Image": image_uri,
                    "ModelDataUrl": model_data,
                    "Environment": {
                        "SAGEMAKER_PROGRAM": "inference.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": model_data,
                    },
                },
                ExecutionRoleArn=role,
            )

            logger.info(f"Model created: {response['ModelArn']}")
            return response["ModelArn"]

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                logger.warning(f"Model {model_name} already exists")
                return f"arn:aws:sagemaker:{self.region}::model/{model_name}"
            raise

    def create_endpoint_config(
        self, config_name, model_name, instance_type="ml.t2.medium", instance_count=1
    ):
        """
        Create endpoint configuration.

        Args:
            config_name: Name for the endpoint configuration
            model_name: Name of the model to deploy
            instance_type: EC2 instance type
            instance_count: Number of instances

        Returns:
            Endpoint config ARN
        """
        logger.info(f"Creating endpoint configuration: {config_name}")

        try:
            response = self.sagemaker.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        "VariantName": "AllTraffic",
                        "ModelName": model_name,
                        "InstanceType": instance_type,
                        "InitialInstanceCount": instance_count,
                        "InitialVariantWeight": 1.0,
                    }
                ],
                DataCaptureConfig={
                    "EnableCapture": True,
                    "InitialSamplingPercentage": 100,
                    "DestinationS3Uri": f"s3://sagemaker-{self.region}/data-capture",
                    "CaptureOptions": [
                        {"CaptureMode": "Input"},
                        {"CaptureMode": "Output"},
                    ],
                },
            )

            logger.info(f"Endpoint config created: {response['EndpointConfigArn']}")
            return response["EndpointConfigArn"]

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                logger.warning(f"Endpoint config {config_name} already exists")
                return f"arn:aws:sagemaker:{self.region}::endpoint-config/{config_name}"
            raise

    def create_or_update_endpoint(self, endpoint_name, config_name):
        """
        Create new endpoint or update existing one.

        Args:
            endpoint_name: Name for the endpoint
            config_name: Name of the endpoint configuration

        Returns:
            Endpoint ARN
        """
        logger.info(f"Creating/updating endpoint: {endpoint_name}")

        try:
            # Try to create new endpoint
            response = self.sagemaker.create_endpoint(
                EndpointName=endpoint_name, EndpointConfigName=config_name
            )
            logger.info(f"Endpoint creation initiated: {response['EndpointArn']}")
            endpoint_arn = response["EndpointArn"]

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                # Endpoint exists, update it
                logger.info(f"Endpoint {endpoint_name} exists, updating...")
                response = self.sagemaker.update_endpoint(
                    EndpointName=endpoint_name, EndpointConfigName=config_name
                )
                endpoint_arn = response["EndpointArn"]
            else:
                raise

        # Wait for endpoint to be in service
        self._wait_for_endpoint(endpoint_name)
        return endpoint_arn

    def _wait_for_endpoint(self, endpoint_name, timeout=600):
        """Wait for endpoint to be in service."""
        logger.info(f"Waiting for endpoint {endpoint_name} to be in service...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]

            if status == "InService":
                logger.info(f"Endpoint {endpoint_name} is now in service!")
                return
            elif status == "Failed":
                raise Exception(f"Endpoint creation failed: {response.get('FailureReason', 'Unknown')}")

            logger.info(f"Current status: {status}. Waiting...")
            time.sleep(30)

        raise TimeoutError(f"Endpoint {endpoint_name} did not reach InService within {timeout}s")

    def setup_monitoring(self, endpoint_name, sns_topic_arn=None):
        """
        Set up CloudWatch alarms for endpoint monitoring.

        Args:
            endpoint_name: Name of the endpoint to monitor
            sns_topic_arn: Optional SNS topic for alarm notifications
        """
        logger.info(f"Setting up monitoring for endpoint: {endpoint_name}")

        alarms = [
            {
                "name": f"{endpoint_name}-high-invocation-errors",
                "metric": "ModelInvocationErrors",
                "threshold": 10,
                "comparison": "GreaterThanThreshold",
                "description": "Alert when model invocation errors exceed threshold",
            },
            {
                "name": f"{endpoint_name}-high-latency",
                "metric": "ModelLatency",
                "threshold": 10000,  # 10 seconds
                "comparison": "GreaterThanThreshold",
                "description": "Alert when model latency is too high",
            },
            {
                "name": f"{endpoint_name}-low-invocations",
                "metric": "Invocations",
                "threshold": 1,
                "comparison": "LessThanThreshold",
                "description": "Alert when endpoint receives no traffic",
            },
        ]

        for alarm in alarms:
            alarm_params = {
                "AlarmName": alarm["name"],
                "AlarmDescription": alarm["description"],
                "MetricName": alarm["metric"],
                "Namespace": "AWS/SageMaker",
                "Statistic": "Average",
                "Period": 300,  # 5 minutes
                "EvaluationPeriods": 2,
                "Threshold": alarm["threshold"],
                "ComparisonOperator": alarm["comparison"],
                "Dimensions": [
                    {"Name": "EndpointName", "Value": endpoint_name},
                    {"Name": "VariantName", "Value": "AllTraffic"},
                ],
            }

            if sns_topic_arn:
                alarm_params["AlarmActions"] = [sns_topic_arn]

            try:
                self.cloudwatch.put_metric_alarm(**alarm_params)
                logger.info(f"Created alarm: {alarm['name']}")
            except ClientError as e:
                logger.error(f"Failed to create alarm {alarm['name']}: {e}")

    def get_endpoint_info(self, endpoint_name):
        """Get endpoint information and return as JSON."""
        response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
        return {
            "endpoint_name": endpoint_name,
            "endpoint_arn": response["EndpointArn"],
            "status": response["EndpointStatus"],
            "creation_time": response["CreationTime"].isoformat(),
            "last_modified_time": response["LastModifiedTime"].isoformat(),
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deploy model to SageMaker endpoint")

    parser.add_argument("--model-data", required=True, help="S3 URI to model.tar.gz")
    parser.add_argument("--endpoint-name", required=True, help="Name for the endpoint")
    parser.add_argument(
        "--role", required=True, help="IAM role ARN for SageMaker execution"
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument(
        "--instance-type", default="ml.t2.medium", help="Instance type for endpoint"
    )
    parser.add_argument(
        "--instance-count", type=int, default=1, help="Number of instances"
    )
    parser.add_argument(
        "--sns-topic", help="SNS topic ARN for CloudWatch alarm notifications"
    )
    parser.add_argument(
        "--image-uri", help="Custom Docker image URI for inference"
    )

    return parser.parse_args()


def main():
    """Main deployment function."""
    args = parse_args()

    logger.info("Starting model deployment...")
    logger.info(f"Model data: {args.model_data}")
    logger.info(f"Endpoint: {args.endpoint_name}")

    # Initialize deployer
    deployer = SageMakerDeployer(region_name=args.region)

    # Generate unique names
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"{args.endpoint_name}-model-{timestamp}"
    config_name = f"{args.endpoint_name}-config-{timestamp}"

    try:
        # Step 1: Create model
        model_arn = deployer.create_model(
            model_name=model_name,
            model_data=args.model_data,
            role=args.role,
            image_uri=args.image_uri,
        )

        # Step 2: Create endpoint configuration
        config_arn = deployer.create_endpoint_config(
            config_name=config_name,
            model_name=model_name,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
        )

        # Step 3: Create or update endpoint
        endpoint_arn = deployer.create_or_update_endpoint(
            endpoint_name=args.endpoint_name, config_name=config_name
        )

        # Step 4: Set up monitoring
        deployer.setup_monitoring(
            endpoint_name=args.endpoint_name, sns_topic_arn=args.sns_topic
        )

        # Get endpoint info
        endpoint_info = deployer.get_endpoint_info(args.endpoint_name)

        logger.info("\n" + "=" * 50)
        logger.info("Deployment completed successfully!")
        logger.info("=" * 50)
        logger.info(f"\nEndpoint details:\n{json.dumps(endpoint_info, indent=2)}")
        logger.info(f"\nTo test the endpoint, run:")
        logger.info(
            f"  python test_inference.py --endpoint-name {args.endpoint_name}"
        )

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()
