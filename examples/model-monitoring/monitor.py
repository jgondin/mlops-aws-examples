#!/usr/bin/env python3
"""
Model monitoring script for detecting data drift and performance degradation.
"""
import argparse
import json
import logging
from datetime import datetime, timedelta

import boto3
import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor deployed models for drift and performance issues."""

    def __init__(self, endpoint_name, region_name="us-east-1"):
        """Initialize monitoring clients."""
        self.endpoint_name = endpoint_name
        self.sagemaker = boto3.client("sagemaker", region_name=region_name)
        self.cloudwatch = boto3.client("cloudwatch", region_name=region_name)
        self.s3 = boto3.client("s3", region_name=region_name)

    def get_endpoint_metrics(self, hours=24):
        """
        Fetch CloudWatch metrics for the endpoint.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Fetching metrics for {self.endpoint_name} (last {hours} hours)")

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        metrics = {}
        metric_definitions = [
            ("Invocations", "Sum"),
            ("ModelLatency", "Average"),
            ("ModelInvocationErrors", "Sum"),
            ("CPUUtilization", "Average"),
            ("MemoryUtilization", "Average"),
        ]

        for metric_name, statistic in metric_definitions:
            response = self.cloudwatch.get_metric_statistics(
                Namespace="AWS/SageMaker",
                MetricName=metric_name,
                Dimensions=[
                    {"Name": "EndpointName", "Value": self.endpoint_name},
                    {"Name": "VariantName", "Value": "AllTraffic"},
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour
                Statistics=[statistic],
            )

            datapoints = sorted(response["Datapoints"], key=lambda x: x["Timestamp"])
            metrics[metric_name] = {
                "values": [dp[statistic] for dp in datapoints],
                "timestamps": [dp["Timestamp"].isoformat() for dp in datapoints],
            }

        return metrics

    def analyze_performance(self, metrics):
        """
        Analyze endpoint performance metrics.

        Args:
            metrics: Dictionary of CloudWatch metrics

        Returns:
            Dictionary of analysis results
        """
        logger.info("Analyzing performance metrics...")

        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": self.endpoint_name,
            "issues": [],
            "statistics": {},
        }

        # Analyze invocations
        invocations = metrics.get("Invocations", {}).get("values", [])
        if invocations:
            analysis["statistics"]["total_invocations"] = sum(invocations)
            analysis["statistics"]["avg_invocations_per_hour"] = np.mean(invocations)

            # Check for traffic drop
            if len(invocations) > 2:
                recent_traffic = np.mean(invocations[-3:])
                historical_traffic = np.mean(invocations[:-3])
                if recent_traffic < historical_traffic * 0.5:
                    analysis["issues"].append({
                        "severity": "warning",
                        "type": "traffic_drop",
                        "message": f"Traffic dropped by {(1 - recent_traffic/historical_traffic)*100:.1f}%",
                    })

        # Analyze latency
        latency = metrics.get("ModelLatency", {}).get("values", [])
        if latency:
            avg_latency = np.mean(latency)
            p95_latency = np.percentile(latency, 95)
            analysis["statistics"]["avg_latency_ms"] = avg_latency
            analysis["statistics"]["p95_latency_ms"] = p95_latency

            # Check for high latency
            if avg_latency > 5000:  # 5 seconds
                analysis["issues"].append({
                    "severity": "critical",
                    "type": "high_latency",
                    "message": f"Average latency is {avg_latency:.0f}ms (threshold: 5000ms)",
                })
            elif avg_latency > 2000:  # 2 seconds
                analysis["issues"].append({
                    "severity": "warning",
                    "type": "elevated_latency",
                    "message": f"Average latency is {avg_latency:.0f}ms",
                })

        # Analyze errors
        errors = metrics.get("ModelInvocationErrors", {}).get("values", [])
        if errors:
            total_errors = sum(errors)
            error_rate = (total_errors / sum(invocations) * 100) if sum(invocations) > 0 else 0
            analysis["statistics"]["total_errors"] = total_errors
            analysis["statistics"]["error_rate_percent"] = error_rate

            # Check error rate
            if error_rate > 5:
                analysis["issues"].append({
                    "severity": "critical",
                    "type": "high_error_rate",
                    "message": f"Error rate is {error_rate:.2f}% (threshold: 5%)",
                })
            elif error_rate > 1:
                analysis["issues"].append({
                    "severity": "warning",
                    "type": "elevated_errors",
                    "message": f"Error rate is {error_rate:.2f}%",
                })

        # Analyze resource utilization
        cpu = metrics.get("CPUUtilization", {}).get("values", [])
        memory = metrics.get("MemoryUtilization", {}).get("values", [])

        if cpu:
            avg_cpu = np.mean(cpu)
            analysis["statistics"]["avg_cpu_percent"] = avg_cpu
            if avg_cpu > 80:
                analysis["issues"].append({
                    "severity": "warning",
                    "type": "high_cpu",
                    "message": f"CPU utilization is {avg_cpu:.1f}% (threshold: 80%)",
                })

        if memory:
            avg_memory = np.mean(memory)
            analysis["statistics"]["avg_memory_percent"] = avg_memory
            if avg_memory > 80:
                analysis["issues"].append({
                    "severity": "warning",
                    "type": "high_memory",
                    "message": f"Memory utilization is {avg_memory:.1f}% (threshold: 80%)",
                })

        return analysis

    def detect_data_drift(self, baseline_data, current_data, threshold=0.05):
        """
        Detect data drift using Kolmogorov-Smirnov test.

        Args:
            baseline_data: DataFrame with baseline feature distributions
            current_data: DataFrame with current feature distributions
            threshold: P-value threshold for drift detection

        Returns:
            Dictionary of drift analysis results
        """
        logger.info("Detecting data drift...")

        drift_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "features_analyzed": 0,
            "features_drifted": 0,
            "drift_details": [],
        }

        # Ensure same features
        common_features = set(baseline_data.columns) & set(current_data.columns)
        if not common_features:
            logger.warning("No common features found between baseline and current data")
            return drift_results

        for feature in common_features:
            # Skip non-numeric features
            if not pd.api.types.is_numeric_dtype(baseline_data[feature]):
                continue

            drift_results["features_analyzed"] += 1

            # Kolmogorov-Smirnov test
            baseline_values = baseline_data[feature].dropna()
            current_values = current_data[feature].dropna()

            if len(baseline_values) < 10 or len(current_values) < 10:
                continue

            ks_statistic, p_value = stats.ks_2samp(baseline_values, current_values)

            # Detect drift
            is_drifted = p_value < threshold

            if is_drifted:
                drift_results["features_drifted"] += 1

                # Calculate distribution statistics
                baseline_mean = baseline_values.mean()
                current_mean = current_values.mean()
                mean_change_percent = ((current_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0

                drift_results["drift_details"].append({
                    "feature": feature,
                    "ks_statistic": float(ks_statistic),
                    "p_value": float(p_value),
                    "baseline_mean": float(baseline_mean),
                    "current_mean": float(current_mean),
                    "mean_change_percent": float(mean_change_percent),
                    "severity": "high" if p_value < 0.01 else "medium",
                })

        drift_results["drift_detected"] = drift_results["features_drifted"] > 0

        if drift_results["drift_detected"]:
            logger.warning(
                f"Data drift detected in {drift_results['features_drifted']}/{drift_results['features_analyzed']} features"
            )
        else:
            logger.info("No significant data drift detected")

        return drift_results

    def generate_report(self, analysis, drift_results=None):
        """Generate monitoring report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": self.endpoint_name,
            "performance_analysis": analysis,
        }

        if drift_results:
            report["drift_analysis"] = drift_results

        # Summary
        report["summary"] = {
            "total_issues": len(analysis.get("issues", [])),
            "critical_issues": sum(1 for i in analysis.get("issues", []) if i.get("severity") == "critical"),
            "warnings": sum(1 for i in analysis.get("issues", []) if i.get("severity") == "warning"),
        }

        if drift_results:
            report["summary"]["drift_detected"] = drift_results.get("drift_detected", False)
            report["summary"]["features_drifted"] = drift_results.get("features_drifted", 0)

        return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Monitor SageMaker endpoint")

    parser.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--hours", type=int, default=24, help="Hours to analyze")
    parser.add_argument("--baseline-data", help="Path to baseline data CSV for drift detection")
    parser.add_argument("--current-data", help="Path to current data CSV for drift detection")
    parser.add_argument("--output", help="Output file for monitoring report (JSON)")

    return parser.parse_args()


def main():
    """Main monitoring function."""
    args = parse_args()

    logger.info(f"Starting monitoring for endpoint: {args.endpoint_name}")

    # Initialize monitor
    monitor = ModelMonitor(args.endpoint_name, region_name=args.region)

    # Get and analyze metrics
    metrics = monitor.get_endpoint_metrics(hours=args.hours)
    analysis = monitor.analyze_performance(metrics)

    # Drift detection (optional)
    drift_results = None
    if args.baseline_data and args.current_data:
        baseline_df = pd.read_csv(args.baseline_data)
        current_df = pd.read_csv(args.current_data)
        drift_results = monitor.detect_data_drift(baseline_df, current_df)

    # Generate report
    report = monitor.generate_report(analysis, drift_results)

    # Output report
    report_json = json.dumps(report, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report_json)
        logger.info(f"Report saved to {args.output}")
    else:
        print("\n" + "=" * 50)
        print("MONITORING REPORT")
        print("=" * 50)
        print(report_json)

    # Exit with error code if critical issues found
    if report["summary"]["critical_issues"] > 0:
        logger.error(f"Found {report['summary']['critical_issues']} critical issues")
        exit(1)


if __name__ == "__main__":
    main()
