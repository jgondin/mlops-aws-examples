#!/usr/bin/env python3
"""
Feature Pipeline: Load Titanic dataset, preprocess, and save to S3.

This script handles data loading, cleaning, feature engineering, and splitting
for the XGBoost binary classification model.
"""
import argparse
import logging
import os
from io import StringIO

import boto3
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Handle Titanic dataset preprocessing and feature engineering."""

    def __init__(self, s3_bucket, s3_prefix='processed-data'):
        """
        Initialize the feature pipeline.

        Args:
            s3_bucket: S3 bucket name for storing processed data
            s3_prefix: S3 prefix/folder for processed data
        """
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client('s3')
        logger.info(f"Initialized FeaturePipeline with bucket: {s3_bucket}")

    def load_titanic_data(self):
        """
        Load Titanic dataset from sklearn.

        Returns:
            pandas DataFrame with Titanic data
        """
        logger.info("Loading Titanic dataset from sklearn...")

        try:
            # Load Titanic dataset from OpenML (via sklearn)
            titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
            df = titanic.frame

            logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
            logger.info(f"Columns: {list(df.columns)}")

            return df

        except Exception as e:
            logger.error(f"Failed to load Titanic dataset: {e}")
            raise

    def preprocess_data(self, df):
        """
        Preprocess Titanic data: handle missing values and encode categoricals.

        Args:
            df: Raw Titanic DataFrame

        Returns:
            Preprocessed DataFrame ready for training
        """
        logger.info("Starting data preprocessing...")

        # Create a copy to avoid modifying original
        data = df.copy()

        # --- Handle Target Variable ---
        # Survived is the target (0 = No, 1 = Yes)
        if 'survived' not in data.columns:
            raise ValueError("Target column 'survived' not found in dataset")

        # Rename target for clarity
        data = data.rename(columns={'survived': 'target'})

        # --- Select Features ---
        # Use a subset of features for this POC
        feature_columns = [
            'pclass',      # Passenger class (1, 2, 3)
            'sex',         # Gender
            'age',         # Age in years
            'sibsp',       # Number of siblings/spouses aboard
            'parch',       # Number of parents/children aboard
            'fare',        # Passenger fare
            'embarked',    # Port of embarkation (C, Q, S)
        ]

        # Keep only selected features + target
        available_features = [col for col in feature_columns if col in data.columns]
        data = data[available_features + ['target']]

        logger.info(f"Selected features: {available_features}")

        # --- Handle Missing Values ---
        # Age: Fill with median
        if 'age' in data.columns:
            median_age = data['age'].median()
            data['age'].fillna(median_age, inplace=True)
            logger.info(f"Filled missing age values with median: {median_age:.1f}")

        # Fare: Fill with median
        if 'fare' in data.columns:
            median_fare = data['fare'].median()
            data['fare'].fillna(median_fare, inplace=True)
            logger.info(f"Filled missing fare values with median: {median_fare:.2f}")

        # Embarked: Fill with mode (most common)
        if 'embarked' in data.columns:
            mode_embarked = data['embarked'].mode()[0] if not data['embarked'].mode().empty else 'S'
            data['embarked'].fillna(mode_embarked, inplace=True)
            logger.info(f"Filled missing embarked values with mode: {mode_embarked}")

        # Drop any remaining rows with missing values
        initial_rows = len(data)
        data = data.dropna()
        dropped_rows = initial_rows - len(data)
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with remaining missing values")

        # --- Encode Categorical Variables ---
        # Sex: Male=1, Female=0
        if 'sex' in data.columns:
            le_sex = LabelEncoder()
            data['sex'] = le_sex.fit_transform(data['sex'])
            logger.info(f"Encoded sex: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

        # Embarked: C=0, Q=1, S=2
        if 'embarked' in data.columns:
            le_embarked = LabelEncoder()
            data['embarked'] = le_embarked.fit_transform(data['embarked'])
            logger.info(f"Encoded embarked: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")

        # --- Final Data Quality Check ---
        logger.info(f"Preprocessing complete. Final dataset shape: {data.shape}")
        logger.info(f"Target distribution:\n{data['target'].value_counts(normalize=True)}")

        return data

    def split_data(self, data, test_size=0.2, random_state=42):
        """
        Split data into train and test sets.

        Args:
            data: Preprocessed DataFrame
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Splitting data: train={1-test_size:.0%}, test={test_size:.0%}")

        train_df, test_df = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=data['target']  # Maintain class balance
        )

        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")

        return train_df, test_df

    def save_to_s3(self, train_df, test_df):
        """
        Save processed datasets to S3 as CSV files.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame

        Returns:
            Dictionary with S3 paths
        """
        logger.info("Saving processed data to S3...")

        s3_paths = {}

        # Save training data
        train_csv = train_df.to_csv(index=False)
        train_key = f"{self.s3_prefix}/train.csv"
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=train_key,
            Body=train_csv
        )
        s3_paths['train'] = f"s3://{self.s3_bucket}/{train_key}"
        logger.info(f"Saved training data to {s3_paths['train']}")

        # Save test data
        test_csv = test_df.to_csv(index=False)
        test_key = f"{self.s3_prefix}/test.csv"
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=test_key,
            Body=test_csv
        )
        s3_paths['test'] = f"s3://{self.s3_bucket}/{test_key}"
        logger.info(f"Saved test data to {s3_paths['test']}")

        # Save metadata
        metadata = {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'features': list(train_df.columns.drop('target')),
            'train_path': s3_paths['train'],
            'test_path': s3_paths['test']
        }

        import json
        metadata_key = f"{self.s3_prefix}/metadata.json"
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2)
        )
        logger.info(f"Saved metadata to s3://{self.s3_bucket}/{metadata_key}")

        return s3_paths

    def run(self):
        """
        Execute the complete feature pipeline.

        Returns:
            Dictionary with S3 paths to processed data
        """
        logger.info("=" * 60)
        logger.info("Starting Feature Pipeline")
        logger.info("=" * 60)

        # Step 1: Load data
        raw_data = self.load_titanic_data()

        # Step 2: Preprocess
        processed_data = self.preprocess_data(raw_data)

        # Step 3: Split
        train_df, test_df = self.split_data(processed_data)

        # Step 4: Save to S3
        s3_paths = self.save_to_s3(train_df, test_df)

        logger.info("=" * 60)
        logger.info("Feature Pipeline Completed Successfully!")
        logger.info("=" * 60)

        return s3_paths


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Titanic Feature Pipeline")

    parser.add_argument(
        '--s3-bucket',
        required=True,
        help='S3 bucket name for storing processed data'
    )
    parser.add_argument(
        '--s3-prefix',
        default='processed-data',
        help='S3 prefix/folder for processed data (default: processed-data)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set (default: 0.2)'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Initialize and run pipeline
    pipeline = FeaturePipeline(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix
    )

    s3_paths = pipeline.run()

    # Print results
    print("\n" + "=" * 60)
    print("FEATURE PIPELINE RESULTS")
    print("=" * 60)
    print(f"Training data: {s3_paths['train']}")
    print(f"Test data:     {s3_paths['test']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
