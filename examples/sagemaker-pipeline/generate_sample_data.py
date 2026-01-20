#!/usr/bin/env python3
"""
Generate sample customer churn data for training and testing.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def generate_churn_data(n_samples=10000, random_seed=42):
    """
    Generate synthetic customer churn dataset.

    Args:
        n_samples: Number of samples to generate
        random_seed: Random seed for reproducibility

    Returns:
        pandas DataFrame with customer features and churn label
    """
    np.random.seed(random_seed)

    # Generate customer demographics
    age = np.random.normal(45, 15, n_samples).clip(18, 90).astype(int)
    tenure_months = np.random.exponential(24, n_samples).clip(0, 120).astype(int)
    monthly_charges = np.random.normal(65, 30, n_samples).clip(10, 200)
    total_charges = monthly_charges * tenure_months + np.random.normal(0, 100, n_samples)

    # Generate service usage features
    num_products = np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    has_online_service = np.random.binomial(1, 0.6, n_samples)
    has_tech_support = np.random.binomial(1, 0.5, n_samples)
    has_streaming = np.random.binomial(1, 0.4, n_samples)

    # Customer engagement
    num_support_tickets = np.random.poisson(2, n_samples)
    num_logins_per_month = np.random.gamma(3, 5, n_samples).clip(0, 100).astype(int)
    avg_session_duration_min = np.random.exponential(15, n_samples).clip(0, 120)

    # Contract and payment
    contract_type = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples,
        p=[0.5, 0.3, 0.2],
    )
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_samples,
        p=[0.3, 0.2, 0.25, 0.25],
    )
    paperless_billing = np.random.binomial(1, 0.6, n_samples)

    # Generate churn based on risk factors
    # Higher risk: short tenure, high charges, many support tickets, month-to-month contract
    churn_probability = (
        0.05  # Base rate
        + (tenure_months < 12) * 0.15  # New customers more likely to churn
        + (monthly_charges > 80) * 0.10  # High charges increase churn
        + (num_support_tickets > 3) * 0.12  # Many support issues
        + (contract_type == "Month-to-month") * 0.20  # No commitment
        + (num_products == 1) * 0.08  # Low engagement
        + (num_logins_per_month < 5) * 0.10  # Low usage
        - (has_tech_support) * 0.05  # Support reduces churn
        - (tenure_months > 36) * 0.15  # Loyal customers less likely to churn
    )

    churn_probability = np.clip(churn_probability, 0, 0.8)
    churn = np.random.binomial(1, churn_probability)

    # Create DataFrame
    data = pd.DataFrame({
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "num_products": num_products,
        "has_online_service": has_online_service,
        "has_tech_support": has_tech_support,
        "has_streaming": has_streaming,
        "num_support_tickets": num_support_tickets,
        "num_logins_per_month": num_logins_per_month,
        "avg_session_duration_min": avg_session_duration_min.round(1),
        "contract_type": contract_type,
        "payment_method": payment_method,
        "paperless_billing": paperless_billing,
        "churn": churn,
    })

    return data


def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train, validation, and test sets.

    Args:
        data: DataFrame to split
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Shuffle data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = data[:train_end]
    val_df = data[train_end:val_end]
    test_df = data[val_end:]

    return train_df, val_df, test_df


def main():
    """Generate and save sample data."""
    parser = argparse.ArgumentParser(description="Generate sample churn data")
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Output directory for data files",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    print(f"Generating {args.n_samples} samples...")

    # Generate data
    data = generate_churn_data(n_samples=args.n_samples, random_seed=args.seed)

    # Split data
    train_df, val_df, test_df = split_data(data)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets
    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nData saved:")
    print(f"  Training:   {train_path} ({len(train_df)} samples)")
    print(f"  Validation: {val_path} ({len(val_df)} samples)")
    print(f"  Test:       {test_path} ({len(test_df)} samples)")

    # Print statistics
    print(f"\nChurn rate:")
    print(f"  Training:   {train_df['churn'].mean():.1%}")
    print(f"  Validation: {val_df['churn'].mean():.1%}")
    print(f"  Test:       {test_df['churn'].mean():.1%}")

    print(f"\nSample data (first 5 rows):")
    print(train_df.head())

    print("\nData generation complete!")


if __name__ == "__main__":
    main()
