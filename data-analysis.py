"""
Data Exploratory Analysis for SemEval-2026-Task13 Dataset
This script loads the dataset from Hugging Face and performs basic EDA.
"""

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def main():
    """
    Main function to parse command-line arguments and run data analysis.
    """
    # Create argument parser for command-line options
    parser = argparse.ArgumentParser(
        description='Data Exploratory Analysis for SemEval-2026-Task13 Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add argument for task selection (A, B, or C)
    parser.add_argument(
        '--task',
        type=str,
        choices=['A', 'B', 'C'],
        default='A',
        help='Task subset to analyze (A, B, or C)'
    )

    # Add argument for output directory where images will be saved
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save visualization images'
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Get the selected task from command-line arguments
    task = args.task
    # Get the output directory path from command-line arguments
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")

    # Login using e.g. `huggingface-cli login` to access this dataset
    # Load the dataset with the specified task
    print(f"\nLoading dataset from Hugging Face (Task {task})...")
    ds = load_dataset("DaniilOr/SemEval-2026-Task13", task)
    print("Dataset loaded successfully.")

    # Display basic information about the dataset
    print("\n" + "="*80)
    print(f"DATASET OVERVIEW - TASK {task}")
    print("="*80)
    print(f"\nDataset structure: {ds}")
    print(f"\nAvailable splits: {list(ds.keys())}")

    # Analyze each split
    for split_name in ds.keys():
        print(f"\n{'-'*80}")
        print(f"Split: {split_name}")
        print(f"{'-'*80}")

        # Get the split data
        split_data = ds[split_name]

        # Basic statistics
        print(f"Number of examples: {len(split_data)}")
        print(f"Features: {split_data.features}")
        print(f"Column names: {split_data.column_names}")

        # Convert to pandas DataFrame for easier analysis
        df = split_data.to_pandas()

        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())

        print(f"\nData types:")
        print(df.dtypes)

        print(f"\nMissing values:")
        print(df.isnull().sum())

        print(f"\nBasic statistics:")
        print(df.describe(include='all'))

        # Analyze text columns (if any)
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            print(f"\n{'='*80}")
            print("TEXT ANALYSIS")
            print(f"{'='*80}")

            for col in text_columns:
                if df[col].dtype == 'object' and df[col].notna().any():
                    print(f"\nColumn: {col}")
                    print(f"Unique values: {df[col].nunique()}")
                    print(f"Sample values:")
                    print(df[col].value_counts().head(10))

                    # Calculate text length if it's a text field
                    try:
                        # Get text lengths for all non-null values in the column
                        text_lengths = df[col].dropna().astype(str).str.len()
                        print(f"\nText length statistics:")
                        print(f"  Mean: {text_lengths.mean():.2f}")
                        print(f"  Median: {text_lengths.median():.2f}")
                        print(f"  Min: {text_lengths.min()}")
                        print(f"  Max: {text_lengths.max()}")
                    except:
                        pass

        # Analyze numeric columns (if any)
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_columns) > 0:
            print(f"\n{'='*80}")
            print("NUMERIC ANALYSIS")
            print(f"{'='*80}")

            for col in numeric_columns:
                print(f"\nColumn: {col}")
                print(f"Value distribution:")
                print(df[col].value_counts().head(10))

        # Create visualizations
        print(f"\n{'='*80}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*80}")

        # Plot distribution of text lengths for text columns
        if len(text_columns) > 0:
            for col in text_columns:
                try:
                    # Calculate text lengths for visualization
                    text_lengths = df[col].dropna().astype(str).str.len()

                    # Create figure with two subplots for text length analysis
                    plt.figure(figsize=(12, 6))

                    # Subplot 1: Histogram of text lengths
                    plt.subplot(1, 2, 1)
                    plt.hist(text_lengths, bins=50, edgecolor='black')
                    plt.xlabel('Text Length (characters)')
                    plt.ylabel('Frequency')
                    plt.title(f'Distribution of Text Length - {col} ({split_name})')
                    plt.grid(True, alpha=0.3)

                    # Subplot 2: Box plot of text lengths
                    plt.subplot(1, 2, 2)
                    plt.boxplot(text_lengths)
                    plt.ylabel('Text Length (characters)')
                    plt.title(f'Box Plot of Text Length - {col} ({split_name})')
                    plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    # Construct filename with task name and save to output directory
                    filename = os.path.join(output_dir, f'text_length_distribution_task{task}_{split_name}_{col}.png')
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"Saved: {filename}")
                    plt.close()
                except Exception as e:
                    print(f"Could not create visualization for {col}: {e}")

        # Plot distribution of numeric columns
        if len(numeric_columns) > 0:
            for col in numeric_columns:
                try:
                    # Create bar plot for numeric column distribution
                    plt.figure(figsize=(10, 6))
                    # Get value counts sorted by index
                    value_counts = df[col].value_counts().sort_index()
                    plt.bar(value_counts.index, value_counts.values, edgecolor='black')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.title(f'Distribution of {col} ({split_name}) - Task {task}')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    # Construct filename with task name and save to output directory
                    filename = os.path.join(output_dir, f'distribution_task{task}_{split_name}_{col}.png')
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"Saved: {filename}")
                    plt.close()
                except Exception as e:
                    print(f"Could not create visualization for {col}: {e}")

        print(f"\n{'-'*80}\n")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    # Run main function when script is executed directly
    main()
