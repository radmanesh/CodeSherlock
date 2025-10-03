"""
Data Exploratory Analysis for SemEval-2026-Task13 Dataset
This script loads the dataset from Hugging Face and performs basic EDA.
"""

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Login using e.g. `huggingface-cli login` to access this dataset
# Load the dataset
print("Loading dataset from Hugging Face...")
ds = load_dataset("DaniilOr/SemEval-2026-Task13", "A")

# Display basic information about the dataset
print("\n" + "="*80)
print("DATASET OVERVIEW")
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
                text_lengths = df[col].dropna().astype(str).str.len()

                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.hist(text_lengths, bins=50, edgecolor='black')
                plt.xlabel('Text Length (characters)')
                plt.ylabel('Frequency')
                plt.title(f'Distribution of Text Length - {col} ({split_name})')
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 2, 2)
                plt.boxplot(text_lengths)
                plt.ylabel('Text Length (characters)')
                plt.title(f'Box Plot of Text Length - {col} ({split_name})')
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f'text_length_distribution_{split_name}_{col}.png', dpi=300, bbox_inches='tight')
                print(f"Saved: text_length_distribution_{split_name}_{col}.png")
                plt.close()
            except Exception as e:
                print(f"Could not create visualization for {col}: {e}")

    # Plot distribution of numeric columns
    if len(numeric_columns) > 0:
        for col in numeric_columns:
            try:
                plt.figure(figsize=(10, 6))
                value_counts = df[col].value_counts().sort_index()
                plt.bar(value_counts.index, value_counts.values, edgecolor='black')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.title(f'Distribution of {col} ({split_name})')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'distribution_{split_name}_{col}.png', dpi=300, bbox_inches='tight')
                print(f"Saved: distribution_{split_name}_{col}.png")
                plt.close()
            except Exception as e:
                print(f"Could not create visualization for {col}: {e}")

    print(f"\n{'-'*80}\n")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
