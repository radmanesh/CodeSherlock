import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeBERTTrainer:
    def __init__(self, task_subset='A', max_length=512, model_name="microsoft/codebert-base"):
        # Task subset identifier (A, B, or C)
        self.task_subset = task_subset
        # Maximum sequence length for tokenization
        self.max_length = max_length
        # Pretrained model name from Hugging Face
        self.model_name = model_name
        # Tokenizer instance (initialized later)
        self.tokenizer = None
        # Model instance (initialized later)
        self.model = None
        # Number of classification labels (determined from data)
        self.num_labels = None

    def load_and_prepare_data(self, debug_samples=None):
        """
        Load and prepare the dataset for training.

        Args:
            debug_samples: Number of samples to use for debugging (None for all data)

        Returns:
            tuple: (train_df, val_df) - Training and validation DataFrames
        """
        logger.info(f"Loading dataset subset {self.task_subset}...")

        try:
            # Load dataset from Hugging Face
            dataset = load_dataset("DaniilOr/SemEval-2026-Task13", self.task_subset)

            # Extract training split
            train_data = dataset['train']
            logger.info(f"Loaded {len(train_data)} training samples")

            # Convert to pandas DataFrame for easier manipulation
            df = train_data.to_pandas()

            logger.info(f"Dataset columns: {df.columns.tolist()}")
            logger.info(f"Sample data:\n{df.head()}")

            # Validate required columns exist
            if 'code' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must contain 'code' and 'label' columns")

            # Remove rows with missing values in critical columns
            df = df.dropna(subset=['code', 'label'])

            # Limit dataset size for debugging if specified
            if debug_samples is not None:
                logger.info(f"DEBUG MODE: Using only {debug_samples} samples")
                df = df.head(debug_samples)

            # Convert labels to integers
            df['label'] = df['label'].astype(int)
            # Determine number of unique labels
            self.num_labels = df['label'].nunique()

            logger.info(f"Number of unique labels: {self.num_labels}")
            logger.info(f"Label range: {df['label'].min()} to {df['label'].max()}")
            logger.info(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

            # Split into train (80%) and validation (20%)
            train_size = int(0.8 * len(df))
            train_df = df[:train_size].reset_index(drop=True)
            val_df = df[train_size:].reset_index(drop=True)

            logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

            return train_df, val_df

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def initialize_model_and_tokenizer(self):
        """
        Initialize the pretrained model and tokenizer for sequence classification.
        """
        logger.info(f"Initializing {self.model_name} model and tokenizer...")

        # Load pretrained tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)

        # Load pretrained model for sequence classification
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )

        logger.info(f"Model initialized with {self.num_labels} labels")

    def tokenize_function(self, examples):
        """
        Tokenize code examples for model input.

        Args:
            examples: Batch of examples containing 'code' field

        Returns:
            dict: Tokenized inputs with input_ids, attention_mask
        """
        return self.tokenizer(
            examples['code'],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def prepare_datasets(self, train_df, val_df):
        """
        Convert DataFrames to HuggingFace datasets and tokenize.

        Args:
            train_df: Training DataFrame with 'code' and 'label' columns
            val_df: Validation DataFrame with 'code' and 'label' columns

        Returns:
            tuple: (train_dataset, val_dataset) - Tokenized datasets
        """
        logger.info("Preparing datasets for training...")

        # Convert pandas DataFrames to HuggingFace Dataset objects
        train_dataset = Dataset.from_pandas(train_df[['code', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['code', 'label']])

        # Apply tokenization to training dataset
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code']
        )
        # Apply tokenization to validation dataset
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code']
        )

        # Rename 'label' to 'labels' (required by Trainer)
        train_dataset = train_dataset.rename_column('label', 'labels')
        val_dataset = val_dataset.rename_column('label', 'labels')

        return train_dataset, val_dataset

    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics for model predictions.

        Args:
            eval_pred: Tuple of (predictions, labels)

        Returns:
            dict: Dictionary containing accuracy, F1, precision, and recall
        """
        # Extract predictions and true labels
        predictions, labels = eval_pred
        # Convert logits to predicted class labels
        predictions = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self, train_dataset, val_dataset, output_dir="./results", num_epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Train the model using the prepared datasets.

        Args:
            train_dataset: Tokenized training dataset
            val_dataset: Tokenized validation dataset
            output_dir: Directory to save model checkpoints and final model
            num_epochs: Number of training epochs
            batch_size: Batch size for training and evaluation
            learning_rate: Learning rate for optimizer

        Returns:
            Trainer: Trained Trainer object
        """
        logger.info("Starting training...")

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,  # Directory for saving model checkpoints
            num_train_epochs=num_epochs,  # Total number of training epochs
            per_device_train_batch_size=batch_size,  # Batch size per GPU/CPU for training
            per_device_eval_batch_size=batch_size,  # Batch size per GPU/CPU for evaluation
            warmup_steps=500,  # Number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # Weight decay for AdamW optimizer
            logging_dir='./logs',  # Directory for TensorBoard logs
            logging_steps=100,  # Log training metrics every N steps
            eval_strategy="steps",  # Evaluate every N steps (changed from evaluation_strategy)
            eval_steps=500,  # Evaluation frequency in steps
            save_strategy="steps",  # Save checkpoint every N steps
            save_steps=500,  # Checkpoint save frequency in steps
            load_best_model_at_end=True,  # Load best model at end of training
            metric_for_best_model="f1",  # Metric to use for selecting best model
            greater_is_better=True,  # Higher F1 is better
            remove_unused_columns=False,  # Keep all columns from dataset
            learning_rate=learning_rate,  # Learning rate for optimizer
            lr_scheduler_type="linear",  # Type of learning rate scheduler
            save_total_limit=2,  # Only keep last 2 checkpoints to save disk space
        )

        # Data collator for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Initialize Trainer with model, datasets, and training configuration
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Start training
        trainer.train()

        # Save final model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Training completed. Model saved to {output_dir}")

        return trainer

    def evaluate_model(self, trainer, val_dataset, output_dir="./results"):
        """
        Evaluate the trained model on validation dataset and generate confusion matrix.

        Args:
            trainer: Trained Trainer object
            val_dataset: Validation dataset for evaluation
            output_dir: Directory to save confusion matrix plot

        Returns:
            PredictionOutput: Predictions and metrics from evaluation
        """
        logger.info("Evaluating model...")

        # Generate predictions on validation set
        predictions = trainer.predict(val_dataset)
        # Convert logits to predicted class labels
        y_pred = np.argmax(predictions.predictions, axis=1)
        # Extract true labels
        y_true = predictions.label_ids

        # Print detailed classification report
        logger.info("Classification Report:")
        print(classification_report(y_true, y_pred))

        # Generate confusion matrix
        logger.info("Generating confusion matrix...")
        cm = confusion_matrix(y_true, y_pred)

        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(self.num_labels),
                    yticklabels=range(self.num_labels))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save confusion matrix plot
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_path = os.path.join(output_dir, f'{current_time}_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {cm_path}")
        plt.close()

        # Print confusion matrix statistics
        logger.info(f"Confusion Matrix:\n{cm}")

        # Calculate per-class accuracy from confusion matrix
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        logger.info("Per-class accuracy:")
        for i, acc in enumerate(per_class_accuracy):
            logger.info(f"  Class {i}: {acc:.4f}")

        return predictions

    def run_full_pipeline(self, output_dir="./results", num_epochs=3, batch_size=16, learning_rate=2e-5, debug_samples=None):
        """
        Run the complete training pipeline from data loading to evaluation.

        Args:
            output_dir: Directory to save trained model
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            debug_samples: Number of samples to use for debugging (None for all data)

        Returns:
            Trainer: Trained Trainer object
        """
        try:
            # Load and prepare data with optional debugging sample limit
            train_df, val_df = self.load_and_prepare_data(debug_samples=debug_samples)

            # Initialize model and tokenizer
            self.initialize_model_and_tokenizer()

            # Prepare datasets for training
            train_dataset, val_dataset = self.prepare_datasets(train_df, val_df)

            # Train the model
            trainer = self.train(
                train_dataset, val_dataset,
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            # Evaluate the model
            self.evaluate_model(trainer, val_dataset, output_dir)

            logger.info("Pipeline completed successfully!")
            return trainer

        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            raise

def main():
    """
    Main function to parse command-line arguments and run training pipeline.
    """
    parser = argparse.ArgumentParser(description='Train CodeBERT on SemEval-2026-Task13')
    parser.add_argument('--task', choices=['A', 'B', 'C'], default='A', help='Task subset to use')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--debug_samples', type=int, default=None, help='Number of samples to use for debugging (default: all data)')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize trainer with specified configuration
    trainer = CodeBERTTrainer(
        task_subset=args.task,
        max_length=args.max_length
    )

    # Run full training pipeline with debugging sample limit if specified
    trainer.run_full_pipeline(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        debug_samples=args.debug_samples
    )

if __name__ == "__main__":
    main()