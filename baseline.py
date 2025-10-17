import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging for tracking training progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineTrainer:
    """
    Baseline trainer for code classification using pretrained language models.
    Adapted from HuggingFace text classification tutorial for SemEval-2026-Task13.
    """

    def __init__(self, task_subset='A', max_length=512, model_name="distilbert/distilbert-base-uncased", use_wandb=True):
        # Task subset identifier (A, B, or C) for dataset loading
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
        # Mapping from label ID to label name
        self.id2label = None
        # Mapping from label name to label ID
        self.label2id = None
        # Device for training (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Whether to use Weights & Biases for experiment tracking
        self.use_wandb = use_wandb
        logger.info(f"Using device: {self.device}")
        logger.info(f"W&B tracking: {'enabled' if use_wandb else 'disabled'}")

    def load_semeval_data(self, debug_samples=None):
        """
        Load the SemEval-2026-Task13 dataset from Hugging Face.

        Args:
            debug_samples: Number of samples to use for debugging (None for all data)

        Returns:
            tuple: (dataset, train_df, val_df) - Full dataset and split DataFrames
        """
        logger.info(f"Loading SemEval-2026-Task13 dataset subset {self.task_subset}...")

        try:
            # Load dataset from Hugging Face hub (similar to load_dataset("imdb"))
            dataset = load_dataset("DaniilOr/SemEval-2026-Task13", self.task_subset)

            # Extract training split from dataset
            train_data = dataset['train']
            logger.info(f"Loaded {len(train_data)} training samples")

            # Convert to pandas DataFrame for easier data manipulation
            df = train_data.to_pandas()

            logger.info(f"Dataset columns: {df.columns.tolist()}")
            logger.info(f"Sample data:\n{df.head()}")

            # Validate that required columns exist in dataset
            if 'code' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must contain 'code' and 'label' columns")

            # Remove rows with missing values in critical columns
            df = df.dropna(subset=['code', 'label'])

            # Apply debug sample limit if specified
            if debug_samples is not None:
                logger.info(f"DEBUG MODE: Using only {debug_samples} samples")
                df = df.head(debug_samples)

            # Convert labels to integers for classification
            df['label'] = df['label'].astype(int)
            # Determine number of unique labels for model configuration
            self.num_labels = df['label'].nunique()

            # Create label mappings (similar to id2label and label2id in tutorial)
            # Convert numpy int64 to Python int for JSON serialization
            unique_labels = sorted(df['label'].unique())
            self.id2label = {int(i): f"CLASS_{int(i)}" for i in unique_labels}
            self.label2id = {f"CLASS_{int(i)}": int(i) for i in unique_labels}

            logger.info(f"Number of unique labels: {self.num_labels}")
            logger.info(f"Label range: {df['label'].min()} to {df['label'].max()}")
            logger.info(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
            logger.info(f"Label mappings:\n  id2label: {self.id2label}\n  label2id: {self.label2id}")

            # Split data into train (80%) and validation (20%)
            train_size = int(0.8 * len(df))
            train_df = df[:train_size].reset_index(drop=True)
            val_df = df[train_size:].reset_index(drop=True)

            logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

            # Create HuggingFace Dataset objects (similar to imdb["train"] and imdb["test"])
            train_dataset = Dataset.from_pandas(train_df[['code', 'label']])
            val_dataset = Dataset.from_pandas(val_df[['code', 'label']])

            # Create a dataset dictionary similar to the tutorial
            dataset_dict = {
                'train': train_dataset,
                'validation': val_dataset
            }

            return dataset_dict, train_df, val_df

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def initialize_tokenizer(self):
        """
        Initialize the pretrained tokenizer for text processing.
        """
        logger.info(f"Initializing tokenizer from {self.model_name}...")

        # Load pretrained tokenizer (similar to AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        logger.info(f"Tokenizer initialized successfully")

    def preprocess_function(self, examples):
        """
        Tokenize code examples for model input.
        Adapted from tutorial's preprocess_function for "text" field to "code" field.

        Args:
            examples: Batch of examples containing 'code' field

        Returns:
            dict: Tokenized inputs with input_ids, attention_mask
        """
        # Tokenize the 'code' field (changed from examples["text"])
        return self.tokenizer(
            examples['code'],  # Changed from examples["text"]
            truncation=True,
            max_length=self.max_length
        )

    def tokenize_datasets(self, dataset_dict):
        """
        Apply tokenization to all datasets in the dictionary.

        Args:
            dataset_dict: Dictionary containing 'train' and 'validation' datasets

        Returns:
            dict: Tokenized datasets
        """
        logger.info("Tokenizing datasets...")

        # Apply tokenization using map (similar to tokenized_imdb = imdb.map(preprocess_function, batched=True))
        tokenized_datasets = {}
        for split_name, dataset in dataset_dict.items():
            tokenized_datasets[split_name] = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=['code']  # Remove original code column after tokenization
            )
            logger.info(f"Tokenized {split_name} dataset: {len(tokenized_datasets[split_name])} samples")

        # Rename 'label' to 'labels' (required by Trainer API)
        for split_name in tokenized_datasets:
            tokenized_datasets[split_name] = tokenized_datasets[split_name].rename_column('label', 'labels')

        return tokenized_datasets

    def setup_metrics(self):
        """
        Load evaluation metrics for model performance assessment.

        Returns:
            evaluate.EvaluationModule: Accuracy metric loader
        """
        logger.info("Loading accuracy metric...")

        # Load accuracy metric (same as tutorial: accuracy = evaluate.load("accuracy"))
        accuracy_metric = evaluate.load("accuracy")

        return accuracy_metric

    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics for model predictions.
        Identical to tutorial's compute_metrics function.

        Args:
            eval_pred: Tuple of (predictions, labels)

        Returns:
            dict: Dictionary containing accuracy
        """
        # Extract predictions and true labels from evaluation output
        predictions, labels = eval_pred
        # Convert logits to predicted class labels using argmax
        predictions = np.argmax(predictions, axis=1)

        # Compute accuracy using the evaluate library
        accuracy_metric = evaluate.load("accuracy")
        return accuracy_metric.compute(predictions=predictions, references=labels)

    def initialize_model(self):
        """
        Initialize the pretrained model for sequence classification.
        Adapted from tutorial to use dynamic label mappings.
        """
        logger.info(f"Initializing model from {self.model_name}...")

        # Load pretrained model for sequence classification with custom label mappings
        # (similar to AutoModelForSequenceClassification.from_pretrained with id2label and label2id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )

        # Move model to appropriate device (GPU/CPU)
        self.model.to(self.device)

        logger.info(f"Model initialized with {self.num_labels} labels on {self.device}")

    def train_model(self, tokenized_datasets, output_dir="./baseline_results",
                   num_epochs=2, batch_size=16, learning_rate=2e-5):
        """
        Train the model using standard fine-tuning approach from tutorial.
        Adapted training arguments to not push to hub.

        Args:
            tokenized_datasets: Dictionary with 'train' and 'validation' datasets
            output_dir: Directory to save model checkpoints and final model
            num_epochs: Number of training epochs
            batch_size: Batch size for training and evaluation
            learning_rate: Learning rate for optimizer

        Returns:
            Trainer: Trained Trainer object
        """
        logger.info("Starting model training...")

        # Create data collator for dynamic padding (same as tutorial)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Determine report_to based on wandb usage
        report_to_services = ["wandb", "tensorboard"] if self.use_wandb else ["tensorboard"]

        # Configure training arguments (adapted from tutorial, removed push_to_hub)
        training_args = TrainingArguments(
            output_dir=output_dir,  # Directory for saving model checkpoints
            learning_rate=learning_rate,  # Learning rate for optimizer
            per_device_train_batch_size=batch_size,  # Batch size per device for training
            per_device_eval_batch_size=batch_size,  # Batch size per device for evaluation
            num_train_epochs=num_epochs,  # Total number of training epochs
            weight_decay=0.01,  # Weight decay coefficient for regularization
            eval_strategy="epoch",  # Evaluate model at the end of each epoch
            save_strategy="epoch",  # Save checkpoint at the end of each epoch
            load_best_model_at_end=True,  # Load best model at end of training
            logging_dir=os.path.join(output_dir, 'logs'),  # Directory for TensorBoard logs
            logging_steps=100,  # Log training metrics every N steps
            report_to=report_to_services,  # Report metrics to W&B and/or TensorBoard
            # Note: Removed push_to_hub=True from tutorial
        )

        # Initialize Trainer with model, datasets, and training configuration
        # (same structure as tutorial)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Start training process (same as tutorial: trainer.train())
        trainer.train()

        # Save final trained model (no push_to_hub as we're not using HF Hub)
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Training completed. Model saved to {output_dir}")

        return trainer

    def evaluate_model(self, trainer, tokenized_datasets, output_dir="./baseline_results"):
        """
        Evaluate the trained model on validation dataset and generate confusion matrix.

        Args:
            trainer: Trained Trainer object
            tokenized_datasets: Dictionary containing tokenized datasets
            output_dir: Directory to save confusion matrix plot

        Returns:
            dict: Evaluation metrics and predictions
        """
        logger.info("Evaluating model on validation set...")

        # Generate predictions on validation set
        predictions = trainer.predict(tokenized_datasets["validation"])
        # Convert logits to predicted class labels
        y_pred = np.argmax(predictions.predictions, axis=1)
        # Extract true labels from predictions
        y_true = predictions.label_ids

        # Print detailed evaluation metrics
        logger.info("\n" + "="*50)
        logger.info("Evaluation Metrics:")
        logger.info("="*50)
        logger.info(f"Metrics: {predictions.metrics}")
        logger.info("="*50)

        # Generate and log confusion matrix
        self._log_confusion_matrix(y_true, y_pred, output_dir)

        return {
            'predictions': y_pred,
            'labels': y_true,
            'metrics': predictions.metrics
        }

    def _log_confusion_matrix(self, y_true, y_pred, output_dir):
        """
        Generate confusion matrix and log to wandb and save as image.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dir: Directory to save confusion matrix plot
        """
        logger.info("Generating confusion matrix...")

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f"Class {i}" for i in range(self.num_labels)],
                    yticklabels=[f"Class {i}" for i in range(self.num_labels)])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save confusion matrix plot
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_path = os.path.join(output_dir, f'confusion_matrix_{current_time}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {cm_path}")

        # Log to wandb if enabled
        if self.use_wandb:
            # Log confusion matrix as image
            wandb.log({"confusion_matrix": wandb.Image(cm_path)})

            # Log confusion matrix as table for interactive viewing
            class_names = [f"Class {i}" for i in range(self.num_labels)]
            wandb.log({"confusion_matrix_table": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )})

            logger.info("Confusion matrix logged to W&B")

        plt.close()

        # Print confusion matrix statistics
        logger.info(f"Confusion Matrix:\n{cm}")

        # Calculate per-class accuracy from confusion matrix
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        logger.info("Per-class accuracy:")
        for i, acc in enumerate(per_class_accuracy):
            logger.info(f"  Class {i}: {acc:.4f}")
            # Log per-class metrics to wandb
            if self.use_wandb:
                wandb.log({f"per_class_accuracy/class_{i}": acc})

    def inference_example(self, code_text, output_dir="./baseline_results"):
        """
        Perform inference on a single code example.
        Adapted from tutorial's inference section.

        Args:
            code_text: String containing code to classify
            output_dir: Directory where trained model is saved

        Returns:
            dict: Prediction results with label and score
        """
        logger.info("Running inference on example code...")

        # Load tokenizer and model from saved checkpoint
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        model.to(self.device)
        model.eval()

        # Tokenize input text
        inputs = tokenizer(code_text, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get predicted class ID and label
        predicted_class_id = logits.argmax().item()
        predicted_label = model.config.id2label[predicted_class_id]

        # Calculate confidence score
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence_score = probabilities[0][predicted_class_id].item()

        logger.info(f"Predicted label: {predicted_label} (class {predicted_class_id})")
        logger.info(f"Confidence score: {confidence_score:.4f}")

        return {
            'label': predicted_label,
            'class_id': predicted_class_id,
            'score': confidence_score
        }

    def run_pipeline(self, output_dir="./baseline_results", num_epochs=2,
                    batch_size=16, learning_rate=2e-5, debug_samples=None,
                    wandb_project="semeval-code-classification", wandb_run_name=None):
        """
        Run the complete training pipeline from data loading to evaluation.
        This follows the tutorial structure exactly.

        Args:
            output_dir: Directory to save trained model
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            debug_samples: Number of samples to use for debugging (None for all data)
            wandb_project: W&B project name for experiment tracking
            wandb_run_name: W&B run name (auto-generated if None)

        Returns:
            tuple: (trainer, evaluation_results)
        """
        try:
            # Initialize W&B if enabled
            if self.use_wandb:
                # Generate run name if not provided
                if wandb_run_name is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    wandb_run_name = f"{self.model_name.split('/')[-1]}_task{self.task_subset}_{timestamp}"

                # Initialize W&B run with configuration
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "model_name": self.model_name,
                        "task_subset": self.task_subset,
                        "max_length": self.max_length,
                        "num_epochs": num_epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "debug_samples": debug_samples,
                    }
                )
                logger.info(f"W&B run initialized: {wandb_run_name}")

            # Step 1: Load dataset (similar to imdb = load_dataset("imdb"))
            dataset_dict, train_df, val_df = self.load_semeval_data(debug_samples=debug_samples)

            # Log dataset statistics to W&B
            if self.use_wandb:
                wandb.log({
                    "dataset/train_size": len(train_df),
                    "dataset/val_size": len(val_df),
                    "dataset/num_labels": self.num_labels,
                })

            # Step 2: Initialize tokenizer (similar to tokenizer = AutoTokenizer.from_pretrained(...))
            self.initialize_tokenizer()

            # Step 3: Tokenize datasets (similar to tokenized_imdb = imdb.map(preprocess_function, batched=True))
            tokenized_datasets = self.tokenize_datasets(dataset_dict)

            # Step 4: Setup metrics (similar to accuracy = evaluate.load("accuracy"))
            accuracy_metric = self.setup_metrics()

            # Step 5: Initialize model (similar to model = AutoModelForSequenceClassification.from_pretrained(...))
            self.initialize_model()

            # Step 6: Train model (similar to trainer.train())
            trainer = self.train_model(
                tokenized_datasets,
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            # Step 7: Evaluate model
            eval_results = self.evaluate_model(trainer, tokenized_datasets, output_dir)

            logger.info("Pipeline completed successfully!")

            # Optional: Run inference example if validation data exists
            if len(val_df) > 0:
                example_code = val_df.iloc[0]['code']
                logger.info("\nRunning inference on sample code...")
                inference_result = self.inference_example(example_code, output_dir)

                # Log inference example to W&B
                if self.use_wandb:
                    wandb.log({
                        "inference_example/predicted_class": inference_result['class_id'],
                        "inference_example/confidence": inference_result['score'],
                    })

            # Finish W&B run
            if self.use_wandb:
                wandb.finish()
                logger.info("W&B run finished")

            return trainer, eval_results

        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            # Finish W&B run even on error
            if self.use_wandb:
                wandb.finish(exit_code=1)
            raise


def main():
    """
    Main function to parse command-line arguments and run training pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Baseline trainer for SemEval-2026-Task13 code classification (HuggingFace tutorial style)'
    )
    parser.add_argument('--task', choices=['A', 'B', 'C'], default='A',
                       help='Task subset to use (A, B, or C)')
    parser.add_argument('--output_dir', default='./results',
                       help='Output directory for model and results')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate for optimizer')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length for tokenization')
    parser.add_argument('--debug_samples', type=int, default=None,
                       help='Number of samples to use for debugging (default: all data)')
    parser.add_argument('--model_name', default='distilbert/distilbert-base-uncased',
                       help='Pretrained model name from Hugging Face')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='Enable Weights & Biases logging (default: True)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', default='semeval-code-classification',
                       help='W&B project name for experiment tracking')
    parser.add_argument('--wandb_run_name', default=None,
                       help='W&B run name (auto-generated if not specified)')

    args = parser.parse_args()

    # Determine wandb usage (--no_wandb overrides --use_wandb)
    use_wandb = args.use_wandb and not args.no_wandb

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize baseline trainer with specified configuration
    trainer = BaselineTrainer(
        task_subset=args.task,
        max_length=args.max_length,
        model_name=args.model_name,
        use_wandb=use_wandb
    )

    # Run full training pipeline following tutorial structure
    trainer.run_pipeline(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        debug_samples=args.debug_samples,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )


if __name__ == "__main__":
    main()
