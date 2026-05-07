
import argparse
import wandb
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    wandb.login()

    # 1. Define the evaluation metric (Accuracy)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 2. Load data and tokenizer
    ds = load_dataset("nyu-mll/glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def preprocess_function(examples):
        # Truncate to max length allowed by the model
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    tokenized_dataset = ds.map(preprocess_function, batched=True)

    # 1. Initialize Weights & Biases for tracking
    wandb.init(project="paraphrase-detection-bert", name="run-1-lr2e5-batch16")

    # 2. Re-initialize the model from scratch for this experiment!
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

    # 3. Define Training Arguments (Hyperparameters to experiment with)
    training_args = TrainingArguments(
        output_dir="./bert_mrpc_results",
        eval_strategy="epoch",    # Evaluate on validation set at the end of each epoch
        save_strategy="epoch",          # Save checkpoints to evaluate on the test set later
        learning_rate=2e-5,             # Hyperparameter to tune
        per_device_train_batch_size=8, # Hyperparameter to tune
        per_device_eval_batch_size=16,
        num_train_epochs=3,             # Must be <= 5 according to instructions
        logging_steps=1,                # Required: log training loss every step
        report_to="wandb",              # Required: track training with Weights & Biases
        weight_decay=0.01
    )

    # 4. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 5. Start Training
    trainer.train()

    # 6. Close the W&B run after training completes
    wandb.finish()


if __name__ == "__main__":
    main()