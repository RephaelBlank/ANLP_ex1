
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
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
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

    train_dataset = tokenized_dataset["train"]
    if args.max_train_samples != -1:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    eval_dataset = tokenized_dataset["validation"]
    if args.max_eval_samples != -1:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    predict_dataset = tokenized_dataset["test"]
    if args.max_predict_samples != -1:
        predict_dataset = predict_dataset.select(range(args.max_predict_samples))

    if args.do_train:
        run_name = f"lr{args.lr}-bs{args.batch_size}-ep{args.num_train_epochs}"
        wandb.init(project="paraphrase-detection-bert", name=run_name)

        model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

        training_args = TrainingArguments(
            output_dir="./bert_mrpc_results",
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            num_train_epochs=args.num_train_epochs,
            logging_steps=1,
            report_to="wandb",
            weight_decay=0.01
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(f"./models/{run_name}")
        wandb.finish()

    if args.do_predict:
        assert args.model_path is not None, "--model_path must be provided for prediction"

        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        model.eval()

        predictor = Trainer(
            model=model,
            processing_class=tokenizer,
        )

        predictions_output = predictor.predict(predict_dataset)
        preds = np.argmax(predictions_output.predictions, axis=-1)

        sentences1 = predict_dataset["sentence1"]
        sentences2 = predict_dataset["sentence2"]

        with open("predictions.txt", "w") as f:
            for s1, s2, pred in zip(sentences1, sentences2, preds):
                f.write(f"{s1}###{s2}###{pred}\n")


        print(f"Predictions saved to predictions.txt ({len(preds)} samples)")


if __name__ == "__main__":
    main()