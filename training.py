from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def tune_model(df, model_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["review"], padding=True, truncation=True)

    train_texts, val_texts, train_labels, val_labels = train_test_split(df["review"], df["label"], test_size=0.2)

    train_dataset = Dataset.from_dict({"review": train_texts.tolist(), "label": train_labels.tolist()}).map(tokenize, batched=True)
    val_dataset = Dataset.from_dict({"review": val_texts.tolist(), "label": val_labels.tolist()}).map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,   # tunable
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer
