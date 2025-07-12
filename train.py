import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os
import numpy as np
import evaluate

def train_sentiment_model():
    # --- بخش ۱: آماده‌سازی داده‌ها ---
    print("--- مرحله ۱: بارگذاری و آماده‌سازی داده‌ها ---")
    data_path = os.path.join("data", "raw", "Snappfood - Sentiment Analysis.csv")
    df = pd.read_csv(data_path, sep='\t')
    required_cols = ['comment', 'label_id']
    df_clean = df[required_cols].copy()
    df_clean.dropna(inplace=True)
    df_clean = df_clean[df_clean['label_id'].isin([0, 1])]
    df_clean['label_id'] = df_clean['label_id'].astype(int)
    df_clean.rename(columns={'comment': 'text', 'label_id': 'label'}, inplace=True)
    train_df, eval_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['label'])
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    dataset_dict = DatasetDict({'train': train_dataset, 'eval': eval_dataset})
    print(f"داده‌ها آماده شد.")

    # --- بخش ۲: توکنایز کردن ---
    print("\n--- مرحله ۲: بارگذاری توکنایزر ---")
    base_model_id = "HooshvareLab/distilbert-fa-zwnj-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    print("توکنایز کردن داده‌ها با موفقیت انجام شد.")

    # --- بخش ۳: تعریف صریح نگاشت برچسب‌ها ---
    print("\n--- مرحله ۳: تعریف صریح نگاشت برچسب‌ها ---")
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # --- بخش ۴: بارگذاری مدل با نگاشت جدید ---
    print("\n--- مرحله ۴: بارگذاری مدل با نگاشت صریح ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id, 
        num_labels=2, 
        id2label=id2label, 
        label2id=label2id
    )

    # --- بخش ۵: آموزش مدل ---
    print("\n--- مرحله ۵: شروع فرآیند آموزش ---")
    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # پارامترهای بهینه‌سازی شده برای حداکثر دقت
    training_args = TrainingArguments(
        output_dir="./training_results",
        num_train_epochs=3,                 # افزایش به ۳ برای یادگیری بیشتر
        learning_rate=2e-5,                 # نرخ یادگیری کمتر برای همگرایی دقیق‌تر
        per_device_train_batch_size=32,     # استفاده بهینه از GPU
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    print("✅ آموزش با موفقیت به پایان رسید!")

    # --- بخش ۶: ذخیره مدل نهایی ---
    print("\n--- مرحله ۶: ذخیره مدل نهایی ---")
    final_model_path = os.path.join("models", "distilbert-fa-sentiment-fine-tuned")
    trainer.save_model(final_model_path)
    print(f"✅ مدل و توکنایزر در پوشه '{final_model_path}' ذخیره شدند.")

if __name__ == "__main__":
    train_sentiment_model()