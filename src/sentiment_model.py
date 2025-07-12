from transformers import pipeline
import torch
import os

def create_sentiment_pipeline():
    """
    مدل DistilBERT شخصی‌سازی شده و آموزش‌دیده را از پوشه محلی بارگذاری می‌کند.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "distilbert-fa-sentiment-fine-tuned")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"پوشه مدل آموزش‌دیده در مسیر '{model_path}' یافت نشد! "
            "لطفاً ابتدا اسکریپت train.py را اجرا کنید."
        )

    print(f"✅ در حال بارگذاری مدل شخصی‌سازی شده از: {model_path}")
    
    # تشخیص و تخصیص دستگاه (GPU یا CPU)
    device_index = 0 if torch.cuda.is_available() else -1
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_path,
        device=device_index
    )
    return sentiment_pipeline