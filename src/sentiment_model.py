from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

def create_sentiment_pipeline():
    """
    یک pipeline تحلیل احساسات با استفاده از مدل DistilBERT فارسی ایجاد می‌کند.
    """
    # استفاده از مدل بهینه‌سازی شده برای فارسی
    # این مدل برچسب‌های 'POS' و 'NEG' را برمی‌گرداند
    model_path = "HooshvareLab/distilbert-fa-zwnj-base-sentiment-dksf"
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_path,
        tokenizer=model_path
    )
    return sentiment_pipeline

def create_langchain_processor(pipeline):
    """
    یک پردازشگر LangChain برای اجرای pipeline روی نظرات ایجاد می‌کند.
    """
    # تبدیل pipeline ترنسفورمرز به یک شیء سازگار با LangChain
    llm_pipeline = HuggingFacePipeline(pipeline=pipeline)
    
    # تعریف یک قالب ساده برای ارسال نظر به مدل
    prompt = PromptTemplate(
        input_variables=["comment"],
        template="تحلیل احساسات نظر زیر:\n{comment}"
    )
    
    # ایجاد زنجیره برای پردازش
    chain = LLMChain(llm=llm_pipeline, prompt=prompt)
    return chain

if __name__ == '__main__':
    # این بخش برای تست مستقل ماژول است
    test_comment = "این غذا واقعا بی‌نظیر و خوشمزه بود"
    
    # ۱. ساخت پایپ‌لاین
    p = create_sentiment_pipeline()
    
    # ۲. تحلیل مستقیم با پایپ‌لاین
    result = p(test_comment)
    print(f"نتیجه مستقیم از Transformers: {result}") # [{'label': 'POS', 'score': 0.99...}]

    # ۳. ساخت پردازشگر LangChain
    chain = create_langchain_processor(p)
    
    # ۴. تحلیل با LangChain
    lc_result = chain.run(test_comment)
    print(f"نتیجه از LangChain: {lc_result}")