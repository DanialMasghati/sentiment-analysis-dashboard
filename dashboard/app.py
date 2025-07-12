import streamlit as st
import pandas as pd
import sys
import os

# افزودن مسیر src به path پایتون برای وارد کردن ماژول‌ها
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sentiment_model import create_sentiment_pipeline

# برای جلوگیری از بارگذاری مجدد مدل در هر بار تعامل با داشبورد
@st.cache_resource
def load_model():
    """مدل تحلیل احساسات را بارگذاری می‌کند."""
    st.info("در حال بارگذاری مدل تحلیل احساسات... لطفاً شکیبا باشید.")
    pipeline = create_sentiment_pipeline()
    st.success("مدل با موفقیت بارگذاری شد!")
    return pipeline

def analyze_sentiment(pipeline, text):
    """احساسات یک متن را با استفاده از pipeline تحلیل می‌کند."""
    if not text.strip():
        return None
    try:
        # پایپ‌لاین لیستی از دیکشنری‌ها را برمی‌گرداند
        result = pipeline(text)[0]
        label = "مثبت" if result['label'] == 'POS' else "منفی"
        score = result['score']
        return f"{label} (امتیاز: {score:.2f})"
    except Exception as e:
        st.error(f"خطا در تحلیل: {e}")
        return None

# --- UI داشبورد ---
st.set_page_config(layout="wide", page_title="داشبورد تحلیل احساسات")

st.title("📊 داشبورد تحلیل احساسات نظرات کاربران")
st.write("این داشبورد نظرات کاربران را با استفاده از مدل DistilBERT فارسی تحلیل می‌کند.")

# بارگذاری مدل
sentiment_pipeline = load_model()

# بخش تحلیل تکی
st.header("🔎 تحلیل یک نظر دلخواه")
user_input = st.text_area("نظر خود را اینجا وارد کنید:", "از کیفیت خدمات شما بسیار راضی هستم.")
if st.button("تحلیل کن"):
    result = analyze_sentiment(sentiment_pipeline, user_input)
    if result:
        st.subheader("نتیجه تحلیل:")
        st.success(result)

# بخش تحلیل گروهی از فایل
st.header("📂 تحلیل گروهی نظرات از فایل CSV")
uploaded_file = st.file_uploader("یک فایل CSV آپلود کنید (باید ستونی به نام 'comment' داشته باشد)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'comment' not in df.columns:
        st.error("فایل CSV باید حتماً ستونی به نام 'comment' داشته باشد.")
    else:
        st.info(f"فایل با موفقیت بارگذاری شد. تعداد نظرات: {len(df)}")
        
        # تحلیل نظرات
        # برای جلوگیری از کندی، فقط ۱۰۰ نظر اول را تحلیل می‌کنیم
        df_sample = df.head(100).copy()
        df_sample.dropna(subset=['comment'], inplace=True)

        # تحلیل و ذخیره نتایج
        results = [analyze_sentiment(sentiment_pipeline, str(comment)) for comment in df_sample['comment']]
        df_sample['sentiment_result'] = results
        
        # استخراج برچسب اصلی (مثبت/منفی)
        df_sample['sentiment_label'] = df_sample['sentiment_result'].apply(lambda x: x.split(' ')[0] if x else 'نامشخص')

        st.subheader("نتایج تحلیل:")
        st.dataframe(df_sample)

        # نمایش نمودار
        st.subheader("نمودار توزیع احساسات")
        sentiment_counts = df_sample['sentiment_label'].value_counts()
        
        if not sentiment_counts.empty:
            st.bar_chart(sentiment_counts)
        else:
            st.warning("هیچ نظری برای نمایش در نمودار وجود ندارد.")