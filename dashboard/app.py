import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from sentiment_model import create_sentiment_pipeline

@st.cache_resource
def load_model():
    """مدل تحلیل احساسات را بارگذاری می‌کند."""
    st.info("در حال بارگذاری مدل تحلیل احساسات... لطفاً شکیبا باشید.")
    pipeline = create_sentiment_pipeline()
    st.success("مدل با موفقیت بارگذاری شد!")
    return pipeline

def analyze_sentiment(pipeline, text):
    """احساسات یک متن را تحلیل کرده و برچسب را به فارسی ترجمه می‌کند."""
    if not isinstance(text, str) or not text.strip():
        return "متن نامعتبر"
    try:
        result = pipeline(text)[0]
        # تفسیر را برعکس می‌کنیم تا با مدل هماهنگ شود
        label = "منفی" if result['label'] == 'POSITIVE' else "مثبت"
        score = result['score']
        return f"{label} (امتیاز: {score:.2f})"
    except Exception as e:
        st.error(f"خطا در تحلیل: {e}")
        return "خطا در تحلیل"

# --- UI داشبورد ---
st.set_page_config(layout="wide", page_title="داشبورد تحلیل احساسات")
st.title("📊 داشبورد تحلیل احساسات نظرات کاربران")
st.write("این داشبورد نظرات کاربران را با استفاده از مدل DistilBERT (آموزش‌دیده) تحلیل می‌کند.")

sentiment_pipeline = load_model()

# بخش تحلیل تکی
st.header("🔎 تحلیل یک نظر دلخواه")
user_input = st.text_area("نظر خود را اینجا وارد کنید:", "کیفیت غذا عالی بود و به موقع رسید.")
if st.button("تحلیل کن"):
    with st.spinner('در حال تحلیل...'):
        result = analyze_sentiment(sentiment_pipeline, user_input)
        if result:
            st.subheader("نتیجه تحلیل:")
            if "مثبت" in result:
                st.success(result)
            else:
                st.error(result)

# بخش تحلیل گروهی از فایل
st.header("📂 تحلیل گروهی نظرات از فایل CSV")
uploaded_file = st.file_uploader("یک فایل CSV آپلود کنید (باید ستونی به نام 'comment' داشته باشد)", type=["csv"])

if uploaded_file is not None:
    # خواندن فایل بر اساس فرمت صحیح (جدا شده با تب)
    try:
        df = pd.read_csv(uploaded_file, sep='\t')
    except Exception:
        # اگر با تب نشد، با کاما امتحان کن
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

    if 'comment' not in df.columns:
        st.error("فایل CSV باید حتماً ستونی به نام 'comment' داشته باشد.")
    else:
        st.info(f"فایل با موفقیت بارگذاری شد. تعداد نظرات برای تحلیل: {len(df)}")
        
        df.dropna(subset=['comment'], inplace=True)
        df['comment'] = df['comment'].astype(str)
        
        # برای نمایش سریع‌تر، روی نمونه کوچک‌تری کار می‌کنیم
        sample_size = 100
        df_sample = df.head(sample_size).copy()
        
        with st.spinner(f'در حال تحلیل {len(df_sample)} نظر...'):
            results = [analyze_sentiment(sentiment_pipeline, comment) for comment in df_sample['comment']]
            df_sample['sentiment_result'] = results
            df_sample['sentiment_label'] = df_sample['sentiment_result'].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else 'نامشخص')

        st.subheader(f"نتایج تحلیل {sample_size} نظر اول:")
        st.dataframe(df_sample)

        st.subheader("نمودار توزیع احساسات")
        sentiment_counts = df_sample['sentiment_label'].value_counts()
        
        if not sentiment_counts.empty:
            st.bar_chart(sentiment_counts)
        else:
            st.warning("هیچ نظری برای نمایش در نمودار وجود ندارد.")