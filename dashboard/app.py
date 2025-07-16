import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display

# --- اصلاحیه اصلی برای رفع خطا ---
# این کد مسیر اصلی پروژه را به پایتون معرفی می‌کند
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.sentiment_model import create_sentiment_pipeline

# --- توابع کمکی ---

@st.cache_resource
def load_model():
    """مدل تحلیل احساسات را یک بار بارگذاری و کش می‌کند."""
    # این تابع بدون تغییر باقی می‌ماند
    st.info("در حال بارگذاری مدل تحلیل احساسات... لطفاً شکیبا باشید.")
    pipeline = create_sentiment_pipeline()
    st.success("مدل با موفقیت بارگذاری شد!")
    return pipeline

def analyze_sentiment(pipeline, text):
    """احساسات یک متن را تحلیل کرده و برچسب را به فارسی ترجمه می‌کند."""
    # این تابع بدون تغییر باقی می‌ماند
    if not isinstance(text, str) or not text.strip():
        return "متن نامعتبر"
    try:
        result = pipeline(text)[0]
        label = "منفی" if result['label'] == 'POSITIVE' else "مثبت"
        score = result['score']
        return f"{label} (امتیاز: {score:.2f})"
    except Exception as e:
        st.error(f"خطا در تحلیل: {e}")
        return "خطا در تحلیل"

# --- UI داشبورد ---

st.set_page_config(layout="wide", page_title="داشبورد تحلیل احساسات")

# --- سایدبار برای کنترل‌ها ---
with st.sidebar:
    st.header("کنترل پنل تحلیل")
    
    st.subheader("🔎 تحلیل یک نظر دلخواه")
    user_input = st.text_area("نظر خود را اینجا وارد کنید:", "کیفیت غذا عالی بود و به موقع رسید.", height=120)
    analyze_button = st.button("تحلیل کن")

    st.subheader("📂 تحلیل گروهی نظرات")
    uploaded_file = st.file_uploader("یک فایل CSV یا TSV آپلود کنید")

# --- صفحه اصلی برای نمایش نتایج ---
st.title("📊 داشبورد تحلیل احساسات نظرات کاربران")

# بارگذاری مدل
sentiment_pipeline = load_model()

# بخش تحلیل تکی
if analyze_button and user_input.strip():
    with st.spinner('در حال تحلیل...'):
        result = analyze_sentiment(sentiment_pipeline, user_input)
        st.subheader("نتیجه تحلیل نظر شما:")
        if "مثبت" in result:
            st.success(result)
        else:
            st.error(result)

# بخش تحلیل گروهی
if uploaded_file is not None:
    try:
        # خواندن فایل بر اساس فرمت صحیح (جدا شده با تب)
        df = pd.read_csv(uploaded_file, sep='\t')
        if len(df.columns) < 2:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=',')
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

    if 'comment' not in df.columns:
        st.error("فایل آپلود شده باید ستونی به نام 'comment' داشته باشد.")
    else:
        st.info(f"فایل با موفقیت بارگذاری شد. تعداد کل نظرات: {len(df)}")
        df.dropna(subset=['comment'], inplace=True)
        df['comment'] = df['comment'].astype(str)
        
        sample_size = min(100, len(df))
        df_sample = df.head(sample_size).copy()
        
        with st.spinner(f'در حال تحلیل {len(df_sample)} نظر...'):
            results = [analyze_sentiment(sentiment_pipeline, comment) for comment in df_sample['comment']]
            df_sample['sentiment_result'] = results
            df_sample['sentiment_label'] = df_sample['sentiment_result'].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else 'نامشخص')

        st.markdown("---")
        st.subheader(f"آمار کلی از تحلیل {sample_size} نظر اول")
        
        # بخش کارت‌های متریک
        sentiment_counts = df_sample['sentiment_label'].value_counts()
        positive_count = sentiment_counts.get('مثبت', 0)
        negative_count = sentiment_counts.get('منفی', 0)
        total_count = positive_count + negative_count
        
        col1, col2, col3 = st.columns(3)
        col1.metric("نظرات مثبت 👍", positive_count)
        col2.metric("نظرات منفی 👎", negative_count)
        col3.metric("درصد رضایت", f"{(positive_count / total_count * 100):.1f}%" if total_count > 0 else "N/A")

        st.markdown("---")
        
        # بخش نمودار و جدول
        col1, col2 = st.columns([1, 2])
        with col1:
            if not sentiment_counts.empty:
                fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=.5,
                                             marker_colors=['#2ECC71', '#E74C3C'])])
                fig.update_layout(title_text="توزیع احساسات", legend_title_text='احساس', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("جدول نتایج تحلیل:")
            st.dataframe(df_sample[['comment', 'sentiment_result']], height=400)

        st.markdown("---")
        
        # بخش ابر کلمات
        st.subheader("ابر کلمات کلیدی در نظرات")
        font_path = 'Vazir.ttf'
        if not os.path.exists(font_path):
            st.warning("فایل فونت فارسی Vazir.ttf یافت نشد. ابر کلمات نمایش داده نمی‌شود.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.write("کلمات پرتکرار در نظرات **مثبت**")
                positive_text = " ".join(comment for comment in df_sample[df_sample['sentiment_label'] == 'مثبت']['comment'])
                if positive_text:
                    reshaped_text = arabic_reshaper.reshape(positive_text)
                    bidi_text = get_display(reshaped_text)
                    wordcloud = WordCloud(font_path=font_path, background_color="#0E1117", colormap='Greens', width=800, height=400).generate(bidi_text)
                    st.image(wordcloud.to_array(), use_column_width=True)
            
            with col2:
                st.write("کلمات پرتکرار در نظرات **منفی**")
                negative_text = " ".join(comment for comment in df_sample[df_sample['sentiment_label'] == 'منفی']['comment'])
                if negative_text:
                    reshaped_text = arabic_reshaper.reshape(negative_text)
                    bidi_text = get_display(reshaped_text)
                    wordcloud = WordCloud(font_path=font_path, background_color="#0E1117", colormap='Reds', width=800, height=400).generate(bidi_text)
                    st.image(wordcloud.to_array(), use_column_width=True)