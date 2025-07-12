import pandas as pd

def load_and_process_data(file_path):
    """
    داده‌ها را از فایل CSV بارگذاری کرده، ستون‌های مورد نیاز را انتخاب
    و نظرات بدون برچسب را حذف می‌کند.
    """
    df = pd.read_csv(file_path)
    
    # انتخاب ستون‌های 'comment' و 'label_id'
    # label_id: 0 -> منفی, 1 -> مثبت
    df = df[['comment', 'label_id']].copy()
    
    # حذف سطرهایی که نظر یا برچسب ندارند
    df.dropna(subset=['comment', 'label_id'], inplace=True)
    
    # تبدیل label_id به نوع صحیح
    df['label_id'] = df['label_id'].astype(int)
    
    # برای سادگی، فقط نظرات مثبت و منفی را نگه می‌داریم
    df = df[df['label_id'].isin([0, 1])]
    
    # تبدیل برچسب عددی به متنی
    df['sentiment'] = df['label_id'].apply(lambda x: 'positive' if x == 1 else 'negative')
    
    print(f"تعداد {len(df)} نظر پردازش شد.")
    return df[['comment', 'sentiment']]

if __name__ == '__main__':
    # این بخش برای تست مستقل ماژول است
    input_path = '../data/raw/Snappfood - Sentiment Analysis.csv'
    output_path = '../data/processed/processed_comments.csv'
    
    processed_df = load_and_process_data(input_path)
    processed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"داده‌های پردازش شده در {output_path} ذخیره شد.")