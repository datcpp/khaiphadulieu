import pandas as pd
import spacy
from nltk.corpus import stopwords
import re
import nltk

# Tải stopwords nếu chưa có
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load mô hình NLP tiếng Anh
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    # Loại bỏ ký tự đặc biệt, số, giữ lại chữ cái
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()

    # Phân tích bằng spaCy
    doc = nlp(text)

    tokens = []
    for token in doc:
        # Bỏ stopword, tên riêng (PROPN), từ vô nghĩa, khoảng trắng
        if token.text in stop_words or token.is_stop or token.pos_ == "PROPN" or token.is_space:
            continue
        # Chỉ giữ các từ loại quan trọng
        if token.pos_ in ["NOUN", "VERB", "ADJ"]:
            lemma = token.lemma_.strip()
            if lemma:
                tokens.append(lemma)

    return " ".join(tokens)

# Load dữ liệu
df = pd.read_csv("processed_data.csv")  # file gốc chứa cột plot
df['processed_plot'] = df['plot'].apply(preprocess)

# Lưu lại dữ liệu đã xử lý
df.to_csv("processed_data.csv", index=False)

print("✅ Hoàn tất tiền xử lý! File saved as processed_data.csv")
