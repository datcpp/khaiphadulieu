import pandas as pd
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pickle

# Định nghĩa tên các chủ đề sát hơn với thực tế dựa trên từ khóa và phân tích phim
topic_names = {
    0: "Chiến tranh & Quân sự",
    1: "Khoa học viễn tưởng & Vũ trụ",
    2: "Tình yêu & Gia đình",
    3: "Tuổi trẻ & Học đường",
    4: "Trinh thám & Tội phạm",
    5: "Hoạt hình & Thiếu nhi",
    6: "Kinh dị & Siêu nhiên",
    7: "Bí ẩn & Khám phá",
    8: "Phiêu lưu & Khám phá",
    9: "Hành động & Giải cứu"
}

def main():
    # Load dữ liệu
    df = pd.read_csv("../data/processed_data.csv")
    texts = [doc.split() for doc in df['processed_plot']]

    # Tạo dictionary và corpus
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Huấn luyện mô hình LDA
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=10,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    # In 10 chủ đề đầu tiên với tên
    for idx, topic in lda_model.print_topics(-1):
        topic_name = topic_names.get(idx, "Chủ đề không xác định")  # Lấy tên chủ đề từ dictionary
        print(f"Chủ đề {idx} ({topic_name}): {topic}")

    # Lưu mô hình
    lda_model.save("lda_model.gensim")
    dictionary.save("dictionary.gensim")
    pickle.dump(corpus, open("corpus.pkl", "wb"))

    # Đánh giá bằng coherence score
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"Coherence Score: {coherence_score:.4f}")

if __name__ == "__main__":
    main()
