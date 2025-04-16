import streamlit as st
import gensim
import pickle
import pandas as pd
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Load mô hình LDA và các đối tượng cần thiết
lda_model = gensim.models.LdaModel.load("lda_model.gensim")
dictionary = corpora.Dictionary.load("dictionary.gensim")
corpus = pickle.load(open("corpus.pkl", "rb"))
df = pd.read_csv("data/processed_data.csv")

# Danh sách tên các chủ đề (dựa trên mô hình LDA đã huấn luyện)
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

# Cấu hình trang Streamlit
st.set_page_config(page_title="Topic Modeling với LDA", layout="wide", initial_sidebar_state="expanded")

st.title("🎬 Topic Modeling các phim với LDA")

# Hiển thị các chủ đề
st.subheader("📌 Danh sách các chủ đề")
for idx, topic in lda_model.print_topics(-1):
    # Hiển thị tên chủ đề và các từ khóa đặc trưng
    topic_name = topic_names.get(idx, "Chủ đề không xác định")  # Lấy tên chủ đề từ dictionary
    st.markdown(f"**{topic_name}**: {topic}")

# Tạo trực quan hóa chủ đề bằng pyLDAvis
st.subheader("📊 Trực quan hóa chủ đề")
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis_html = pyLDAvis.prepared_data_to_html(vis)
st.components.v1.html(pyLDAvis_html, width=1600, height=800)

# Tra cứu chủ đề của một bộ phim
st.subheader("🔎 Xem chủ đề của một phim cụ thể")
film_title = st.selectbox("Chọn tên phim", df["title"].tolist())

# Hiển thị thông tin chi tiết về phim
film_info = df[df["title"] == film_title].iloc[0]
st.markdown(f"**Thông tin phim _{film_title}_**:")

# Hiển thị thông tin chi tiết về phim (Xử lý trường hợp không có cột 'year')
if 'year' in df.columns:
    st.markdown(f"📅 **Năm phát hành**: {film_info['year']}")
else:
    st.markdown(f"📅 **Năm phát hành**: Không có thông tin")

st.markdown(f"🌍 **Quốc gia**: {film_info['Countries']}")
st.markdown(f"🗣️ **Ngôn ngữ**: {film_info['Languages']}")
st.markdown(f"⭐ **Đánh giá trung bình**: {film_info['averageRating']}")
st.markdown(f"🎬 **Thể loại**: {film_info['Genres']}")

# Tính toán chủ đề của bộ phim
film_idx = df[df["title"] == film_title].index[0]
bow = dictionary.doc2bow(film_info["processed_plot"].split())
topics = lda_model.get_document_topics(bow)

# Lấy chủ đề có xác suất lớn nhất
dominant_topic = max(topics, key=lambda x: x[1])

st.markdown(f"**Chủ đề chính của phim _{film_title}_**: {topic_names[dominant_topic[0]]} - Xác suất: {dominant_topic[1]:.2f}")

# Tùy chọn cho người dùng để khám phá thêm
st.sidebar.subheader("🔧 Tùy chọn")
show_topic_details = st.sidebar.checkbox("Hiển thị chi tiết về từng chủ đề", value=True)
if show_topic_details:
    topic_num = st.sidebar.number_input("Chọn số chủ đề (0-9)", min_value=0, max_value=9, value=0)
    st.sidebar.write(f"Chi tiết chủ đề {topic_num}:")
    st.sidebar.markdown(f"**{topic_names[topic_num]}**: {lda_model.print_topics(num_topics=10)[topic_num][1]}")
