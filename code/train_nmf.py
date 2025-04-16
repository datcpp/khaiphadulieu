
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px

# ----------------------------------------
# BƯỚC 0: CHUẨN BỊ DỮ LIỆU
print("BƯỚC 0: CHUẨN BỊ DỮ LIỆU")
df = pd.read_csv('D:/khaiphadl/processed_data.csv')
print("-> Các cột trong file:", df.columns.tolist())

column_name = 'final_clean_plot'
if column_name not in df.columns:
    raise ValueError(f"Không tìm thấy cột '{column_name}' trong file CSV.")
texts = df[column_name].fillna('')
print(f"-> Tổng số văn bản: {len(texts)}")

# ----------------------------------------
# BƯỚC 1: CHUYỂN VĂN BẢN THÀNH MA TRẬN TF-IDF
print("\nBƯỚC 1: CHUYỂN VĂN BẢN THÀNH MA TRẬN TF-IDF")
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
V = vectorizer.fit_transform(texts).toarray()
m, n = V.shape
print(f"-> Ma trận TF-IDF có kích thước: {m} x {n}")

# ----------------------------------------
# BƯỚC 2: KHỞI TẠO MA TRẬN W VÀ H
print("\nBƯỚC 2: KHỞI TẠO MA TRẬN W VÀ H")
r = 5  # số lượng chủ đề
np.random.seed(42)
W = np.abs(np.random.rand(m, r))
H = np.abs(np.random.rand(r, n))
print(f"-> W: {W.shape}, H: {H.shape} (khởi tạo ngẫu nhiên không âm)")

# ----------------------------------------
# BƯỚC 3: CẬP NHẬT LẶP
print("\nBƯỚC 3: CẬP NHẬT LẶP (Multiplicative Update)")
def nmf_custom(V, W, H, max_iter=100, epsilon=1e-9):
    for i in range(max_iter):
        H *= (W.T @ V) / (W.T @ W @ H + epsilon)
        W *= (V @ H.T) / (W @ H @ H.T + epsilon)

        error = np.linalg.norm(V - W @ H, 'fro')
        if i % 10 == 0 or error < 1e-4:
            print(f"  - Iteration {i}, Frobenius error: {error:.6f}")
        if error < 1e-4:
            print("  -> Đạt ngưỡng hội tụ.")
            break
    return W, H

W, H = nmf_custom(V, W, H)

# ----------------------------------------
# BƯỚC 4: DIỄN GIẢI KẾT QUẢ
print("\nBƯỚC 4: DIỄN GIẢI KẾT QUẢ")
feature_names = vectorizer.get_feature_names_out()
n_top_words = 10
topics = []

for topic_idx, topic in enumerate(H):
    top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    topic_words = ", ".join(top_words)
    topics.append(topic_words)
    print(f"-> Chủ đề {topic_idx+1}: {topic_words}")

# ----------------------------------------
# GÁN CHỦ ĐỀ CHO VĂN BẢN
print("\nBƯỚC 5: GÁN CHỦ ĐỀ CHO VĂN BẢN")
topic_assignments = W.argmax(axis=1)
df['predicted_topic'] = topic_assignments
print("-> Đã gán chủ đề cho từng văn bản.")

# ----------------------------------------
# VẼ BIỂU ĐỒ 3D

topic_counts = df['predicted_topic'].value_counts().sort_index()
plot_df = pd.DataFrame({
    'Chủ đề': [f'Chủ đề {i+1}' for i in topic_counts.index],
    'Số lượng': topic_counts.values,
    'Chỉ số': topic_counts.index
})

fig = px.scatter_3d(
    plot_df,
    x='Chủ đề',
    y='Số lượng',
    z='Chỉ số',
    color='Chủ đề',
    size='Số lượng',
    size_max=30,
    title='Biểu đồ 3D: Phân bố văn bản theo chủ đề'
)
fig.show()

# ----------------------------------------
#: KẾT LUẬN
print("\nKẾT LUẬN:")
for i in range(r):
    count = (df['predicted_topic'] == i).sum()
    print(f"- Chủ đề {i+1} ({topics[i]}): {count} văn bản")


