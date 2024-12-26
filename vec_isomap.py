
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 임베딩 벡터 및 메타데이터 불러오기
embedding_file_path = '/home/user/bob/fakeguard/analysis_result/sy/A_embeddings.tsv'  # 임베딩 벡터 파일 경로
metadata_file_path = '/home/user/bob/fakeguard/analysis_result/sy/A_metadata copy.tsv'  # 메타데이터 파일 경로

embedding_data = pd.read_csv(embedding_file_path, sep='\t', header=None)
metadata = pd.read_csv(metadata_file_path, sep='\s+', header=None, names=['Name', 'Label'])

# NaN 값이 있는 행 제거
merged_data = pd.concat([embedding_data, metadata], axis=1)
merged_data = merged_data.dropna(subset=['Label'])

# embedding_data와 metadata 업데이트
embedding_data = merged_data.iloc[:, :-2]  # 마지막 두 열을 제외한 부분 (임베딩 데이터)
labels = merged_data['Label']  # 레이블만 따로 추출


# LDA를 사용한 차원 축소
lda = LinearDiscriminantAnalysis(n_components=2)
embedding_lda = lda.fit_transform(embedding_data, metadata['Label'])

# 5. 레이블을 숫자로 변환
label_mapping = {label: idx for idx, label in enumerate(labels.unique())}
numeric_labels = labels.map(label_mapping)

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(embedding_lda[:, 0], embedding_lda[:, 1], c=numeric_labels, cmap='viridis', alpha=0.5)
# plt.scatter(embedding_lda[:, 0], embedding_lda[:, 1], c=metadata['Label'], cmap='viridis', alpha=0.5)
for i, name in enumerate(metadata['Name']):
    plt.annotate(name, (embedding_lda[i, 0], embedding_lda[i, 1]), fontsize=6, alpha=0.6)
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('2D Visualization of Embedding Vectors using LDA')
plt.colorbar(label='Class Label')
plt.savefig(f'/home/user/bob/fakeguard/lda.png')
