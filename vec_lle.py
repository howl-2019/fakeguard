import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding

# 1. 임베딩 벡터 및 메타데이터 불러오기
embedding_file_path = '/home/user/bob/fakeguard/analysis_result/sy/A_embeddings.tsv'  # 임베딩 벡터 파일 경로
metadata_file_path = '/home/user/bob/fakeguard/analysis_result/sy/A_metadata.tsv'  # 메타데이터 파일 경로

embedding_data = pd.read_csv(embedding_file_path, sep='\t', header=None)
metadata = pd.read_csv(metadata_file_path, sep='\t', header=None, names=['Name'])

# 2. 데이터 확인 (처음 몇 개의 데이터 포인트와 메타데이터 확인)
print(embedding_data.head())
print(metadata.head())

# 3. LLE를 사용한 차원 축소 및 시각화
n_components = 2  # 축소할 차원 수 설정
n_neighbors = 8  # LLE에 사용할 이웃 수 설정

lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
embedding_lle = lle.fit_transform(embedding_data)

plt.figure(figsize=(10, 6))
plt.scatter(embedding_lle[:, 0], embedding_lle[:, 1], c='blue', label='LLE Points', alpha=0.5)
for i, name in enumerate(metadata['Name']):
    plt.annotate(name, (embedding_lle[i, 0], embedding_lle[i, 1]), color='blue', fontsize=6, alpha=0.6)
plt.xlabel('LLE Component 1')
plt.ylabel('LLE Component 2')
plt.title('2D Visualization of Embedding Vectors using LLE')
plt.legend()
plt.savefig(f'/home/user/bob/fakeguard/lle_{n_neighbors}.png')
