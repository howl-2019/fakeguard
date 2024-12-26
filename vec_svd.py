
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

# 1. 임베딩 벡터 및 메타데이터 불러오기
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


# 2. 데이터 확인
print(embedding_data.head())
print(merged_data.head())

# 3. SVD를 사용한 차원 축소 (TruncatedSVD 사용)
# 차원 축소할 개수 설정 (예: 2차원으로 축소)
n_components = 2
svd = TruncatedSVD(n_components=n_components)
embedding_svd = svd.fit_transform(embedding_data)

# 4. SVD 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(embedding_svd[:, 0], embedding_svd[:, 1], alpha=0.5)

# 메타데이터에 있는 이름을 각 점에 표시
for i, name in enumerate(metadata['Name']):
    plt.annotate(name, (embedding_svd[i, 0], embedding_svd[i, 1]), fontsize=8, alpha=0.7)

plt.xlabel('SVD Component 1')
plt.ylabel('SVD Component 2')
plt.title('2D Visualization of Embedding Vectors using SVD')
plt.savefig('/home/user/bob/fakeguard/svd.png')
