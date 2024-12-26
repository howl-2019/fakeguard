import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# 1. Load the TSV file containing embedding vectors
ts_file_path = '/home/user/bob/fakeguard/analysis_result/sy/F_embeddings.tsv'  # replace this with your file path
embeddings = pd.read_csv(ts_file_path, sep='\t', header=None)

# 2. Load the metadata containing names of the embeddings
metadata_file_path = '/home/user/bob/fakeguard/analysis_result/sy/F_metadata.tsv'  # replace this with your metadata file path
metadata = pd.read_csv(metadata_file_path, sep='\t', header=None)
labels = metadata[0].values

# 3. Convert embeddings to numpy array
embedding_array = embeddings.values

# 4. Apply MDS for dimensionality reduction
maxiter = 100
n_init = 9
mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean', max_iter=maxiter, n_init=n_init, metric=True)
reduced_embeddings = mds.fit_transform(embedding_array)

# 5. Plot the reduced dimensions
plt.figure(figsize=(10, 8))
# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', alpha=0.5)
for i, label in enumerate(labels):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], alpha=0.5)
    plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label, fontsize=9, alpha=0.7)
plt.title('MDS Projection of Embeddings')
plt.suptitle(f'max iter = {maxiter}, n init = {n_init}, metric = True', fontsize=10)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()
plt.savefig(f'/home/user/bob/fakeguard/mds_{maxiter}_{n_init}.png')