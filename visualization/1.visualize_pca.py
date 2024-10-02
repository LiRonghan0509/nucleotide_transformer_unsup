import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

# random sample

print("Generating random sampling index")

# load sampled data and labels
print("Loading sampled data and labels...")
embeddings_path = '/storage1/fs1/yeli/Active/l.ronghan/projects/4.nt_unsupervised/output-2/mean_embeddings_sampled_128.csv'
labels_path = "/storage1/fs1/yeli/Active/l.ronghan/data/humanbrain_cCREs/nt_data/subclass/sampled_humanbrain_cCREs_subclass.csv"

# embeddings_df = pd.read_csv(embeddings_path, skiprows=lambda x: x not in sample_indices)
embeddings_df = pd.read_csv(embeddings_path)

labels_df = pd.read_csv(labels_path, usecols=['cell'])
print("Embeddings shape:", embeddings_df.shape)
print("Labels shape:", labels_df.shape)

# initialize 2D PCA
print("Initializing 2D PCA...")
ipca = IncrementalPCA(n_components=2, batch_size=10000)  
pca_result = ipca.fit_transform(embeddings_df)

# store PCA results
pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
pca_df['class'] = labels_df

# plot 2D PCA
print("Starting plot 2D PCA...")
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='class', data=pca_df, palette='tab10')
plt.title('PCA of Sampled Mean Embeddings')
plt.savefig('figures-2/pca_mean_embeddings_128.png', dpi=300)

# initialize 3D PCA
print("Initializing 3D PCA...")
pca_3d = PCA(n_components=3)
pca_3d_result = pca_3d.fit_transform(embeddings_df)

pca_3d_df = pd.DataFrame(pca_3d_result, columns=['PC1', 'PC2', 'PC3'])
labels_df['cell_encoded'] = pd.Categorical(labels_df['cell']).codes
pca_3d_df['label'] = labels_df['cell_encoded'].values
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 3D scatters
scatter = ax.scatter(pca_3d_df['PC1'], pca_3d_df['PC2'], pca_3d_df['PC3'], 
                     c=pca_3d_df['label'], cmap='viridis', s=10)

# 设置轴标签
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# 添加颜色条用于区分不同的 class
plt.colorbar(scatter)

# 保存图像
plt.savefig('figures-2/pca_mean_embeddings_3d_128.png', dpi=300)



# load sampled data and labels
print("Loading sampled data and labels...")
embeddings_path = '/storage1/fs1/yeli/Active/l.ronghan/projects/4.nt_unsupervised/output-2/embeddings_sampled_128.csv'
labels_path = "/storage1/fs1/yeli/Active/l.ronghan/data/humanbrain_cCREs/nt_data/subclass/sampled_humanbrain_cCREs_subclass.csv"

# embeddings_df = pd.read_csv(embeddings_path, skiprows=lambda x: x not in sample_indices)
embeddings_df = pd.read_csv(embeddings_path)

labels_df = pd.read_csv(labels_path, usecols=['cell'])
print("Embeddings shape:", embeddings_df.shape)
print("Labels shape:", labels_df.shape)

# initialize 2D PCA
print("Initializing 2D PCA...")
ipca = IncrementalPCA(n_components=2, batch_size=10000)  
pca_result = ipca.fit_transform(embeddings_df)

# store PCA results
pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
pca_df['class'] = labels_df

# plot 2D PCA
print("Starting plot 2D PCA...")
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='class', data=pca_df, palette='tab10')
plt.title('PCA of Sampled Embeddings')
plt.savefig('figures-2/pca_embeddings_128.png', dpi=300)

# initialize 3D PCA
print("Initializing 3D PCA...")
pca_3d = PCA(n_components=3)
pca_3d_result = pca_3d.fit_transform(embeddings_df)

pca_3d_df = pd.DataFrame(pca_3d_result, columns=['PC1', 'PC2', 'PC3'])
labels_df['cell_encoded'] = pd.Categorical(labels_df['cell']).codes
pca_3d_df['label'] = labels_df['cell_encoded'].values
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 3D scatters
scatter = ax.scatter(pca_3d_df['PC1'], pca_3d_df['PC2'], pca_3d_df['PC3'], 
                     c=pca_3d_df['label'], cmap='viridis', s=10)

# set axis labels
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.colorbar(scatter)

# save images
plt.savefig('figures-2/pca_embeddings_3d_128.png', dpi=300)

