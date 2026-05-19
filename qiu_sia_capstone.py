import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder

# preprocessing
SEED = 42
random.seed(SEED)

df = pd.read_csv('musicData.csv')

# drop columns 1, 2, 3, and 16
df = df.drop(columns=['instance_id', 'artist_name', 'track_name', 'obtained_date'])

# check NaN values
numeric_features = ['popularity', 'acousticness', 'danceability', 'duration_ms', 
                   'energy', 'instrumentalness', 'liveness', 'loudness', 
                   'speechiness', 'tempo', 'valence']
categorical_features = ['key', 'mode']

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 'tempo' has 4985 missing values while other columns each has 5 missing values
# 'duration_ms' has 4939 values of -1.0, which is impossible for duration
# fill in NaN values in the 'tempo' and 'duration_ms' columns by the median tempo or duration per genre
df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')
df['tempo'] = df['tempo'].fillna(df.groupby('music_genre')['tempo'].transform('median'))
df['duration_ms'] = df['duration_ms'].replace(-1.0, np.nan)
df['duration_ms'] = df['duration_ms'].fillna(df.groupby('music_genre')['duration_ms'].transform('median'))
df = df.dropna(subset=['music_genre'])

# dummy coding 'key' and 'mode'
key_mapping = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}
df['key'] = df['key'].map(key_mapping)
df['mode'] = df['mode'].map({'Minor': 0, 'Major': 1})

# Correlation heatmap
X_corr = df.drop('music_genre', axis=1)
corr_multi = X_corr.corr()

plt.figure(figsize=(6,6))
plt.imshow(corr_multi, interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr_multi.columns)), corr_multi.columns, rotation=90)
plt.yticks(range(len(corr_multi.columns)), corr_multi.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# train/test split
train_df, test_df = train_test_split(df, test_size=5000, random_state=SEED, stratify=df['music_genre'])

print("Training set count by genre:\n", train_df['music_genre'].value_counts())
print("Test set count by genre:\n", test_df['music_genre'].value_counts())

# map 'music_genre' to numerical labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['music_genre'])
y_test = le.transform(test_df['music_genre'])

# normalize numerical (not categorical) features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(train_df[numeric_features])
X_test_scaled = scaler.transform(test_df[numeric_features])

# concatenate normalized numeric features and categorical features
X_train_final = np.hstack([X_train_scaled, train_df[categorical_features].values])
X_test_final = np.hstack([X_test_scaled, test_df[categorical_features].values])

# dimension reduction with PCA

# examine cumulative variance plot to choose a number of components to keep
pca_full = PCA(random_state=SEED)
pca_full.fit(X_train_final)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_) * 100

plt.figure(figsize=(12, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
plt.axhline(y=95, color='r', linestyle='--', label='95% Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.ylim(0, 105)
plt.xticks(range(1,14))
plt.fill_between(range(1, 14), cumulative_variance, alpha=0.1, color='steelblue')
plt.grid(True)
plt.legend()

for i, var in enumerate(cumulative_variance):
    print(f"Components: {i+1}, Cumulative Variance: {var:.3f}")

# visualization in 2D using only the numeric features (without 'key' and 'mode')
pca_vis1 = PCA(n_components=2, random_state=SEED)
X_pca_vis1 = pca_vis1.fit_transform(X_train_scaled)

# use a subset of 5000 points so the plot isn't too crowded
plt.figure(figsize=(12, 8))
for genre_id in range(len(le.classes_)):
    idx = np.where(y_train == genre_id)[0][:500] 
    plt.scatter(X_pca_vis1[idx, 0], X_pca_vis1[idx, 1], 
    label=le.classes_[genre_id], alpha=0.6, s=15)

plt.title('Genre Clusters in PCA Space Without Key and Mode (PC1 vs. PC2)')
plt.xlabel(f'Principal Component 1 ({pca_vis1.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'Principal Component 2 ({pca_vis1.explained_variance_ratio_[1]:.1%})')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)

# loadings of PC1 and PC2
loadings = pd.DataFrame(
    pca_vis1.components_.T, 
    columns=['PC1', 'PC2'], 
    index=numeric_features
)

cov_matrix = np.cov(X_train_scaled, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
ev_above_one = np.sum(eigenvalues > 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for pc_idx, ax in enumerate(axes):
    pc_loadings = pca_vis1.components_[pc_idx]
    var_explained = eigenvalues[pc_idx] / eigenvalues.sum() * 100
    bar_colors = ['steelblue' if v >= 0 else '#e8816d' for v in pc_loadings]
    
    ax.bar(range(len(pc_loadings)), pc_loadings, color=bar_colors, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(pc_loadings)))
    ax.set_xticklabels(numeric_features, rotation=45, ha='right')
    ax.set_ylabel("Loading Value")
    ax.set_xlabel("Features")
    ax.set_title(f"Numeric Features Only PCA: Loadings for PC{pc_idx+1} ({var_explained:.1f}% variance)")
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=0.3, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=-0.3, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# visualization in 2D using all features
pca_vis2 = PCA(n_components=2, random_state=SEED)
X_pca_vis2 = pca_vis2.fit_transform(X_train_final)

# use a subset of 5000 points so the plot isn't too crowded
plt.figure(figsize=(12, 8))
for genre_id in range(len(le.classes_)):
    idx = np.where(y_train == genre_id)[0][:500] 
    plt.scatter(X_pca_vis2[idx, 0], X_pca_vis2[idx, 1], 
    label=le.classes_[genre_id], alpha=0.6, s=15)

plt.title('Genre Clusters in PCA Space With Key and Mode (PC1 vs. PC2)')
plt.xlabel(f'Principal Component 1 ({pca_vis2.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'Principal Component 2 ({pca_vis2.explained_variance_ratio_[1]:.1%})')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)

# loadings of PC1 and PC2
loadings = pd.DataFrame(
    pca_vis2.components_.T, 
    columns=['PC1', 'PC2'], 
    index=numeric_features + categorical_features
)

cov_matrix = np.cov(X_train_final, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
ev_above_one = np.sum(eigenvalues > 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for pc_idx, ax in enumerate(axes):
    pc_loadings = pca_vis2.components_[pc_idx]
    var_explained = eigenvalues[pc_idx] / eigenvalues.sum() * 100
    bar_colors = ['steelblue' if v >= 0 else '#e8816d' for v in pc_loadings]
    
    ax.bar(range(len(pc_loadings)), pc_loadings, color=bar_colors, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(pc_loadings)))
    ax.set_xticklabels(numeric_features + categorical_features, rotation=45, ha='right')
    ax.set_ylabel("Loading Value")
    ax.set_xlabel("Features")
    ax.set_title(f"All Features PCA: Loadings for PC{pc_idx+1} ({var_explained:.1f}% variance)")
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=0.3, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=-0.3, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# final PCA with the top 9 PCs to retain 95% of variance
pca = PCA(n_components=9, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_final)
X_test_pca = pca.transform(X_test_final)
print("Reduced feature count after PCA:", X_train_pca.shape[1])

# K-Means on continous features only
pca_km = PCA(n_components=9, random_state=SEED)
X_train_km = pca_km.fit_transform(X_train_scaled)
X_test_km = pca_km.transform(X_test_scaled)

km = KMeans(n_clusters=10, n_init=10, random_state=SEED).fit(X_train_km)
labels_km = km.labels_
sil_km = silhouette_score(X_train_km, labels_km)
distortion = km.inertia_
centroids_km = km.cluster_centers_
print(f"k-means silhouette score for continuous features: {sil_km:.3f}")
print(f"Total Distortion for k=10: {distortion:.3f}")
print(f"Centroids (standardized space):\n{centroids_km}")

plt.figure(figsize=(12, 8))
for genre_id in range(len(le.classes_)):
    idx = np.where(y_train == genre_id)[0][:500] 
    plt.scatter(X_train_km[idx, 0], X_train_km[idx, 1], 
    label=le.classes_[genre_id], alpha=0.6, s=15)

plt.scatter(centroids_km[:, 0], centroids_km[:, 1], 
            marker='X', s=100, linewidths=1, 
            color='black', label='K-Means Centroids')

plt.title('Genre Clusters vs. k-Means Centroids on PCA Projection with Continuous Features')
plt.xlabel(f'Principal Component 1 ({pca_km.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'Principal Component 2 ({pca_km.explained_variance_ratio_[1]:.1%})')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)

# K-Means on all features
km_all = KMeans(n_clusters=10, n_init=10, random_state=SEED).fit(X_train_pca)
labels_km_all = km_all.labels_
sil_km_all = silhouette_score(X_train_pca, labels_km_all)
distortion_all = km_all.inertia_
centroids_km_all = km_all.cluster_centers_
print(f"k-Means silhouette score for all features: {sil_km_all:.3f}")
print(f"Total Distortion for k=10: {distortion_all:.3f}")
print(f"Centroids (standardized space):\n{centroids_km_all}")

plt.figure(figsize=(12, 8))
for genre_id in range(len(le.classes_)):
    idx = np.where(y_train == genre_id)[0][:500] 
    plt.scatter(X_train_pca[idx, 0], X_train_pca[idx, 1], 
    label=le.classes_[genre_id], alpha=0.6, s=15)

plt.scatter(centroids_km_all[:, 0], centroids_km_all[:, 1], 
            marker='X', s=100, linewidths=1, 
            color='black', label='K-Means Centroids')

plt.title('Genre Clusters vs. k-Means Centroids on PCA Projection with All Features')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)

train_clusters = km_all.predict(X_train_pca).reshape(-1, 1)
test_clusters = km_all.predict(X_test_pca).reshape(-1, 1)
oh = OneHotEncoder(sparse_output=False)
train_clusters_oh = oh.fit_transform(train_clusters)
test_clusters_oh = oh.transform(test_clusters)
X_train_augmented = np.hstack([X_train_pca, train_clusters_oh])
X_test_augmented = np.hstack([X_test_pca, test_clusters_oh])

# models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_features='sqrt', oob_score=True, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.1, random_state=SEED),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,100,50), activation='logistic', max_iter=400, random_state=SEED, early_stopping=True, learning_rate_init = 0.001)
    }

# binarize output labels for multi-class plotting
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

fig, ax = plt.subplots(1,2,figsize=(18,7))
for name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_probs = model.predict_proba(X_test_pca)
    
    # ROC Curve
    auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_probs.ravel())
    ax[0].plot(fpr, tpr, label=f'{name} AUC: {auc:.3f}')
    print(f"{name} AUC: {auc:.3f}")
    
    # PR Curve
    ap = average_precision_score(y_test, y_probs)
    precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_probs.ravel())
    ax[1].plot(recall, precision, label=f'{name} AP: {ap:.3f}')
    
    if name == "Random Forest":
        print(f"Random Forest OOB Score: {model.oob_score_:.3f}")

# ROC Curve formatting
ax[0].plot([0,1], [0,1], 'k--', label="Random Classifier")
ax[0].fill_between([0,1], [0,1], alpha=0.05, color='gray')
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
ax[0].set_title("ROC Curves - All Classifiers", fontweight='bold')
ax[0].legend(loc='lower right')
ax[0].grid(True, alpha=0.3)

# PR Curve formatting
ax[1].axhline(1/10, color='gray', linestyle='--', label='Baseline P: 0.100')
ax[1].set_xlabel('Recall')
ax[1].set_ylabel('Precision')
ax[1].set_title('Precision-Recall Curves - All Classifiers', fontweight='bold')
ax[1].legend(loc='upper right')
ax[1].grid(True, alpha=0.3)
ax[1].fill_between([0, 1], 0, 1/10, color='gray', alpha=0.05)

fig.show()
plt.show()
