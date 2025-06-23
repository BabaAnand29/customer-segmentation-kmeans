import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load dataset from the data folder
df = pd.read_csv('data/mall_customers_sample.csv')

# Show first few rows
print("First 5 rows of dataset:")
print(df.head())

# Select features for clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal k
inertia = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Apply KMeans with chosen number of clusters (e.g. 4)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Count of points in each cluster
plt.figure(figsize=(6,4))
sns.countplot(x='Cluster', data=df, palette='Set2')
plt.title('Customer Count per Cluster')
plt.show()

# Scatter plot of clusters with centroids
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', palette='Set2', s=100)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 1], centers[:, 2], c='black', s=200, marker='X', label='Centroids')
plt.title('Customer Segmentation')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Plotly interactive scatter
fig = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)',
                 color=df['Cluster'].astype(str), hover_data=['Age', 'CustomerID'],
                 title='Interactive Customer Segmentation')
fig.show()

# Print cluster centers
centers_df = pd.DataFrame(centers, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
print("\nCluster Centers:")
print(centers_df)
