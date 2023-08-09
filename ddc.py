import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ripser import Rips

def euclidean_distance(point1, point2):                                                     #Calculates the Euclidean distance between two data points
    return np.sqrt(np.sum((point1 - point2) ** 2))

def assign_to_clusters(data, centroids):                                                    #Assigns data points to the nearest cluster based on current centroids
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_idx = np.argmin(distances)
        clusters[cluster_idx].append(point)
    return clusters

def update_centroids(clusters):                                                             #Updates the centroids based on the mean of the data points in each cluster
    return [np.mean(cluster, axis=0) for cluster in clusters]

def dynamic_kmeans(data, k, tolerance=1e-4):                                                #Performs dynamic k-means clustering
    np.random.seed(42)                                                                      #Randomly initializes centroids
    centroids = np.array([data[i] for i in np.random.choice(len(data), k)])
    while True:
        old_centroids = np.copy(centroids)
        clusters = assign_to_clusters(data, centroids)                                      #Assigns data points to clusters using the current centroids
        centroids = update_centroids(clusters)                                              #Updates the centroids based on the current cluster assignments
        centroid_shift = np.sum(np.sqrt(np.sum((centroids - old_centroids) ** 2, axis=1)))  #Calculates the shift in centroids to check for convergence
        print(f"Centroid Shift = {centroid_shift}")
        if centroid_shift < tolerance:                                                      #Check for convergence
            print("Converged!")
            break
    return centroids, clusters

def topological_data_analysis(data, k, maxdim=1):
    centroids, clusters = dynamic_kmeans(data, k)                                           #Performs dynamic k-means clustering
    rips = Rips(maxdim=maxdim)                                                              #Computes persistent homology using ripser
    data_np = np.array(data)
    diagrams = rips.fit_transform(data_np)
    plt.figure(figsize=(12, 6))                                                             #Visualizes the clustered data and topological features
    plt.subplot(1, 2, 1)
    for i, cluster in enumerate(clusters):                                                  #Plots data points in each cluster
        cluster_data = np.array(cluster)
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {i+1}")
    centroids_data = np.array(centroids)                                                    #Plots cluster centroids
    plt.scatter(centroids_data[:, 0], centroids_data[:, 1], color='red', marker='X', s=200, label='Centroids')
    plt.ylabel('Revenue')
    plt.xlabel('Employees')
    plt.title('Dynamic K-Means Clustering')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    for diagram in diagrams:
        if len(diagram) > 0:
            birth, death = diagram[:, :2].T
            plt.plot([0, maxdim], [0, maxdim], 'k--', alpha=0.5)
            plt.scatter(birth, death, c='b', marker='o', label='Topological Features')
            plt.xlabel('Birth')
            plt.ylabel('Death')
            plt.title('Persistent Homology Diagram')
            plt.legend()
            plt.grid(True)
    plt.tight_layout()
    plt.show()

file_path = "LargestCompaniesInUSAbyReveneue.csv"                                           #Reading data from CSV file
data_df = pd.read_csv(file_path)
data_df['Revenue'] = data_df['Revenue'].str.replace(',', '').astype(float)                  #Converts comma separated values to float data type
data_df['Employees'] = data_df['Employees'].str.replace(',', '').astype(float)
data = data_df[['Employees','Revenue']].values.tolist()                                     #Selects the 'Revenue' and 'Employees' column for clustering
data_df['Employees'] = pd.to_numeric(data_df['Employees'], errors='coerce')                 #Converts non-numeric values to numeric values
data_df['Revenue'] = pd.to_numeric(data_df['Revenue'], errors='coerce')
data_df.dropna(subset=['Employees', 'Revenue'], inplace=True)                               #Removes rows with NaN values
k=3                                                                                         #Number of clusters

topological_data_analysis(data, k)                                                          #Performs topological data analysis
