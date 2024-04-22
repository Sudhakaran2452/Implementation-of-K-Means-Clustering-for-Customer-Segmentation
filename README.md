# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Choose the number of clusters (K): Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

2.Initialize cluster centroids: Randomly select K data points from your dataset as the initial centroids of the clusters.

3.Assign data points to clusters: Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.

4.Update cluster centroids: Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

5.Repeat steps 3 and 4: Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

6.Evaluate the clustering results: Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

7.Select the best clustering solution: If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements



## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SUDHAKARAN S
RegisterNumber: 212222220051
*/
```
```

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = "k-means++")
  kmeans.fit(data.iloc[:, 3:])
  wcss.append(kmeans.inertia_)
  
plt.plot(range(1, 11), wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:, 3:])

y_pred = km.predict(data.iloc[:, 3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c = "red", label = "cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c = "black", label = "cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c = "blue", label = "cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c = "green", label = "cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c = "magenta", label = "cluster4")
plt.legend()
plt.title("Customer Segments")

```


## Output:

## data.head():
![278949588-20e28c10-49ec-4912-9b52-aa1fa6046cdd](https://github.com/820NaveenKumar208/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154746066/d3776e69-b527-4c1f-a65e-e5d37163dff8)

## data.info():
![278949592-b72586a8-e2c9-46ab-bbbe-36120412beb3](https://github.com/820NaveenKumar208/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154746066/797cd4a2-527f-410f-9368-85a612a0e17d)

## NULL VALUES:
![278949606-807815d1-9dd7-4139-a7d0-50d75fb3286c](https://github.com/820NaveenKumar208/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154746066/15759e58-2730-4652-985b-5470276f9b3e)

## ELBOW GRAPH:

![278949617-b4d0d533-6132-4eb8-b1de-be37fee48eff](https://github.com/820NaveenKumar208/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154746066/4dad7bce-e11a-4d1f-b692-c45c5fcb985e)

## CLUSTER FORMATION:
![278949632-9ea2de21-b25c-473c-a445-be867560c5a5](https://github.com/820NaveenKumar208/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154746066/5e0d233e-911c-474d-af06-9ac98fe63291)


## PREDICICTED VALUE:
![278949643-7d1d3af3-1df5-4b47-baa7-2105225f1ea0](https://github.com/820NaveenKumar208/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154746066/56a0ea3d-1271-40d8-a087-2a6511eff030)

## FINAL GRAPH(D/O):
![278949652-f14deb56-9d40-4fe5-9100-677e33629c56](https://github.com/820NaveenKumar208/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154746066/9a92904f-ef70-492b-a399-7b1593874b43)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
