---
title: Cluster Analysis using Python
date: "2020-05-09T22:40:32.169Z"
description: Demonstrating how Cluster Analysis can be done using Python
---


## Goal of Cluster Analysis

 - The goal of cluster analysis is to **partition** the data into **distinct sub-groups or clusters** such that observations belonging to the same cluster is very similar or **homogeneous** and observations belonging to different clusters are **different or heterogenous.** 
 - The measurement of **similarity** may be **distance, correlation, cosine similarity** etc. depending on the **context/domain** of the problem.

### Application of Clustering

 - One very popular application of cluster analysis in business is **market segmentation.** Here, customers are grouped into distinct clusters or market segments and **each segment is targeted with different marketing mix** such as different promotional messages, different products, different prices, and different distribution channels.

 - Other example of clustering may be **clustering of products** into different sub-groups based on attributes like price-elasticity, genres etc.

 - In a way, clustering **compresses** the entire data into a reduced set of **sub-groups.** So, clustering is a data reduction technique.

### K-Means Clustering

 - The idea behind K-mean clustering is that a good clustering is one for which within-cluster variation is as small as possible. 

 - The one possible measure of within-cluster variation for the kth cluster is the sum of all the pairwise distance between the observations in the kth cluster, divided by the total number of observations in the kth cluster. 

 - The total within-cluster variation is sum of the all within-cluster variation for 1 to kth cluster. 

 - Minimizing this total within-cluster variation is the optimization problem for K-means clustering.

 - K-means algorithm provides a local optimum- nevertheless, a good solution to the above optimization problem. 

### K-Means Algorithm

 1.  Randomly assign a number, from 1 to K, to each of the observations. These serve as     initial cluster assignments for the observations.
 2.  Iterate until there is no change in cluster centroids (change is below a tolerance limit):
    - For each of the K clusters, compute the cluster centroids. The kth cluster centroid is the vector of all the feature means for the observations in the kth cluster. 
    - Assign each observation to the cluster whose centroid is closest. 

### Mathematical perspective of clustering

 - Let C1,…, Ck denote sets containing the indices of the observations in each cluster. The set should satisfy two properties:
    1. C1 ∪ C2 ∪ … Ck = {1,…,n}. That is each observation belong to at least one of the k clusters.
    2. Ci ∩ Cj =∅ for all i≠j.  No observation belongs to more than one cluster.
 - K-means cost Function:
    - Let Z1,…,Zk are the cluster centroids. 
    - Cost (C1,…,Ck,Z1,…,Zk)=∑_(j=1)^k▒〖∑▒〖i∈Cj〗||Xi-Zj||2〗

### Isues with K-Means Clustering

 Because, k-means algorithm finds a **local** rather than a **global optimum.** The result depends on **initial random cluster assignment.** Hence, it is important to run the algorithm **multiple times** from different random initial configurations. Then select **the best solution** for which **the total within-cluster variation** is **smallest.**

### How many clusters to extract?

 - Decide the number of clusters which is most interpretable and actionable.
 - Other Guidelines: 
    - Plot of k-means cost versus K.
    - Silhouette score
    - Visualizing data on the first two Principal Components.

### References 

 1.  Mitchell, T. M. (1997). Machine Learning. New York: McGraw-Hill 
 2. Murphy, Kevin P.(2012). Machine Learning: A Probabilistic Perspective. Cambridge, MA: MIT Press
 3.  Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. (2013). An introduction to statistical learning : with applications in R. New York :Springer