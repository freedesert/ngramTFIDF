# ngramTFIDF
Summary: Single linkage clustering of DNA sequences using a technique of TFIDF

1.Conversion of sequence data to term weighted frequency of k-mer
2.Laplace score for feature selection:
3. Single linkage hierarchical clustering:
We used python library scipy for single linkage hierarchical algorithm . Input data for clustering algorithm  consist of (m-sample, n-dimension) and we get output as the dendrogram representing the cluster for given input data. We can get the cluster labels by specifying the height at which we want to cut the dendrogram.
