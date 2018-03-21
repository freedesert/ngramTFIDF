import itertools
import random
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn import random_projection
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from collections import Counter
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score
from scipy.cluster.hierarchy import fclusterdata


#generates all possible n-grams from ACTG
def nGramList(n):
	nGrList = []
	base = 'ACTG'
	for i in range(n,n+1):
		nGrList = [''.join(x) for x in itertools.product(base, repeat=i)]
	#print nGrList
	return nGrList
#counts the n gram freequency from given sequence
def countNgram(ngram,seq):
	count=0
	num_kmers = len(seq)-len(ngram)+1
	for i in range(num_kmers):
		kmer = seq[i:i+len(ngram)]
		if kmer == ngram:
			count=count+1
	return count
	
#print countNgram('ACT', 'ACTACTTTACTAAAC')
#nGramList(3)
#find freequency of each possible n-gram from given sequence
def allNgramWithFreq(seq):
	ngrList = nGramList(3)
	freqList = []
	for ngr in ngrList:
		freqList.append((ngr,countNgram(ngr,seq)))
	return freqList
#print allNgramWithFreq('ACTGGACCCATACTAGGGACTCCAAATTGACT')

def generateFeatureDataset():
	inFile = open('top10.txt','r')
	outFile = open('feature.txt','w')

	for line in inFile:
		outLine = ''
		#freeqList = [('AAA', 1), ('AAC', 0), ('AAT', 1)]
		freeqList = allNgramWithFreq(line.strip())
		for (x,y) in freeqList:
			outLine = outLine+str(y)+' '
		outFile.write(outLine)
		outFile.write("\n")

#generateFeatureDataset()
def pca(X = np.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print "Preprocessing the data using PCA..."
	(n, d) = X.shape;
	X = X - np.tile(np.mean(X, 0), (n, 1));
	(l, M) = np.linalg.eig(np.dot(X.T, X));
	Y = np.dot(X, M[:,0:no_dims]);
	#print Y
	return Y;


generateFeatureDataset()

X = np.loadtxt('feature.txt')

#feature selection
kwargs_W = {"metric":"euclidean","neighbor_mode":"knn","weight_mode":"heat_kernel","k":5,'t':1}
W = construct_W.construct_W(X, **kwargs_W)
score = lap_score.lap_score(X, W=W)
idx = lap_score.feature_ranking(score)
num_fea = 9
X1 = X[:, idx[0:num_fea]]
print X1

cutoff = 1.0
cluster_assignments = fclusterdata(X1, cutoff)
print cluster_assignments
# #baseline results:
# label_true = open('labels.0','r')
# t_label = [ int(t) for t in label_true]
# t_Count = Counter(t_label)
# tBiggest = t_Count.most_common(10)
# unique_t = np.unique(t_label)
# tclust = len(unique_t)
# print 'No of cluster in labels.0 :',tclust
# print tBiggest

# #CD_HIT results:
# label_cd = open('cdHitLabels','r')
# cd_label = [ int(t) for t in label_cd]
# cd_Count = Counter(cd_label)
# cdBiggest = cd_Count.most_common(10)
# unique_cd = np.unique(cd_label)
# cdclust = len(unique_cd)
# print 'No of cluster in cdHitLabels :',cdclust
# print cdBiggest

# #X = StandardScaler().fit_transform(X)
# #print X

# #print y,len(y)
# # lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x, y=None)
# # model = SelectFromModel(lsvc, prefit=True)
# # X_new = model.transform(X)
# # print X_new.shape
# pca = pca(X,5)
# # print pca.shape

# # kmeans = KMeans(n_clusters=5, random_state=0).fit(pca)
# # print kmeans.labels_

# #variance threshold method:
# # sel = VarianceThreshold(threshold=(.96 * (1 - .96)))
# # y = sel.fit_transform(X)
# #print y.shape


# #mean shift algorithm
# bandwidth = estimate_bandwidth(pca, quantile=0.2, n_samples=500)

# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# ms.fit(pca)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_

# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)
# print labels
# print "Mean shift Algorithm"

# Mlabel = [ int(t) for t in labels]
# #calculate biggest cluster and no of seqs for true and predicted labels
# MCount = Counter(Mlabel)
# MBiggest = MCount.most_common(10)
# print MBiggest

# print("number of estimated clusters : %d" % n_clusters_)




# #select k best approach
# # X_new = SelectKBest(chi2, k=20)
# # print X_new

# #random projection
# # transformer = random_projection.SparseRandomProjection()
# # X_new = transformer.fit_transform(X)
# # print X_new.shape


# # #############################################################################
# # Compute DBSCAN
# print "DBSCAN algorithm"
# db = DBSCAN(eps=0.3, min_samples=10).fit(pca)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
# #print labels,len(labels)
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print labels
# print n_clusters_
# # print('Estimated number of clusters: %d' % n_clusters_)
# # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# # print("Adjusted Rand Index: %0.3f"
# #       % metrics.adjusted_rand_score(labels_true, labels))
# # print("Adjusted Mutual Information: %0.3f"
# #       % metrics.adjusted_mutual_info_score(labels_true, labels))
# # print("Silhouette Coefficient: %0.3f"
# #       % metrics.silhouette_score(X, labels))
# ##########################
# #birch Algorithm
# print "Birch Algorithm"
# brc = Birch(branching_factor=50, n_clusters=None, threshold=0.5,compute_labels=True)
# brc.fit(X1)
# l= brc.predict(X1)


# #print 10 clusters
# tlabel = [ int(t) for t in l]
# unique = np.unique(tlabel)
# n_clust = len(unique)
# #calculate biggest cluster and no of seqs for true and predicted labels
# tCount = Counter(tlabel)
# t10 = tCount.most_common(10)
# print t10
# print 'number of cluster :',n_clust
