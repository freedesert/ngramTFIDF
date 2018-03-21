import itertools
import random
import numpy as np
import math
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score
from scipy.cluster.hierarchy import fclusterdata,linkage,dendrogram,fcluster
from sklearn.metrics.cluster import normalized_mutual_info_score
from matplotlib import pyplot as plt
from collections import Counter
import re
#generates all possible n-grams from ACTG
def nGramList(n):
	nGrList = []
	base = 'ACTG'
	for i in range(n,n+1):
		nGrList = [''.join(x) for x in itertools.product(base, repeat=i)]
	#print nGrList
	return nGrList
#counts the n gram freequency from given sequence
def getTermFreq(ngram,seq):
	count=0
	tf =0
	num_kmers = len(seq)-len(ngram)+1
	for i in range(num_kmers):
		kmer = seq[i:i+len(ngram)]
		if kmer == ngram:
			count=count+1
	if count !=0:
		tf = float(count)/len(seq)
	return tf
	
#print getTermFreq('ACT', 'ACTACTTTACTAAAC')
#nGramList(3)
#find freequency of each possible n-gram from given sequence
def allNgramWithTermFreq(seq):
	ngrList = nGramList(3)
	freqList = []
	for ngr in ngrList:
		freqList.append((ngr,getTermFreq(ngr,seq)))
	return freqList
#print allNgramWithFreq('ACTGGACCCATACTAGGGACTCCAAATTGACT')

#check if ngram is present in sequence (if present return 1 else 0)
def checkGram(ngram,seq):
	count=0
	num_kmers = len(seq)-len(ngram)+1
	for i in range(num_kmers):
		kmer = seq[i:i+len(ngram)]
		if kmer == ngram:
			count=1
			break
	return count

def getIDF(ngram,seqLis):
	ngCount = 0
	idf = 0
	for seq in seqLis:
		if checkGram(ngram,seq)==1:
			ngCount = ngCount +1
	
	idf =  math.log10(float(len(seqLis))/(ngCount +1))
	return idf
def getListIDF(seqLis):
	lis = []
	i=0
	ngrList = nGramList(3)
	for ngr in ngrList:
		lis.insert(i,getIDF(ngr,seqLis))
	return lis


def getSeqsList(infile):
	Seqs=[]
	k=0
	f= open(infile,'r')
	for line in f:
		
		if re.search('>', line):
			pass
		else:
			Seqs.insert(k,line.strip())
			k=k+1
	return Seqs


def generateTFIDFdataset():
	#inFile = open('cdro3.fasta','r')
	outFile = open('featureTF.txt','w')
	lis = getSeqsList('top10.txt')
	idfList =getListIDF(lis)
	#print idfList
	for line in lis:
		outLine = ''
		#freeqList = [('AAA', 0.1), ('AAC', 0), ('AAT', 0.12)]
		freeqList = allNgramWithTermFreq(line)
		#print freeqList
		i=64
		for (x,y) in freeqList:
			#idf = getIDF(x,lis)
			i-=1
			idf = idfList[i]

			
			tfidf = float(y)*float(idf)
			outLine = outLine+str(round(tfidf,3))+' '
		outFile.write(outLine)
		outFile.write("\n")
	outFile.close()

def cluster_indices(cluster_assignments):
    n = cluster_assignments.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster_assignments == cluster_number)[0])
    return indices

generateTFIDFdataset()
# print 0.18*0.3
# inFile = open('test.txt','r')
# lis = getSeqsList(inFile)
# idf = getIDF('ACT', lis)
# print idf,lis

# feature selection 

X = np.loadtxt('featureTF.txt')
kwargs_W = {"metric":"euclidean","neighbor_mode":"knn","weight_mode":"heat_kernel","k":5,'t':1}
W = construct_W.construct_W(X, **kwargs_W)
score = lap_score.lap_score(X, W=W)
#print score
idx = lap_score.feature_ranking(score)
#print idx
num_fea = 15
selected_features = X[:, idx[0:num_fea]]
#print selected_features

# single linkage clustering
# Compute the clusters.
Z = linkage(X, 'ward')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram( Z
    #leaf_rotation=90.,  # rotates the x axis labels
    #leaf_font_size=7.,  # font size for the x axis labels
    )
plt.show()
max_d = 0.02
#clusters = fcluster(Z, max_d, criterion='distance')
clusters =fcluster(Z, 10, criterion='maxclust')
#print clusters
clabel = [ int(t) for t in clusters]
############################
cutoff = 0.99
cluster_assignments = fclusterdata(selected_features, cutoff)
#print cluster_assignments
# Print the indices of the data points in each cluster.
num_clusters = cluster_assignments.max()
#print "%d clusters" % num_clusters
# indices = cluster_indices(cluster_assignments)
# for k, ind in enumerate(indices):
#     print "cluster", k + 1, "is", ind
###########################################
#applying pca for visualization


def pca(X = np.array([]), no_dims = 2):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print "Preprocessing the data using PCA..."
	(n, d) = X.shape;
	X = X - np.tile(np.mean(X, 0), (n, 1));
	(l, M) = np.linalg.eig(np.dot(X.T, X));
	Y = np.dot(X, M[:,0:no_dims]);
	#print Y
	return Y;
# Xpca = pca(selected_features,2)
# plt.scatter(Xpca[:,0], Xpca[:,1], 100, clabel);
# plt.show();
###############################################
#calculation of nmi score
label_true= open('top10label.0', 'r')
tlabel = [ int(t) for t in label_true]
plabel = [ int(t) for t in cluster_assignments]
tCount = Counter(tlabel)
t10 = tCount.most_common(10)
print t10

pCount = Counter(plabel)
# p10 = pCount.most_common(10)
#print pCount
cCount = Counter(clabel)
print cCount

#print tlabel
#print 'NMI score1 : ', normalized_mutual_info_score(tlabel, plabel)
print 'NMI score2 : ', normalized_mutual_info_score(tlabel, clabel)