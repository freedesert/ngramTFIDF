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
from tSne import tsne
#generates all possible n-grams from ACTG
def nGramList(n):
	nGrList = []
	base = 'ACTG'
	for i in range(n,n+1):
		nGrList = [''.join(x) for x in itertools.product(base, repeat=i)]
	#print nGrList
	return nGrList
#counts the n gram freequency from given sequence
def getFreq(ngram,seq):
	count=0
	num_kmers = len(seq)-len(ngram)+1
	for i in range(num_kmers):
		kmer = seq[i:i+len(ngram)]
		if kmer == ngram:
			count=count+1
	return count
	
#print getTermFreq('ACT', 'ACTACTTTACTAAAC')
#nGramList(3)
#find freequency of each possible n-gram from given sequence
def allNgramWithFreq(seq):
	ngrList = nGramList(3)
	freqList = []
	for ngr in ngrList:
		freqList.append((ngr,getFreq(ngr,seq)))
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
	
	idf =  math.log10(float(len(seqLis))/(ngCount + 1))
	return idf
def getListIDF(seqLis):
	lis = []
	i=0
	ngrList = nGramList(3)
	for ngr in ngrList:
		lis.insert(i,getIDF(ngr,seqLis))
	return lis


def getSeqsList(infile):
	lis = []
	i = 0
	for line in infile:
		lis.insert(i,line.strip())
		i+=1
	return lis


def generateTFIDFdataset():
	inFile = open('top10.txt','r')
	outFile = open('featureTF.txt','w')
	lis = getSeqsList(inFile)
	#print lis
	idfList =getListIDF(lis)
	#print idfList
	for line in lis:
		outLine = ''
		#freeqList = [('AAA', 0.1), ('AAC', 0), ('AAT', 0.12)]
		freeqList = allNgramWithFreq(line)
		#print freeqList
		#sort the freeqList and find maximum freequency
		maxFreq = max(freeqList,key=lambda item:item[1])
		#print freeqList

		#print maxFreq
		i=64
		for (x,y) in freeqList:

			tf = 0.5 + (float(y)/maxFreq[1])*0.5
			#idf = getIDF(x,lis)
			i-=1
			idf = idfList[i]

			
			tfidf = float(tf)*float(idf)
			outLine = outLine+str(round(tfidf,3))+' '
		outFile.write(outLine)
		outFile.write("\n")
	outFile.close()

def EditDistance(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]

generateTFIDFdataset()
###########################
# s1 ='TGTGCGAAAGGATCCGCAGCAGTTGGTACTCCCGGCGACTGGTACTTTGACTACTGG'
# s2 ='TGTGCGAAAGGACCGCAGCAGTTGGTACTCCCGGCGACTGGTACTTTGACTACTGG'
# print "edit Distance :",EditDistance(s1,s2)
# a = np.loadtxt('featureTF.txt')
# b = linkage(a, 'single')
# print b



###########################
# print 0.18*0.3
# inFile = open('test.txt','r')
# lis = getSeqsList(inFile)
# idf = getIDF('ACT', lis)
# print idf,lis

# feature selection 

X = np.loadtxt('featureTF.txt')
#X =[[1,2,3,4,5,6],[1,2,3,4,5,6],[3,4,5,6,7,8],[4,5,6,7,7,8],[1,3,4,5,6,7]]
#X= [[1],[2],[3],[4],[5],[6]]

kwargs_W = {"metric":"euclidean","neighbor_mode":"knn","weight_mode":"heat_kernel","k":5,'t':1}
W = construct_W.construct_W(X, **kwargs_W)
#print W.shape

score = lap_score.lap_score(X, W=W)
#print score
idx = lap_score.feature_ranking(score)
#print idx
num_fea = 64
selected_features = X[:, idx[0:num_fea]]
#print selected_features

# single linkage clustering
# Compute the clusters.
Z = linkage(X, 'ward')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram( Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=7.,  # font size for the x axis labels
    )
plt.show()
max_d = 0.1
#clusters = fcluster(Z, max_d, criterion='distance')
clusters =fcluster(Z, 10, criterion='maxclust')
print "predicted labels: "
print clusters
clabel = [ int(t) for t in clusters]
############################
#cutoff = 0.99
#cluster_assignments = fclusterdata(selected_features, cutoff)
#print cluster_assignments
# Print the indices of the data points in each cluster.
#num_clusters = cluster_assignments.max()
#print "%d clusters" % num_clusters
# indices = cluster_indices(cluster_assignments)
# for k, ind in enumerate(indices):
#     print "cluster", k + 1, "is", ind

###############################################
#calculation of nmi score
label_true= open('top10label.0', 'r')

tlabel = [ int(t) for t in label_true]
print "true labels:"
print tlabel
#plabel = [ int(t) for t in cluster_assignments]
tCount = Counter(tlabel)
t10 = tCount.most_common(10)
print t10

#pCount = Counter(plabel)
# p10 = pCount.most_common(10)
#print pCount
cCount = Counter(clabel)
print cCount

#print tlabel
#print 'NMI score1 : ', normalized_mutual_info_score(tlabel, plabel)
print 'NMI score2 : ', normalized_mutual_info_score(tlabel, clabel)

############################################
#tSne visualization
# Y = tsne(selected_features, 2, num_fea, 20.0);
# plt.scatter(Y[:,0], Y[:,1], 80, clabel);
# plt.show();