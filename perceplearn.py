import string
import sys 
import os 
import re
import math
import random 
import json 
import copy
def readData(path):
	neg_dec_path = path+ "\\negative_polarity\\deceptive_from_MTurk"
	neg_tru_path = path+"\\negative_polarity\\truthful_from_Web"
	pos_dec_path = path+ "\\positive_polarity\\deceptive_from_MTurk"
	pos_tru_path = path+ "\\positive_polarity\\truthful_from_TripAdvisor"

	neg_dec = []
	neg_tru = []
	pos_dec = []
	pos_tru = []

	neg_dec_train = []
	neg_tru_train = []
	pos_dec_train = []
	pos_tru_train = []
	neg_dec_dev = []
	neg_tru_dev = []
	pos_dec_dev = []
	pos_tru_dev = []
	for folder in os.listdir(neg_dec_path):
		if(folder[0]!="f"):
			continue
		else:
			path_to_folder = neg_dec_path+"\\"+folder
			for review in os.listdir(path_to_folder):
				path_to_txt = path_to_folder+"\\"+review
				neg_dec.append(readtext(path_to_txt))
				if(folder!="fold1"):
					neg_dec_train.append(readtext(path_to_txt))
				else:
					neg_dec_dev.append(readtext(path_to_txt))
	for folder in os.listdir(neg_tru_path):
		if(folder[0]!="f"):
			continue
		else:
			path_to_folder = neg_tru_path+"\\"+folder
			for review in os.listdir(path_to_folder):
				path_to_txt = path_to_folder+"\\"+review
				neg_tru.append(readtext(path_to_txt))
				if(folder!="fold1"):
					neg_tru_train.append(readtext(path_to_txt))
				else:
					neg_tru_dev.append(readtext(path_to_txt))
	for folder in os.listdir(pos_dec_path):
		if(folder[0]!="f"):
			continue
		else:
			path_to_folder = pos_dec_path+"\\"+folder
			for review in os.listdir(path_to_folder):
				path_to_txt = path_to_folder+"\\"+review
				pos_dec.append(readtext(path_to_txt))
				if(folder!="fold1"):
					pos_dec_train.append(readtext(path_to_txt))
				else:
					pos_dec_dev.append(readtext(path_to_txt))
	for folder in os.listdir(pos_tru_path):
		if(folder[0]!="f"):
			continue
		else:
			path_to_folder = pos_tru_path+"\\"+folder
			for review in os.listdir(path_to_folder):
				path_to_txt = path_to_folder+"\\"+review
				pos_tru.append(readtext(path_to_txt))
				if(folder!="fold1"):
					pos_tru_train.append(readtext(path_to_txt))
				else:
					pos_tru_dev.append(readtext(path_to_txt))
	# print(neg_dec_train)
	n_pos_neg = 800
	n_tru_dec = 1000
# 	for n_pos_neg in range(500,1001,50):
# 		W_pos_neg,b_pos_neg,feat_map_pos_neg = pos_neg_classification(neg_dec_train,neg_tru_train,pos_dec_train,pos_tru_train,n_pos_neg,average=True)
# 		# W_tru_dec,b_tru_dec,feat_map_tru_dec = tru_dec_classification(neg_dec_train,neg_tru_train,pos_dec_train,pos_tru_train,n_tru_dec)
# #### pos_neg validation error #####
# 		neg_dev =  neg_dec_dev+neg_tru_dev
# 		pos_dev =  pos_dec_dev+pos_tru_dev
# 		neg_dev_cleaned = list(map(mapper,neg_dev))
# 		pos_dev_cleaned = list(map(mapper,pos_dev))
# 		data_feature = buildFeature(feat_map_pos_neg,pos_dev_cleaned,neg_dev_cleaned,1,-1)
# 		error_pos_neg = prediction(data_feature,W_pos_neg,b_pos_neg)
# 		print(error_pos_neg)
# 		print("featrue number: "+str(n_pos_neg))
#### tru_dec validation error #####
	# for n_tru_dec in range(500,1001,50):
	# 	W_tru_dec,b_tru_dec,feat_map_tru_dec = tru_dec_classification(neg_dec_train,neg_tru_train,pos_dec_train,pos_tru_train,n_tru_dec,average=True)
	# 	dec_dev =  pos_dec_dev+neg_dec_dev
	# 	tru_dev =  pos_tru_dev+neg_tru_dev
	# 	dec_dev_cleaned = list(map(mapper,dec_dev))
	# 	tru_dev_cleaned = list(map(mapper,tru_dev))
	# 	data_feature = buildFeature(feat_map_tru_dec,tru_dev_cleaned,dec_dev_cleaned,1,-1)
	# 	error_pos_neg = prediction(data_feature,W_tru_dec,b_tru_dec)
	# 	print(error_pos_neg)
	# 	print("featrue number: "+str(n_tru_dec))
#### testing and write it to a file ##### 
	
	n_pos_neg = 800
	n_tru_dec = 1000
	#### vanilla ####
	Map = {}
	W_pos_neg,b_pos_neg,feat_map_pos_neg = pos_neg_classification(neg_dec_train,neg_tru_train,pos_dec_train,pos_tru_train,n_pos_neg,average=False)
	W_tru_dec,b_tru_dec,feat_map_tru_dec = tru_dec_classification(neg_dec_train,neg_tru_train,pos_dec_train,pos_tru_train,n_tru_dec,average=False)
	Map["W_pos_neg"] = W_pos_neg
	Map["b_pos_neg"] = b_pos_neg
	Map["feat_map_pos_neg"] = feat_map_pos_neg
	Map["W_tru_dec"] = W_tru_dec
	Map["b_tru_dec"] = b_tru_dec
	Map["feat_map_tru_dec"] = feat_map_tru_dec
	writeFile("vanillamodel.txt",Map)
	#### average ####
	Map = {}
	W_pos_neg,b_pos_neg,feat_map_pos_neg = pos_neg_classification(neg_dec_train,neg_tru_train,pos_dec_train,pos_tru_train,n_pos_neg,average=True)
	W_tru_dec,b_tru_dec,feat_map_tru_dec = tru_dec_classification(neg_dec_train,neg_tru_train,pos_dec_train,pos_tru_train,n_tru_dec,average=True)
	Map["W_pos_neg"] = W_pos_neg
	Map["b_pos_neg"] = b_pos_neg
	Map["feat_map_pos_neg"] = feat_map_pos_neg
	Map["W_tru_dec"] = W_tru_dec
	Map["b_tru_dec"] = b_tru_dec
	Map["feat_map_tru_dec"] = feat_map_tru_dec
	writeFile("averagedmodel.txt",Map)

def pos_neg_classification(neg_dec,neg_tru,pos_dec,pos_tru,n,average = False):
	neg =  neg_dec+neg_tru
	pos =  pos_dec+pos_tru
	neg_cleaned = list(map(mapper,neg))
	pos_cleaned = list(map(mapper,pos))
	unique_word = set() ### set to store unique words
	for review in neg_cleaned:
		for word in review:
			unique_word.add(word)
	for review in pos_cleaned:
		for word in review:
			unique_word.add(word)
	unique_word = sorted(unique_word)
	MI_word = sorted(MutualInfo(pos_cleaned,neg_cleaned,unique_word),key = lambda x :x[1],reverse = True)[0:n]
	# print(MI_word)
	feature = sorted(MI_word,key = lambda x:x[0])
	feature_map = {}#### feature map that holds the hash value #####
	for i in range(len(feature)):
		feature_map[feature[i][0]] = i
	data_feature = buildFeature(feature_map,pos_cleaned,neg_cleaned,1,-1)
	# print(data_feature)
	W = [0 for i in range(len(feature_map))]
	b = 0
	W,b = perceptrons(W,b,data_feature)
	error_rate = prediction(data_feature,W,b)
	# print(W)
	# print(b)
	# print(error_rate)
	return W,b,feature_map
def tru_dec_classification(neg_dec,neg_tru,pos_dec,pos_tru,n,average = False):
	tru = pos_tru+neg_tru
	dec = pos_dec+neg_dec
	tru_cleaned = list(map(mapper,tru))
	dec_cleaned = list(map(mapper,dec))
	unique_word = set()
	for review in dec_cleaned:
		for word in review:
			unique_word.add(word)
	for review in tru_cleaned:
		for word in review:
			unique_word.add(word)
	unique_word = sorted(unique_word)
	MI_word = sorted(MutualInfo(tru_cleaned,dec_cleaned,unique_word),key = lambda x :x[1],reverse = True)[0:n]
	feature = sorted(MI_word,key = lambda x:x[0])
	feature_map = {}#### feature map that holds the hash value #####
	for i in range(len(feature)):
		feature_map[feature[i][0]] = i
	data_feature = buildFeature(feature_map,tru_cleaned,dec_cleaned,1,-1)
	W = [0 for i in range(len(feature_map))]
	b = 0
	W,b = perceptrons(W,b,data_feature)
	error_rate = prediction(data_feature,W,b)
	# print(W)
	# print(b)
	# print(error_rate)
	return W,b,feature_map
def perceptrons(W,b,data_feature,average = False):
	c = 1
	u = copy.deepcopy(W)
	beta = 0
	for j in range(100):
		random.shuffle(data_feature)
		for i in range(len(data_feature)):
			label = data_feature[i][1]
			X = data_feature[i][0]
			a = dot_product(W,X) + b
			if(label*a<=0):
				W = matrix_addition(W,matrix_element_wise_multi(X,label))
				b+=label
				if(average):
					u = matrix_addition(u,matrix_element_wise_multi(matrix_element_wise_multi(X,label)),c)
					beta = beta + label*c
			c+=1
	if(average):
		return matrix_addition(W,matrix_element_wise_multi(u,-(1/c))),b - (1/c)*beta
	else:
		return W,b
def prediction(data_feature,W,b):
	mistake = 0
	for i in range(len(data_feature)):
		X = data_feature[i][0]
		Y = data_feature[i][1]
		a = dot_product(W,X) + b
		if(a*Y<=0):
			mistake+=1
	return mistake/len(data_feature)
def dot_product(W,X):
	result = 0
	for i in range(len(W)):
		if(W[i]==0 or X[0]==0):
			continue
		result += W[i]*X[i]
	return result
def matrix_addition(A,B):
	result = []
	for i in range(len(A)):
		result.append(A[i]+B[i])
	return result
def matrix_element_wise_multi(X,a):
	return list(map(lambda x:x*a,X))
def buildFeature(feature_map,pos_cleaned,neg_cleaned,class1,class2):
	data_num = []
	for review in pos_cleaned:
		feat = [0 for i in range(len(feature_map))]
		for word in review:
			if(word in feature_map):
				feat[feature_map[word]]+=1
		data_num.append((feat,class1))
	for review in neg_cleaned:
		feat = [0 for i in range(len(feature_map))]
		for word in review:
			if(word in feature_map):
				feat[feature_map[word]]+=1
		data_num.append((feat,class2))
	return data_num
def mapper(review):
	result = []
	review = re.sub('[0-9]','',review)
	review_no_pun = removePun(review).lower().split(" ")
	stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
	for word in review_no_pun:
		if(word in stopwords):
			continue
		if(word != ""):
			result.append(word)
	return result
def writeFile(outputname,output):
	with open(outputname,"w") as file:
		json.dump(output,file)
def removePun(s):
	trans = str.maketrans(string.punctuation," "*len(string.punctuation))
	return s.translate(trans)
def MutualInfo(pos_cleaned,neg_cleaned,unique_word):
	result = []
	for word in unique_word:
		N11 = 0
		for review in pos_cleaned:
			if word in review:
				N11+=1
		N01 = len(pos_cleaned)-N11

		N10 = 0 
		for review in neg_cleaned:
			if(word in review):
				N10+=1
		N00 = len(neg_cleaned)-N10

		N = N11+N01+N10+N00
		N_1 = N11 + N01
		N1_ = N11 + N10
		N_0 = N10 + N00
		N0_ = N01 + N00

		a = (N*N11)/(N1_*N_1) if (N*N11)/(N1_*N_1)!=0 else 1
		b = (N*N01)/(N0_*N_1) if (N*N01)/(N0_*N_1)!=0 else 1
		c = (N*N10)/(N1_*N_0) if (N*N10)/(N1_*N_0)!=0 else 1 
		d = (N*N00)/(N0_*N_0) if (N*N00)/(N0_*N_0)!=0 else 1 
		# print(a,b,c,d)
		I = (N11/N)*math.log2 (a) + (N01/N)*math.log2(b) + (N10/N)*math.log2(c) + (N00/N)*math.log2(d)
		result.append((word,I))
	return result
def readtext(path):
	result = ""
	with open(path,"r") as file:
		result = file.read()
	return result
input = "op_spam_training_data"
readData(input)