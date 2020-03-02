import string
import sys 
import os 
import re
import math
import random 
import json 
def main(model_path,input_path):
	# data_clean = readData(path)
	out_path = "percepoutput.txt"
	with open(model_path) as file:
		model = json.load(file)

	W_pos_neg = model["W_pos_neg"]
	b_pos_neg = model["b_pos_neg"]
	feat_map_pos_neg = model["feat_map_pos_neg"]
	W_tru_dec = model["W_tru_dec"]
	b_tru_dec = model["b_tru_dec"]
	feat_map_tru_dec = model["feat_map_tru_dec"]
	# print(W_pos_neg)


	result = []
	for comment in os.listdir(input_path):
		review_path = input_path+"\\"+comment
		if(not valid_dir(review_path)):
			continue
		for level1 in os.listdir(review_path):
			review_path_level1 = review_path + "\\" + level1
			if(not valid_dir(review_path_level1)):
				continue
			for level2 in os.listdir(review_path_level1):
				review_path_level2 = review_path_level1 + "\\" + level2
				if(not valid_dir(review_path_level2)):
					continue
				for comments in os.listdir(review_path_level2):
					comment_path = review_path_level2 + "\\" + comments
					if(not valid_file(comment_path)):
						continue
					with open(comment_path,'r') as file:
						review = file.read()
						# print(review)
						label_1 = predict(review,W_pos_neg,b_pos_neg,feat_map_pos_neg,["positive","negative"])
						label_2 = predict(review,W_tru_dec,b_tru_dec,feat_map_tru_dec,["truthful","deceptive"])

						# label_1 = predict(review,train_result_1,prior_1,feature_map_1,["pos","neg"])
						# label_2 = predict(review,train_result_2,prior_2,feature_map_2,["tru","dec"])
						
						# label_1 = "positive" if label_1=="pos" else "negative"
						# label_2 = "truthful" if label_2=="tru" else "deceptive"
						result.append(label_2+" "+label_1+" "+comment_path+"\n")
						writeOutpue(out_path,result)
def valid_dir(filename):
	return os.path.isdir(filename)
def valid_file(filename):
	return os.path.isfile(filename)
def writeOutpue(out_path,result):
	with open(out_path,"w+") as file:
		for res in result:
			file.write(res)

def predict(review,W,b,feature_map,classes):
	review_cleaned = list(map(mapper,[review]))[0]
	X = [0 for i in range(len(feature_map))]
	for token in review_cleaned:
		if(token in feature_map):
			X[feature_map[token]]+=1

	Y = dot_product(W,X)+b
	if(Y>0):
		return classes[0]
	else:
		return classes[1]
	# print(review_cleaned)

def dot_product(W,X):
	result = 0
	for i in range(len(W)):
		if(W[i]==0 or X[0]==0):
			continue
		result += W[i]*X[i]
	return result
def unpack(train_result):
	feature_map = train_result["feature"]
	prior = train_result["prior"]
	del train_result["feature"]
	del train_result["prior"]
	return train_result,prior,feature_map


def readData(path):
	# print(os.listdir(path))
	data = []
	for comment in os.listdir(path):
		# print(comment)
		review_path = path+"\\"+comment
		with open(review_path,'r') as file:
			data.append(file.read())
	data_clean = list(map(mapper,data))
	return data_clean
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
def removePun(s):
	trans = str.maketrans(string.punctuation," "*len(string.punctuation))
	return s.translate(trans)
model_path = sys.argv[1]
input_path = sys.argv[2]
main(model_path,input_path)