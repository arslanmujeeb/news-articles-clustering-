# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 01:20:39 2018

@author: Arslan Mujeeb
"""

import os
import glob
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

file_list=glob.glob(os.path.join(os.getcwd(),"clustering","*.txt"))

#Y=os.listdir("clustering")
#print(Y)                        #print input files

arg=[]    
test=[]
for file_path in file_list:
     with open(file_path) as f_input:
         arg.append(f_input.read())
print("\n********************************************")
print("WE HAVE A CORPUS OF====> ",len(arg),"FILES")



vectorizer=TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')
X=vectorizer.fit_transform(arg)
#LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
#X=LemVectorizer.fit_transform(X)


#print(X)                 #printing the tfidf vector


##labels = vectorizer.get_feature_names() 
#print(labels)                   #this prints the names terms of respective tfidf value


#slabels=vectorizer.get_stop_words() 
#print(slabels)                  #this prints the words that were stopped

#print(len(labels))        
#km=KMeans(n_clusters = 5, init='k-means++', max_iter=100,n_init=1, verbose = True)
#km.fit(X)

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(X)
y=km.cluster_centers_
print("\nTHE CENTROIDS ARE \n\n",y)
clusters = km.labels_.tolist()
print(clusters)


#testing..................................!

file_=glob.glob(os.path.join(os.getcwd(),"test","abcc.txt"))
for file_pat in file_:
     with open(file_pat) as f:
          y=f.read()

arg.insert(0,y)  
Z=vectorizer.fit_transform(arg) #HERE THE TFIDF OF TEXT FILE IS CALCULATED

y_kmeans = km.predict(Z[0]) 
  
print("\n\n ALGORITHM HAS PREDICTED THAT THE GIVEN TEST FILE BELONGS TO CLUSTER NO.===>",y_kmeans) 

#joblib.dump(km,  'doc_cluster.pkl')

#km = joblib.load('doc_cluster.pkl')
#clusters = km.labels_.tolist()
#print(clusters)
