import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import nltk
import string
import sklearn
import sys
import psycopg2
conn = psycopg2.connect("host=localhost dbname=db_PDB user=postgres password=postgresql")
cur = conn.cursor()

#load dataset from csv file
df = pd.read_csv('D:\\Learning\\python\\project\\data_input\\data_input.csv', encoding='latin-1')
#df=df[['v1','v2']]
#df.head(3)
df.columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)

df.rename({"v1":"label","v2":"message"},axis=1,inplace=True)
df

## To check the the dataset is Balanced or Imbalaced 
df["label"].value_counts()

#print(bal)

import seaborn as sns
sns.countplot(x=df["label"])

import re
from nltk.corpus import stopwords
from nltk.stem import *

corpus=[]
stemmer=PorterStemmer()

corpus=[]
for i in range(len(df['message'])):
    review=re.sub('[^a-zA-Z]',' ',df['message'][i]) ## First remove the non word characters
    review=review.lower()
    review=review.split()
    review=[stemmer.stem(word) for word in review if not word in stopwords.words("english")]
    review=" ".join(review)
    corpus.append(review)

corpus[5]
cp=pd.DataFrame(corpus)
#print(cp)
cp.to_csv('D:\\Learning\\python\\project\\data_output\\data_corpus.csv', index=True, encoding='utf-8')
#insert data result corpus
cur.execute('truncate table public.tbl_klasifikasi_text;')
with open('D:\\Learning\\python\\project\\data_output\\data_corpus.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip the header row.
    for row in reader:
        cur.execute(
        "INSERT INTO public.tbl_klasifikasi_text VALUES (%s, %s)",
        row
    )
conn.commit()

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=5000) 
## we are just trying top 5000 Words,then we can tune it accoring to accuracy and other matrix
X = vectorizer.fit_transform(corpus).toarray()
sk = X,X.shape
sys.stdout = open("D:\Learning\python\project\data_output\data_vector.txt", "w")
print(sk)
sys.stdout.close()

x = df["label"].unique()

y=pd.get_dummies(df["label"])
#print(f'print1 {y}')
sys.stdout = open("D:\Learning\python\project\data_output\data_model.txt", "w")
rpt = print(y)
sys.stdout.close()

df.to_csv('D:\\Learning\\python\\project\\data_output\\dataset.csv', index=True, encoding='utf-8')

y.to_csv('D:\\Learning\\python\\project\\data_output\\data_model_clasifikasi.csv', index=True, encoding='utf-8')
#insert data klasifikasi
cur.execute('truncate table public.tbl_klasifikasi;')
#load Dataset
with open('D:\\Learning\\python\\project\\data_output\\data_model_clasifikasi.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip the header row.
    for row in reader:
        cur.execute(
        "INSERT INTO public.tbl_klasifikasi VALUES (%s, %s, %s)",
        row
    )
conn.commit()

## Now we are getting the Spam Column into the model
y=pd.get_dummies(df["label"],drop_first=True)
#print(f'print 2 : {y}')
sys.stdout = open("D:\Learning\python\project\data_output\data_model_1.txt", "w")
rpt = print(y)
sys.stdout.close()

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

## To get unique values in test data and total size of test data
y_test.value_counts(),len(y_test)

from sklearn.metrics import *

acc=accuracy_score(y_test, y_pred)
cm=confusion_matrix(y_test, y_pred)
a=precision_recall_fscore_support(y_test, y_pred, average='binary')

#Generate report to txt file
sys.stdout = open("D:\Learning\python\project\data_output\Gaus_rpt.txt", "w")
print(f'Hasil Matrix Model Gaussian Naive Bayes, \n\n Akurasi: {acc} ,\n\n Confusion Matrix: \n\n{cm}, \n\n Precision, Recall, F-1 score for Binary Average of data are \n{a}\n\n')
target_names = ['ham', 'spam']
report = classification_report(y_test, y_pred, target_names=target_names,labels=None)
rpt = print(f"Classification report: \n{report}")
sys.stdout.close()

#akurasi file
sys.stdout = open("D:\Learning\python\project\data_output\.akurasi_result.csv", "w")
rpt = print(f'Akurasi: {acc}')
sys.stdout.close()

#Generate Report to scv file
sys.stdout = open("D:\Learning\python\project\data_output\Gaus_rpt.csv", "w")
rpt = print(f"Classification report for Gaussian Naive Bayes \n\n{report}")
sys.stdout.close()

#clean data table before insert
#cur.execute('truncate table public.tbl_result;')
#insert result data to table
