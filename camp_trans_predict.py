import numpy as np
import nltk
from gensim.utils import tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
def get_lemama2(word):
    return WordNetLemmatizer.lemmatize(word)
english_stop_w = set(nltk.corpus.stopwords.words('english'))
common_words = ['pretoria','academic','school','would','since','Polokwane','Pretoria','eMalahleni','Witbank','Soshanguve','Sosha','Campus','campus','transfer','reason']
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in english_stop_w]
    tokens = [token for token in tokens if token not in common_words]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

def prepare_data():
    motivation_label_list = []
    with open("C:\\Users\\DlaminiN3\\Desktop\\work_research\\merged_raw.csv") as fl:
        for line in fl:
            field = line.split(";")
            motivation = ""
            label = ""
            try:
                motivation = field[1]
                label = field[2]
            except:
                continue
            motivation_list = prepare_text_for_lda(motivation)
            current_line = ""
            for m in motivation_list:
                current_line +=" "+m

            line = current_line+","+label
            motivation_label_list.append(line)

    with open("C:\\Users\\DlaminiN3\\Desktop\\work_research\\merged_normalized.csv",'a+') as file_write:
        for line in motivation_label_list:
            file_write.write(line+'\n')

def train_model():

    data = pd.read_csv("C:\\Users\\DlaminiN3\\Desktop\\work_research\\merged_normalized.csv",encoding='latin-1')
    data.head()
    data.describe()
    reg = LinearRegression()

    labels = data['Outcome']

    train1 = data.drop(['Outcome'],axis=1)
    print(train1.shape)
    print(labels.shape)

    x_train, x_test,y_train,y_test = train_test_split(train1,labels,test_size=0.10,random_state=2)
    #print(len(labels))
    #print(len(train1))
    count_vector = CountVectorizer()
    x_train_counts = count_vector.fit_transform(x_train)
    x_test_counts = count_vector.fit_transform(x_test)
    x_test_counts = count_vector.fit_transform(x_test)
    y_train_counts=count_vector.fit_transform(y_train)
    y_test_counts=count_vector.fit_transform(y_test)
    print(x_train_counts.shape)
    print(y_train_counts.shape)


    #data.reshape((999, 1)) data.values.reshape

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf=tfidf_transformer.fit_transform(x_train_counts)
    x_test_tfidf =tfidf_transformer.fit_transform(x_test_counts)
    y_train_tfidf=tfidf_transformer.fit_transform(y_train_counts)
    y_test_tfidf=tfidf_transformer.fit_transform(y_test_counts)
    print(x_train_tfidf.shape)
    print(y_train_tfidf.shape)

    #clf = MultinomialNB().fit(x_train_tfidf, y_train_tfidf)
    #reg.fit(x_train_tfidf,y_train_counts)
    #reg.score(x_test_tfidf,y_test_counts)
#prepare_data()
train_model()







