
import nltk
from gensim.utils import tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import gensim

from gensim import corpora
import pyLDAvis.gensim

import pickle



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
text_data =[]
with open('C:\\Users\\DlaminiN3\\Desktop\\data.csv') as f:
    for line in f:
        tokens = prepare_text_for_lda(line)
        text_data.append(tokens)
dictionary = corpora.Dictionary(text_data)

corpus = [dictionary.doc2bow(text) for text in text_data]
pickle.dump(corpus,open('corpus.pkl','wb'))
dictionary.save('dictionary.gensim')

NUM_TOPICS =5
ldamodel = gensim.models.ldamodel.LdaModel(corpus,num_topics=NUM_TOPICS,id2word=dictionary,passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)
pyLDAvis.enable_notebook()
ldamodel10 = gensim.models.ldamodel.LdaModel(corpus,num_topics=10,id2word=dictionary,passes=15)
#ldamodel7 = gensim.models.ldamodel.LdaModel(corpus,num_topics=7,id2word=dictionary,passes=15)

lda_display10 = pyLDAvis.gensim.prepare(ldamodel10,corpus,dictionary,sort_topics=False)
#pyLDAvis.prepared_data_to_html(lda_display10)
pyLDAvis.show(lda_display10)
#pyLDAvis.display(lda_display10)
















