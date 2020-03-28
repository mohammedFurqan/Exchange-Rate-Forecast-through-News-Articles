import re
from gensim import models, corpora
import nltk
# You only need to do this once when you start running
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

import spacy
import gensim
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel

# To save the models
import pickle

start_year = 1981
end_year = 2020

# Combine data from individual files into only giant list
data = []
for k in range(end_year - start_year):
    try:
        yearly_articles = (pd.read_csv('../../data/news/articles/articles_{}.csv'.format(start_year+k)))['article']
        yearly_articles = [re.sub('\s+', ' ', str(sent)) for sent in yearly_articles]
        yearly_articles = [re.sub("\'", "", str(sent)) for sent in yearly_articles]
        for article in yearly_articles:
            data.append(article)
    except:
        continue

NO_DOCUMENTS = len(data)
print(NO_DOCUMENTS)
NUM_TOPICS = 6
STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['new', 'inc', 'like', 'one', 'two', 'inc.', 'nan'])

# ps = PorterStemmer()
lemm = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
 
def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text

# # Build the bigram models
# def make_bigrams(texts):
#     return [bigram_mod[doc] for doc in texts]

# bigram = gensim.models.Phrases(tokenized_data, min_count=5, threshold=100) # higher threshold fewer phrases.
# bigram_mod = gensim.models.phrases.Phraser(bigram)

# data_with_bigrams = make_bigrams(tokenized_data)
 
# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = [clean_text(document) for document in data]

data_lemmatized = lemmatization(tokenized_data, allowed_postags=['NOUN', 'ADJ'])
 
# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(data_lemmatized)
dictionary.save('../saved_model_data/lemmatized_dictionary.dic')
 
# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in data_lemmatized]
corpora.MmCorpus.serialize('../saved_model_data/serialised_corpus.mm', corpus)
 
# Build the LDA model
dictionary = corpora.Dictionary.load('../saved_model_data/lemmatized_dictionary.dic')
mm = corpora.MmCorpus('../saved_model_data/serialised_corpus.mm')
lda_model = models.ldamulticore.LdaMulticore(corpus=mm, random_state=100, num_topics=NUM_TOPICS, id2word=dictionary, workers=2)
# lda_model.save('../saved_model_data/lda.model')
lda_model = models.LdaModel.load('../saved_model_data/lda.model')


def doc_topics(mapping, num_topics):
    doc_topic_mapping = []
    for index, doc in enumerate(mapping):
        obj = {}
        for i in range(num_topics):
            obj['news_topic#{}'.format(i)] = 0
        for topic in doc:
            obj['news_topic#{}'.format(topic[0])] = 1
        doc_topic_mapping.append(obj)
    return pd.DataFrame(doc_topic_mapping)
document_topics = doc_topics([lda_model.get_document_topics(item) for item in mm], NUM_TOPICS)


documents = pd.DataFrame()
for year in range(end_year - start_year):
    filename = '../../data/news/articles/articles_{}.csv'.format(start_year+year)
    if documents.empty:
        documents = pd.read_csv(filename)
    else:
        documents = pd.concat([documents, pd.read_csv(filename)], sort=False)

documents = documents.reset_index(drop=True)
combined_data = pd.concat([documents,document_topics], axis=1, sort=False).reset_index(drop=True)
combined_data.to_csv('documents_to_topics.csv', index = False)


print("LDA Model:")
 
for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, re.sub('[\"\+\s\d\.*]+', ' ', lda_model.print_topic(idx, 40)))
 
print("- " * 20)
