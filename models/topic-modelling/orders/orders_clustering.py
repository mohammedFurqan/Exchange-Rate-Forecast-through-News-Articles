import re
from gensim import models, corpora
import nltk
from nltk import FreqDist
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


data = (pd.read_csv('../../../data/orders/executive_orders.csv'))['title']
for doc in data:
    doc = (re.sub(r"[\n\d\(\)\[\]:\.\,\"\';]", "", doc)).lower().strip()

giant_string = ""
for doc in data:
    giant_string = giant_string + doc + " "

all_words = giant_string.split() # list of all the words in your corpus
fdist = FreqDist(all_words) # a frequency distribution of words (word count over the corpus)
k = 10000 # say you want to see the top 10,000 words
top_k_words, _ = zip(*fdist.most_common(k)) # unzip the words and word count tuples
# print(top_k_words)

NO_DOCUMENTS = len(data)
STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['executive', 'order'])

# # ps = PorterStemmer()
lemm = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = [clean_text(document) for document in data]
data_lemmatized = lemmatization(tokenized_data, allowed_postags=['NOUN', 'ADJ'])

# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(data_lemmatized)
dictionary.save('lemmatized_dictionary.dic')

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in data_lemmatized]
corpora.MmCorpus.serialize('serialised_corpus.mm', corpus)

# Build the LDA model
dictionary = corpora.Dictionary.load('lemmatized_dictionary.dic')
mm = corpora.MmCorpus('serialised_corpus.mm')
for NUM_TOPICS in range(6,12):
    lda_model = models.ldamulticore.LdaMulticore(corpus=mm, random_state=100, num_topics=NUM_TOPICS, id2word=dictionary, workers=2)
    lda_model.save('lda_{}.model'.format(NUM_TOPICS))
    lda_model = models.LdaModel.load('lda_{}.model'.format(NUM_TOPICS))

    # def doc_topics(mapping, num_topics):
    #     doc_topic_mapping = []
    #     for index, doc in enumerate(mapping):
    #         obj = {}
    #         for i in range(num_topics):
    #             obj['news_topic#{}'.format(i)] = 0
    #         for topic in doc:
    #             obj['news_topic#{}'.format(topic[0])] = 1
    #         doc_topic_mapping.append(obj)
    #     return pd.DataFrame(doc_topic_mapping)
    # document_topics = doc_topics([lda_model.get_document_topics(item) for item in mm], NUM_TOPICS)


    # documents = pd.DataFrame()
    # for year in range(end_year - start_year):
    #     filename = '../../data/executive\ orders/orders_{}.csv'.format(start_year+year)
    #     if documents.empty:
    #         documents = pd.read_csv(filename)
    #     else:
    #         documents = pd.concat([documents, pd.read_csv(filename)], sort=False)
    
    # documents = documents.reset_index(drop=True)
    # combined_data = pd.concat([documents,document_topics], axis=1, sort=False).reset_index(drop=True)
    # combined_data.to_csv('documents_to_topics_{}.csv'.format(NUM_TOPICS), index = False)
    
    print("LDA Model:")
    
    for idx in range(NUM_TOPICS):
        # Print the first 10 most representative topics
        print("Topic #%s:" % idx, re.sub('[\"\+\s\d\.*]+', ' ', lda_model.print_topic(idx, 30)))
    
    print("- " * 20)
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(mm))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
