import re
from gensim import models, corpora
# import nltk
# You only need to do this once when you start running
# nltk.download('stopwords')
# nltk.download('punkt')
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer 

# import spacy
import gensim
import pandas as pd

# start_year = 1981
# end_year = 2020

# data = []
# for k in range(end_year - start_year):
#     try:
#         yearly_articles = (pd.read_csv('../../data/news/articles/articles_{}.csv'.format(start_year+k)))['article']
#         yearly_articles = [re.sub('\s+', ' ', str(sent)) for sent in yearly_articles]
#         yearly_articles = [re.sub("\'", "", str(sent)) for sent in yearly_articles]
#         for article in yearly_articles:
#             data.append(article)
#     except:
#         continue
# NO_DOCUMENTS = len(data)
# print(NO_DOCUMENTS)
# NUM_TOPICS = 8
# STOPWORDS = stopwords.words('english')
# STOPWORDS.extend(['new', 'first', 'inc', 'like', 'one', 'two', 'today', 'inc.', 'day', 'year', 'nan', 'last'])

# # ps = PorterStemmer()
# lemm = WordNetLemmatizer()
# nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#     texts_out = []
#     for sent in texts:
#         doc = nlp(" ".join(sent)) 
#         texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
#     return texts_out
 
# def clean_text(text):
#     tokenized_text = word_tokenize(text.lower())
#     cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
#     return cleaned_text

# def make_bigrams(texts):
#     return [bigram_mod[doc] for doc in texts]
 
# # For gensim we need to tokenize the data and filter out stopwords
# tokenized_data = []
# for text in data:
#     tokenized_data.append(clean_text(text))

# # # Build the bigram models
# # bigram = gensim.models.Phrases(tokenized_data, min_count=5, threshold=100) # higher threshold fewer phrases.
# # bigram_mod = gensim.models.phrases.Phraser(bigram)

# # data_with_bigrams = make_bigrams(tokenized_data)

# data_lemmatized = lemmatization(tokenized_data, allowed_postags=['NOUN', 'ADJ'])
 
# # Build a Dictionary - association word to numeric id
# dictionary = corpora.Dictionary(data_lemmatized)
# dictionary.save('../saved_model_data/lemmatized_dictionary.dic')
 
# # Transform the collection of texts to a numerical form
# corpus = [dictionary.doc2bow(text) for text in data_lemmatized]
# corpora.MmCorpus.serialize('../saved_model_data/serialised_corpus.mm', corpus)
 
# Build the LDA model
NUM_TOPICS = 8
dictionary = corpora.Dictionary.load('../saved_model_data/lemmatized_dictionary.dic')
mm = corpora.MmCorpus('../saved_model_data/serialised_corpus.mm')
lda_model = models.ldamulticore.LdaMulticore(corpus=mm, random_state=100, num_topics=NUM_TOPICS, id2word=dictionary, workers=2)

# lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, per_word_topics=True, random_state=100, id2word=dictionary, alpha=0.2, eta=0.2)

# # Build the LSI model
# lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)


print("LDA Model:")
 
for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, re.sub('[\"\+\s\d\.*]+', ' ', lda_model.print_topic(idx, 40)))
 
print("=" * 20)

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
 
# print("LSI Model:")
 
# for idx in range(NUM_TOPICS):
#     # Print the first 10 most representative topics
#     print("Topic #%s:" % idx, lsi_model.print_topic(idx, 10))
 
# print("=" * 20)