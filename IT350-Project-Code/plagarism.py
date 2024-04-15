import nltk
from nltk.corpus import brown,stopwords,wordnet
from gensim.models import Word2Vec
import numpy as np
import os
import re

from nltk.stem import WordNetLemmatizer, PorterStemmer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

brown_corpus = brown.sents()+brown.words()

# brown_corpus = [[stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in sentence if word.lower() not in stopwords.words('english') and re.match(r'^[a-zA-Z]+$', word)] for sentence in brown_corpus]

# model = Word2Vec(brown_corpus, vector_size=300, window=10, min_count=10, epochs=400)

# model.save("model.bin")

model = Word2Vec.load("trained_model.bin")


suspect_documents = ["answer-scripts/student1.txt","answer-scripts/student2.txt","answer-scripts/student3.txt","answer-scripts/student4.txt","answer-scripts/student5.txt","answer-scripts/student6.txt","answer-scripts/student7.txt","answer-scripts/student8.txt","answer-scripts/student9.txt","answer-scripts/student10.txt","answer-scripts/student11.txt","answer-scripts/student12.txt","answer-scripts/student13.txt","answer-scripts/student14.txt","answer-scripts/student15.txt"]

suspect_vectors = []
for suspect_document in suspect_documents:
    with open(suspect_document, 'r') as file:
        content = file.read()
    sentences = nltk.sent_tokenize(content.lower())
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    synonyms_sentences = []
    for sentence in sentences:
        synonyms = []
        for word in sentence:
            synsets = wordnet.synsets(word)
            if synsets:
                synonyms.extend(synsets[0].lemmas())
        synonyms = list(set([synonym.name().replace('_', ' ') for synonym in synonyms]))
        sentence_with_synonyms = sentence + synonyms
        synonyms_sentences.append(sentence_with_synonyms)
    synonyms_sentences = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in synonyms_sentences]
    synonyms_sentences = [[stemmer.stem(word) for word in sentence] for sentence in synonyms_sentences]
    stop_words = set(stopwords.words('english'))
    synonyms_sentences = [[word for word in sentence if word not in stop_words] for sentence in synonyms_sentences]
    vector = np.mean([model.wv.get_vector(word) for sentence in synonyms_sentences for word in sentence if word in model.wv.index_to_key], axis=0)
    suspect_vectors.append(vector)

source_documents = ["answer-scripts/student1.txt","answer-scripts/student2.txt","answer-scripts/student3.txt","answer-scripts/student4.txt","answer-scripts/student5.txt","answer-scripts/student6.txt","answer-scripts/student7.txt","answer-scripts/student8.txt","answer-scripts/student9.txt","answer-scripts/student10.txt","answer-scripts/student11.txt","answer-scripts/student12.txt","answer-scripts/student13.txt","answer-scripts/student14.txt","answer-scripts/student15.txt"]

similarities = []
for suspect_vector in suspect_vectors:
    temp_similarities = []
    for source_document in source_documents:
        with open(source_document, 'r') as file:
            content = file.read()
        sentences = nltk.sent_tokenize(content.lower())
        sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        synonyms_sentences = []
        for sentence in sentences:
            synonyms = []
            for word in sentence:
                synsets = wordnet.synsets(word)
                if synsets:
                    synonyms.extend(synsets[0].lemmas())
            synonyms = list(set([synonym.name().replace('_', ' ') for synonym in synonyms]))
            sentence_with_synonyms = sentence + synonyms
            synonyms_sentences.append(sentence_with_synonyms)
        synonyms_sentences = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in synonyms_sentences]
        synonyms_sentences = [[stemmer.stem(word) for word in sentence] for sentence in synonyms_sentences]
        stop_words = set(stopwords.words('english'))
        synonyms_sentences = [[word for word in sentence if word not in stop_words] for sentence in synonyms_sentences]
        source_vector = np.mean([model.wv.get_vector(word) for sentence in synonyms_sentences for word in sentence if word in model.wv.index_to_key], axis=0)
        similarity = np.dot(source_vector, suspect_vector)/(np.linalg.norm(source_vector)*np.linalg.norm(suspect_vector))
        temp_similarities.append(similarity)
    similarities.append(temp_similarities)

for i, row in enumerate(similarities):
    print(f"Similarity scores for student_document_{i+1}:")
    for j, score in enumerate(row):
        print(f"  {source_documents[j]}: {int(score*100)}")
    print("--------------------------------------------------------------------------------")


# print(similarities)

