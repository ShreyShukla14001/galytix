import pandas as pd
import setup
import gensim
from gensim.models import KeyedVectors
import numpy as np
from scipy.spatial.distance import cosine
df = pd.read_csv('C:/Users/shrey/Downloads/DE assessment/phrases (1).csv',encoding='unicode_escape')
wv=KeyedVectors.load_word2vec_format('vectors.csv')
def assign_embeddings(df):
    phrase_embeddings=[]
    for index,rows in df.itterrows():
        phrase=rows['Phrases']
        words=phrase.split()
        word_embeddings=[]
        for word in words:
            if word in wv:
                word_embeddings.append(wv[word])
        if word_embeddings:
            phrase_embedding=sum(word_embeddings)/len(word_embeddings)
            phrase_embeddings.append(phrase_embedding)
        else:
            phrase_embeddings.append(None)
    return phrase_embeddings
def cosine_distance(df):
    phrase_embeddings=assign_embeddings(df)
    num_phrases = len(phrase_embeddings)
    l2_distances = np.zeros((num_phrases, num_phrases))
    cosine_distances = np.zeros((num_phrases, num_phrases))
    phrase_embeddings=assign_embeddings(df)
    num_phrases=len(phrase_embeddings)
    # Calculate distances for all pairs of phrases
    cosine_distances=np.zeores((num_phrases,num_phrases))
    for i in range(num_phrases):

        for j in range(i, num_phrases):
            # Calculate Cosine distance
            cosine_dist = cosine(phrase_embeddings[i], phrase_embeddings[j])
            cosine_distances[i][j] = cosine_dist
            cosine_distances[j][i] = cosine_dist

    return cosine_distances

def find_closest_match(input_string,df):
    phrase_embeddings=assign_embeddings(df)

    input_embeddings=[]
    input_words=input_string.split()

    for word in input_words:
        if word in wv:
            input_embeddings.append(wv[word])
    if not input_embeddings:
        return "No matching phrase"
    input_embedding=sum(input_embeddings)/len(input_embeddings)
    cosine_distances=[cosine(input_embedding,phrase_emb) for phrase_emb in phrase_embeddings]
    closest_match_index=np.argmin(cosine_distances)
    closest_match_phrase=df['Phrases'][closest_match_index]
    closest_match_distance=cosine_distances[closest_match_index]

    return closest_match_phrase,closest_match_distance


