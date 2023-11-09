import pandas as pd
import gensim
from gensim.models import KeyedVectors
df = pd.read_csv('C:/Users/shrey/Downloads/DE assessment/phrases (1).csv',encoding='unicode_escape')
pth="C:/Users/shrey/Downloads/archive (4)/GoogleNews-vectors-negative300.bin.gz"
wv=KeyedVectors.load_word2vec_format(pth,binary=True,limit=1000000)
wv.save_word2vec_format('vectors.csv')



