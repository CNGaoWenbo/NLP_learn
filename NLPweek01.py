# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:56:47 2019

@author: Gao Wenbo
"""

import numpy as np
from nltk.tokenize import WordPunctTokenizer
f = open("E:/quora.txt",'r', encoding='UTF-8')
data = list(f)
##分词##
tokenizer = WordPunctTokenizer()
data_tok = [tokenizer.tokenize(lines.lower()) for lines in data]#小写化并获取token
#数据预处理检查
assert all(isinstance(row, (list, tuple)) for row in data_tok), "please convert each line into a list of tokens (strings)"
assert all(all(isinstance(tok, str) for tok in row) for row in data_tok), "please convert each line into a list of tokens (strings)"
is_latin = lambda tok: all('a' <= x.lower() <= 'z' for x in tok)
assert all(map(lambda l: not is_latin(l) or l.islower(), map(' '.join, data_tok))), "please make sure to lowercase the data"
#

##词向量##
from gensim.models import Word2Vec
model = Word2Vec(data_tok, 
                 size=100,      # embedding vector size
                 min_count=5,  # consider words that occured at least 5 times
                 window=5).wv  # define context as a 5-word window around the target word
#词频最高的1000个词
words = sorted(model.vocab.keys(), 
               key=lambda word: model.vocab[word].count,
             reverse=True)[:1000]
word_vectors=[]
for word in words:
    try:
        word_vectors.append(model.get_vector(word)) #词向量获取
    except:
        pass
word_vectors=np.array(word_vectors)
#词向量检查
assert isinstance(word_vectors, np.ndarray)
assert word_vectors.shape == (len(words), 100)
assert np.isfinite(word_vectors).all()

##PCA分析##
from sklearn.decomposition import PCA
pca = PCA(n_components=2) #主成分维度
word_vectors_pca = pca.fit_transform(word_vectors)
mini = np.amin(word_vectors_pca,axis=0)
maxi = np.amax(word_vectors_pca,axis=0)
mean = np.mean(word_vectors_pca,axis=0)
c=0
for i in word_vectors_pca[:,0]:
    word_vectors_pca[c,0] = (i-mean[0])/(maxi[0]-mini[0])
    c=c+1
c=0
for j in word_vectors_pca[:,1]:
    word_vectors_pca[c,1] = (i-mean[1])/(maxi[1]-mini[1])
    c=c+1
#PCA检查
assert word_vectors_pca.shape == (len(word_vectors), 2), "there must be a 2d vector for each word"
#assert max(abs(word_vectors_pca.mean(0))) < 1e-5, "points must be zero-centered"
#assert max(abs(1.0 - word_vectors_pca.std(0))) < 1e-2, "points must have unit variance"

##画词向量图##
import bokeh.models as bm, bokeh.plotting as pl
from bokeh.io import output_notebook
output_notebook()   
def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=600, height=400, show=True, **kwargs):
    """ draws an interactive plot for data points with auxilirary info on hover """
    if isinstance(color, str):
        color = [color] * len(x)
    data_source = bm.ColumnDataSource({ 'x' : x, 'y' : y, 'color': color, **kwargs })

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show: 
        pl.show(fig)
    return fig
draw_vectors(word_vectors_pca[:, 0], word_vectors_pca[:, 1], token=words)


