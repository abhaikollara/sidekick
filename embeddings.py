import numpy as np
from gensim.models import KeyedVectors

class Embedding(object):

    def __init__(self):
        pass
    
    def load_word2vec(self, path, binary=True):
        self._model = KeyedVectors.load_word2vec_format(path, binary=binary)
        self._index2word = self._model.index2word
        self._matrix = self._model.syn0
        self._vocab_size, self._dim = self._matrix.shape

    def load_glove(self, path):
        pass
    
    def get_keras_layer(self, trainable=False):
        try:
            from keras.layers.embeddings import Embedding
        except:
            raise ImportError('Keras not found')
        
        return Embedding(self._vocab_size, self._dim, weights=[self._matrix])
    
    def get_pytorch_layer(self, trainable=False):
        try:
            import torch
            from torch.nn import Embedding
        except:
            raise ImportError('PyTorch not found')

        return Embedding.from_pretrained(torch.FloatTensor(self._matrix))

    @property
    def model(self):
        return self._model
    
    @property
    def matrix(self):
        return self._matrix
    
    @property
    def vocab_size(self):
        return self._vocab_size
    
    @property
    def dim(self):
        return self._dim