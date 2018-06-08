import numpy as np
from gensim.models import KeyedVectors

class Embedding(object):

    def __init__(self):
        pass
    
    def load_word2vec(self, path, binary=True):
        """Load word2vec model from file

        Args:
            path (str): Path to the word2vec file
            binary (bool): Whether the file is in binary format
        """
    
        self._model = KeyedVectors.load_word2vec_format(path, binary=binary)
        self._index2word = self._model.index2word
        self._matrix = self._model.syn0
        self._vocab_size, self._dim = self._matrix.shape

    def load_glove(self, path):
        pass
    
    def get_keras_layer(self, trainable=False, **kwargs):
        """Creates a Keras embedding layer with the loaded vectors
        as weights

        Args:
            trainable (bool): Whether to freeze the layer weights
            **kwargs: Other kwargs to Keras
        """

        try:
            from keras.layers.embeddings import Embedding
        except:
            raise ImportError('Keras not found')
        
        return Embedding(self._vocab_size, self._dim, weights=[self._matrix], trainable=trainable, **kwargs)
    
    def get_pytorch_layer(self, trainable=False):
        """Creates a Pytorch embedding layer with the loaded vectors
        as weights

        Args:
            trainable (bool): Whether to freeze the layer weights
        """

        try:
            import torch
            from torch.nn import Embedding
        except:
            raise ImportError('PyTorch not found')

        return Embedding.from_pretrained(torch.FloatTensor(self._matrix), freeze=trainable)

    @property
    def model(self):
        return self._model
    
    @property
    def matrix(self):
        """np.ndarray: The embedding matrix of shape vocab_size x embedding dim"""
        return self._matrix
    
    @property
    def vocab_size(self):
        """int: Total number of words in the vocabulary"""
        return self._vocab_size
    
    @property
    def dim(self):
        """int: The vector size/embedding dimension"""
        return self._dim