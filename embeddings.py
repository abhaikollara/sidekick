import numpy as np
from gensim.models import KeyedVectors

class Embedding(object):

    def __init__(self, matrix=None, index2word=None):
        self._matrix = matrix
        self._index2word = index2word

        if matrix is not None:
            if index2word is None:
                raise TypeError('index2word cannot be None if matrix is provided')
            if matrix.shape[0] != len(index2word):
                raise ValueError('Embedding matrix and index2word contain unequal number of items')
    
    def load_word2vec(self, path, binary=True):
        """Load word2vec model from file

        Args:
            path (str): Path to the word2vec file
            binary (bool): Whether the file is in binary format
        """
    
        self._model = KeyedVectors.load_word2vec_format(path, binary=binary)
        self._index2word = self._model.index2word
        self._matrix = self._model.syn0

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
    
    def create_subset(self, index2word):
        indices = [self._model.vocab[word].index for word in index2word]
        matrix = self.matrix[indices]
        return Embedding(matrix=matrix, index2word=index2word)

    @property
    def model(self):
        return self._model
    
    @property
    def matrix(self):
        """np.ndarray: The embedding matrix of shape vocab_size x embedding dim"""
        return self._matrix
    
    @property
    def index2word(self):
        """list: A list of all words in the vocab"""
        return self._index2word

    @property
    def vocab_size(self):
        """int: Total number of words in the vocabulary"""
        return self._matrix.shape[0]
    
    @property
    def dim(self):
        """int: The vector size/embedding dimension"""
        return self._matrix.shape[1]