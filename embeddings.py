import numpy as np
from gensim.models import KeyedVectors

class Embedding(object):

    def __init__(self, matrix=None, index2word=None):
        """Initialize an embedding optionally by providing the embedding matrix and index2word

        Args:
            matrix (np.ndarray, optional): The embedding matrix of the embedding
            index2word (list, optional): A list of words in the order of corresponding vectors in the matrix 
        """

        self._matrix = matrix
        self._index2word = index2word
        if index2word is not None:
            self._index_dict = {index2word[i]:i for i in range(len(index2word))}

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
        self._matrix = self._model.syn0
        self._index2word = self._model.index2word
        self._index_dict = {self._index2word[i]:i for i in range(len(self._index2word))}

    def load_glove(self, path, vocab_size=None, dim=None):
        if vocab_size is None:
            vocab_size = 0
            with open(path, 'r') as f:
                for line in f:
                    vocab_size += 1

        if dim is None:
            with open(path, 'r') as f:
                dim = len(f.readline().split(u' ')) - 1

        self._matrix = np.zeros((vocab_size, dim))
        self._index2word = []

        update = self._index2word.append #Speedup
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                split = line.split(u' ')
                update(split[0])
                self._matrix[i] = np.asarray([float(val) for val in split[1:]])

        self._index_dict = {self._index2word[i]:i for i in range(len(self._index2word))}
    
    def get_keras_layer(self, trainable=False, **kwargs):
        """Creates a Keras embedding layer with the loaded vectors
        as weights

        Args:
            trainable (bool): Whether to freeze the layer weights
            **kwargs: Other kwargs to Keras
        
        Returns:
            An instance of keras.embeddings.Embedding
        """

        try:
            from keras.layers.embeddings import Embedding
        except:
            raise ImportError('Keras not found')
        
        return Embedding(self.vocab_size, self.dim, weights=[self._matrix], trainable=trainable, **kwargs)
    
    def get_pytorch_layer(self, trainable=False):
        """Creates a Pytorch embedding layer with the loaded vectors
        as weights

        Args:
            trainable (bool): Whether to freeze the layer weights
        
        Returns:
            An instance of torch.nn.Embedding
        """

        try:
            import torch
            from torch.nn import Embedding
        except:
            raise ImportError('PyTorch not found')

        return Embedding.from_pretrained(torch.FloatTensor(self._matrix), freeze=trainable)
    
    def create_subset(self, index2word):
        """Create another embedding containing the vectors of a subset of the original vocabulary

        Args:
            index2word (list): A list of words in the subset

        Returns:
            An Embedding object containing for the subset of words
        """
        indices = [self._index_dict[word] for word in index2word]
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
    
    def __contains__(self, item):
        return item in self._index_dict

    def __getitem__(self, idx):
        return (self.index2word[idx], self.matrix[idx])