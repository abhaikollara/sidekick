import numpy as np
from gensim.models import KeyedVectors


class Embedding(object):

    def __init__(self, matrix=None, vocab=None):
        """Initialize an embedding optionally by providing the embedding matrix and vocab

        Args:
            matrix (np.ndarray, optional): The embedding matrix of the embedding
            vocab (list, optional): A list of words in the order of corresponding vectors in the matrix 
        """

        self._matrix = matrix
        self._vocab = vocab

        if matrix is not None:
            if vocab is None:
                raise TypeError(
                    'vocab cannot be None if matrix is provided')
            if matrix.shape[0] != len(vocab):
                raise ValueError(
                    'Embedding matrix and vocab contain unequal number of items')

    def load_word2vec(self, path, binary=True):
        """Load word2vec model from file

        Args:
            path (str): Path to the word2vec file
            binary (bool): Whether the file is in binary format
        """

        model = KeyedVectors.load_word2vec_format(path, binary=binary)
        self._matrix = model.syn0
        self._vocab = model.vocab

    def load_glove(self, path, vocab_size=None, dim=None):
        """Load glove model from file

        Args:
            path (str): Path to the word2vec file
            binary (bool): Whether the file is in binary format
            vocab_size (int, optional): Number of words in the vocabulary, providing this reduces the loading time
            dim (int, optional): Embedding dimension
        """

        # Infer vocab size by reading number of lines
        if vocab_size is None:
            vocab_size = 0
            with open(path, 'r') as f:
                for line in f:
                    vocab_size += 1

        # Infer vector dim by reading a single line
        if dim is None:
            with open(path, 'r') as f:
                dim = len(f.readline().split(u' ')) - 1

        self._matrix = np.zeros((vocab_size, dim))
        words = []

        update = words.append  # Speedup
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                split = line.split(u' ')
                update(split[0])
                self._matrix[i] = np.asarray([float(val) for val in split[1:]])

        self.vocab = words

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

    def create_subset(self, vocab):
        """Create another embedding of a subset of the original vocabulary

        Args:
            vocab (list): A list of words in the subset

        Returns:
            An Embedding object containing for the subset of words
        """
        indices = [self._index_dict[word] for word in vocab]
        matrix = self.matrix[indices]
        return Embedding(matrix=matrix, vocab=vocab)

    @property
    def matrix(self):
        """np.ndarray: The embedding matrix of shape vocab_size x embedding dim"""
        return self._matrix

    @property
    def vocab(self):
        """list: A list of all words in the vocab"""
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        self._vocab = vocab
        if vocab is not None:
            self._index_dict = {vocab[i]: i for i in range(len(vocab))}

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
        return (self.vocab[idx], self.matrix[idx])

    def __add__(self, other):
        matrix = np.concatenate([self.matrix, other.matrix], axis=0)
        vocab = self.vocab + other.vocab

        return Embedding(matrix=matrix, vocab=vocab)
