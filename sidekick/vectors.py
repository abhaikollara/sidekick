import numpy as np
import tqdm
from tqdm import tqdm

class Vectors(object):

    def __init__(self, matrix=None, vocab=None):
        """Initialize an embedding optionally by providing the embedding matrix and vocab

        Args:
            matrix (np.ndarray, optional): The embedding matrix of the embedding
            vocab (list, optional): A list of words in the order of corresponding vectors in the matrix
        """

        self._matrix = matrix
        self.vocab = vocab

        if matrix is not None:
            if vocab is None:
                raise TypeError(
                    'vocab cannot be None if matrix is provided')
            if matrix.shape[0] != len(vocab):
                raise ValueError(
                    'Embedding matrix and vocab contain unequal number of items')

    def load_word2vec(self, path, binary=True, reserve_zero=True, reserve_oov_token=True):
        """Load word2vec model from file

        Args:
            path (str): Path to the word2vec file
            binary (bool): Whether the file is in binary format
        """
        self.allow_oov = reserve_oov_token
        self.reserve_zero = reserve_zero
        words = []
        if self.reserve_zero:
            words.append('__ZERO__')
        if self.allow_oov:
            words.append('__OUT_OF_VOCAB__')
            self.oov_index = len(words) - 1

        if binary:
            with open(path, 'rb') as f:
                # Get number of vectors and vector size
                # from first line
                num_vectors, vector_size = map(
                    int, f.readline().decode('UTF-8').split())
                FLOAT_SIZE = 4

                self._matrix = np.zeros(
                    [num_vectors + len(words), vector_size], dtype='float32')
                # Assign random vector for OOV token if it exists
                if self.allow_oov:
                    self._matrix[self.oov_index] = np.random.randn(vector_size, )

                update = words.append  # Speedup
                for i in tqdm(range(len(words), num_vectors+len(words))):
                    # Reads until a whitespace is found (get a word)
                    word = b""
                    while True:
                        char = f.read(1)
                        if char == b" ":
                            break
                        word += char
                    update(word)
                    #Read vector
                    vecs = f.read(FLOAT_SIZE * vector_size)
                    self._matrix[i] = np.frombuffer(vecs, 'f')

                self.vocab = words
        else:
            print("This feature is yet to be implemented")

    def load_glove(self, path, vocab_size=None, dim=None, reserve_zero=True, reserve_oov_token=True):
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

        words = []

        self.allow_oov = reserve_oov_token
        self.reserve_zero = reserve_zero

        if self.reserve_zero:
            words.append('__ZERO__')
        if self.allow_oov:
            words.append('__OUT_OF_VOCAB__')
            self.oov_index = len(words) - 1

        self._matrix = np.zeros((vocab_size+len(words), dim))
        # Assign random vector for OOV token if it exists
        if "__OUT_OF_VOCAB__" in words:
            self._matrix[self.oov_index] = np.random.randn(dim, )

        update = words.append  # Speedup
        with open(path, 'r') as f:
            for i in tqdm(range(len(words), self.matrix.shape[0])):
                split = f.readline().split(u' ', 1)
                update(split[0])
                self._matrix[i] = np.fromstring(split[1], 'f', sep=u' ')

        self.vocab = words

    def load_subset(self, vocab):
        """Create another embedding of a subset of the original vocabulary

        Args:
            vocab (list): A list of words in the subset

        Returns:
            An Embedding object containing for the subset of words
        """
        if self.reserve_zero:
            vocab.insert(0, '__ZERO__')
        if self.allow_oov:
            vocab.insert(self.oov_index, '__OUT_OF_VOCAB__')
            indices = []
            for word in vocab:
                try:
                    indices.append(self._index_dict[word])
                except KeyError:
                    indices.append(self.oov_index)
        else:
            indices = [self._index_dict[word] for word in vocab]
        matrix = self.matrix[indices]
        return Vectors(matrix=matrix, vocab=vocab)

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

    @property
    def index_dict(self):
        return self._index_dict

    def __contains__(self, item):
        return item in self._index_dict

    def __getitem__(self, word):
        try:
            idx = self.index_dict[word]
            return self.matrix[idx]
        except KeyError:
            raise KeyError("\"{}\" not found in vocabulary".format(word))

    def __add__(self, other):
        matrix = np.concatenate([self.matrix, other.matrix], axis=0)
        vocab = self.vocab + other.vocab

        return Vectors(matrix=matrix, vocab=vocab)
