

class Tokenizer(object):

    def __init__(self, token2idx={}, frozen=False, oov_idx=None):
        """
        Converts tokens to numerical indices
        Accepts:
            token2idx (dict): A mapping from words/tokens to corresponding indices
            frozen (bool): If set to True, new words will be converted to oov_idx
                           instead of being added to the vocabulary
            oov_idx (int): If frozen==True, unknown words are converted to this index
        Raises:
            AssertionError: When frozen is set to True and oov_idx is None
        """
        self.token2idx = token2idx
        self.frozen = frozen
        if frozen:
            assert oov_idx is not None, "Assign a word index for out-of-vocabulary words"
            self.oov_idx = oov_idx
        self.idx = len(tok2idx)

    def tokenize(self, token):
        """Converts a single token to a numerical index.

        Args:
            token (str): A single token to be converted into a numerical index
        """
        if token not in self.tok2idx:
            if self.frozen:
                return self.oov_idx
            else:
                self.tok2idx[token] = self.idx
                self.idx += 1
        return self.tok2idx[token]

    def split(self, sequence):
        """Method to split the sequence
        Re implement this method for other tokenizers
        """
        return sequence.split()

    def tokenize_sequence(self, sequence):
        """
        Splits and converts a sequence to a list
        numerical indices
        Accepts:
            sentence: Sentence to be converted
            split: Delimiter for splitting
        Returns:
            A list of numerical indices
        """
        return [self.tokenize(word) for word in self.split(sequence)]
    
    def tokenize_list_of_sequences(self, sequence_list):
        """
        Splits and converts a list of sequences to a list
        numerical indices
        Accepts:
            sentence: Sentence to be converted
            split: Delimiter for splitting
        Returns:
            A list of list of numerical indices
        """

        return [self.tokenize_sequence(sequence) for sequence in sequence_list]
