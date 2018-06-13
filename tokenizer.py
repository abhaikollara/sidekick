

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
        self.idx = len(token2idx)

    def tokenize(self, token):
        """Converts a single token to a numerical index.

        Args:
            token (str): A single token to be converted into a numerical index
        """
        if token not in self.token2idx:
            if self.frozen:
                return self.oov_idx
            else:
                self.token2idx[token] = self.idx
                self.idx += 1
        return self.token2idx[token]

    def split(self, sentence):
        """Method to split the sequence
        Re implement this method for other tokenizers
        """
        return sentence.split()

    def tokenize_sentence(self, sentence, char_level=False):
        """
        Splits and converts a sequence to a list
        numerical indices
        Accepts:
            sentence: Sentence to be converted
            split: Delimiter for splitting
        Returns:
            A list of numerical indices
        """
        if char_level:
            return [[self.tokenize(char) for char in list(word)] for word in self.split(sentence)]
        else:
            return [self.tokenize(word) for word in self.split(sentence)]
    
    def tokenize_list_of_sentences(self, sentence_list):
        """
        Splits and converts a list of sequences to a list
        numerical indices
        Accepts:
            sentence: Sentence to be converted
            split: Delimiter for splitting
        Returns:
            A list of list of numerical indices
        """

        return [self.tokenize_sentence(sentence) for sentence in sentence_list]

