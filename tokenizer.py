import re


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
            char_level: Tokenize at char_level
        Returns:
            A list of numerical indices
        """
        if char_level:
            return [[self.tokenize(char) for char in list(word)] for word in self.split(sentence)]
        else:
            return [self.tokenize(word) for word in self.split(sentence)]

    def tokenize_list_of_sentences(self, sentence_list, char_level=False):
        """
        Splits and converts a list of sequences to a list
        numerical indices
        Accepts:
            sentence: Sentence to be converted
            split: Delimiter for splitting
        Returns:
            A list of list of numerical indices
        """

        return [self.tokenize_sentence(sentence, char_level=char_level) for sentence in sentence_list]


class TreebankWordTokenizer(Tokenizer):
    """
    The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank.
    This is the method that is invoked by ``word_tokenize()``.  It assumes that the
    text has already been segmented into sentences, e.g. using ``sent_tokenize()``.
    This tokenizer performs the following steps:
    - split standard contractions, e.g. ``don't`` -> ``do n't`` and ``they'll`` -> ``they 'll``
    - treat most punctuation characters as separate tokens
    - split off commas and single quotes, when followed by whitespace
    - separate periods that appear at the end of line
        >>> from nltk.tokenize import TreebankWordTokenizer
        >>> s = '''Good muffins cost $3.88\\nin New York.  Please buy me\\ntwo of them.\\nThanks.'''
        >>> TreebankWordTokenizer().tokenize(s)
        ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks', '.']
        >>> s = "They'll save and invest more."
        >>> TreebankWordTokenizer().tokenize(s)
        ['They', "'ll", 'save', 'and', 'invest', 'more', '.']
        >>> s = "hi, my name can't hello,"
        >>> TreebankWordTokenizer().tokenize(s)
        ['hi', ',', 'my', 'name', 'ca', "n't", 'hello', ',']
    """

    # starting quotes
    STARTING_QUOTES = [
        (re.compile(r'^\"'), r'``'),
        (re.compile(r'(``)'), r' \1 '),
        (re.compile(r"([ \(\[{<])(\"|\'{2})"), r'\1 `` '),
    ]

    # punctuation
    PUNCTUATION = [
        (re.compile(r'([:,])([^\d])'), r' \1 \2'),
        (re.compile(r'([:,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'[;@#$%&]'), r' \g<0> '),
        # Handles the final period.
        (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
        (re.compile(r'[?!]'), r' \g<0> '),

        (re.compile(r"([^'])' "), r"\1 ' "),
    ]

    # Pads parentheses
    PARENS_BRACKETS = (re.compile(r'[\]\[\(\)\{\}\<\>]'), r' \g<0> ')

    # Optionally: Convert parentheses, brackets and converts them to PTB symbols.
    CONVERT_PARENTHESES = [
        (re.compile(r'\('), '-LRB-'), (re.compile(r'\)'), '-RRB-'),
        (re.compile(r'\['), '-LSB-'), (re.compile(r'\]'), '-RSB-'),
        (re.compile(r'\{'), '-LCB-'), (re.compile(r'\}'), '-RCB-')
    ]

    DOUBLE_DASHES = (re.compile(r'--'), r' -- ')

    # ending quotes
    ENDING_QUOTES = [
        (re.compile(r'"'), " '' "),
        (re.compile(r'(\S)(\'\')'), r'\1 \2 '),
        (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
    ]

    # List of contractions adapted from Robert MacIntyre's tokenizer.
    CONTRACTIONS2 = [r"(?i)\b(can)(?#X)(not)\b",
                     r"(?i)\b(d)(?#X)('ye)\b",
                     r"(?i)\b(gim)(?#X)(me)\b",
                     r"(?i)\b(gon)(?#X)(na)\b",
                     r"(?i)\b(got)(?#X)(ta)\b",
                     r"(?i)\b(lem)(?#X)(me)\b",
                     r"(?i)\b(mor)(?#X)('n)\b",
                     r"(?i)\b(wan)(?#X)(na)\s"]
    CONTRACTIONS3 = [r"(?i) ('t)(?#X)(is)\b", r"(?i) ('t)(?#X)(was)\b"]
    CONTRACTIONS2 = list(map(re.compile, CONTRACTIONS2))
    CONTRACTIONS3 = list(map(re.compile, CONTRACTIONS3))

    def split(self, text, convert_parentheses=False):
        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Handles parentheses.
        regexp, substitution = self.PARENS_BRACKETS
        text = regexp.sub(substitution, text)
        # Optionally convert parentheses
        if convert_parentheses:
            for regexp, substitution in self.CONVERT_PARENTHESES:
                text = regexp.sub(substitution, text)

        # Handles double dash.
        regexp, substitution = self.DOUBLE_DASHES
        text = regexp.sub(substitution, text)

        # add extra space to make things easier
        text = " " + text + " "

        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r' \1 \2 ', text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r' \1 \2 ', text)

        # We are not using CONTRACTIONS4 since
        # they are also commented out in the SED scripts
        # for regexp in self._contractions.CONTRACTIONS4:
        #     text = regexp.sub(r' \1 \2 \3 ', text)

        return text.split()
