# Sidekick
Helper functions for deep learning NLP tasks [WIP]

## Quick Start
## Word Vectors
### Loading word embeddings
```python
from sidekick.vectors import Vectors

glove = Vectors()
glove.load_glove("/Users/username/Downloads/glove.6B.50d.txt", reserve_zero=True, reserve_oov_token=True)

100%|██████████| 400000/400000 [00:07<00:00, 56646.83it/s]
```
### Get the vector of a word
```python
glove["universe"]
```

### Get the entire weight matrix
```python
glove.matrix
```

```python
glove.matrix.shape
(400002, 50) # 400000 words + 2 extra tokens for zero and out of vocab words
```

### Generate a Keras embedding layer
```python
glove.get_keras_layer(trainable=True)
```

### Create a subset of the vocab
```python
groot_speak = glove.load_subset(["I", "am", "groot"])
```
