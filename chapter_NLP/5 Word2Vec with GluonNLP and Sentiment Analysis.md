# Word2Vec with GluonNLP and Sentiment Analysis

In this notebook, we'll demonstrate how to use pre-trained word embeddings for Word2Vec using GluonNLP.

To begin, let's first import a few packages that we'll need for this example:

```{.python .input  n=1}
import warnings
warnings.filterwarnings('ignore')

import mxnet as mx
import random
import time
import multiprocessing as mp
import numpy as np
from mxnet import gluon
from mxnet import nd, autograd
import gluonnlp as nlp
import re

random.seed(123)
np.random.seed(123)
mx.random.seed(123)
```

Now that we've imported the packages, we'll demonstrate how to index words, attach pre-trained word embeddings for them, and use such embeddings in Gluon.

Recall in the count-based vector notebook, we created a `vocab` object from the d2l package. Here we'll do the same but with GluonNLP's advanced features we can do it much easier and have a `vocab` object that allows for more manipulation and extensibility. Let's assign a unique ID and word vector to each word in the vocabulary in just a few lines of code.

### Creating Vocabulary from Data Sets

To begin, suppose that we have a simple text data set consisting of newline-separated strings.

```{.python .input  n=2}
text = """
       hello world \n
       hello nice world \n
       hi world \n
       """
```

To start, let's implement a simple tokenizer to separate the words and then count the frequency of each word in the data set. We can use our defined tokenizer to count word frequency in the data set. Later, we'll use a more advanced version of this simple tokenizer provided by Spacy.

```{.python .input  n=3}
def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return filter(None, re.split(token_delim + '|' + seq_delim, source_str))
counter = nlp.data.count_tokens(simple_tokenize(text))
```

The obtained `counter` behaves like a Python dictionary whose key-value pairs consist of words and their frequencies, respectively.
We can then instantiate a `Vocab` object with the `counter` we just defined.

Because the `counter` object tracks word frequencies, we are able to specify arguments like `max_size` (maximum size) and `min_freq` (minimum frequency) to the `Vocab` constructor to restrict the size of the resulting vocabulary. Maximum size will restrict the max size of the vocabulary, and minimim frequency specifies the minimum frequency a word must occur before it's added to the `Vocab`.

Suppose that we want to build indices for all the keys in counter.
If we simply want to construct a  `Vocab` containing every word, then we can supply `counter` as the only argument. As simple as this:

```{.python .input  n=4}
vocab = nlp.Vocab(counter)
```

A `Vocab` object associates each word with an index. We can easily access words by their indices using the `vocab.idx_to_token` attribute.

```{.python .input  n=5}
for word in vocab.idx_to_token:
    print(word)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "<unk>\n<pad>\n<bos>\n<eos>\nworld\nhello\nhi\nnice\n"
 }
]
```

Contrarily, we can also grab an index given a token using `vocab.token_to_idx`.

```{.python .input  n=6}
print(vocab.token_to_idx["<unk>"])
print(vocab.token_to_idx["world"])
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0\n4\n"
 }
]
```

In Gluon NLP, for each word, there are three representations: the index of where it occurred in the original input (idx), the embedding (or vector/vec), and the token (the actual word). At any point, we may use any of the following methods to switch between the three representations: `idx_to_vec`, `idx_to_token`, `token_to_idx`.

Now that we have a vocab object and the word embeddings, we need to attach them to one another so they can reference each other during our next language modeling step.

Our next step will be to attach word embeddings to the words indexed by `vocab`.
We are using pre-trained Word2Vec embeddings (that we learned to train and construct from scratch in the previous notebook) pre-trained on the *GoogleNews-vectors-negative300* dataset.

First, we'll want to create a word embedding instance by calling `nlp.embedding.create`,
specifying the embedding type `word2vec` (an unnamed argument) and the source `source='GoogleNews-vectors-negative300'` (the named argument). This will take some time if you previously did not have the freebase-vectors dataset previously downloaded.

```{.python .input  n=7}
word2vec_simple = nlp.embedding.create('word2vec', source='GoogleNews-vectors-negative300')
```

To attach the newly loaded word embeddings `word2vec_simple` to indexed words in `vocab`, we can simply call vocab's `set_embedding` method:

```{.python .input  n=8}
vocab.set_embedding(word2vec_simple)
```

To see other available sources of pretrained word embeddings using the word2vec algorithm,
we can call `text.embedding.list_sources`.

```{.python .input  n=9}
nlp.embedding.list_sources('word2vec')[:5]
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "['GoogleNews-vectors-negative300',\n 'freebase-vectors-skipgram1000-en',\n 'freebase-vectors-skipgram1000']"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

The created vocabulary `vocab` includes four different words and a special
unknown token. Let us check the size of `vocab`.

```{.python .input  n=10}
len(vocab)
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "8"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

By default, the vector of any token that is unknown to `vocab` is a zero vector.
Its length is equal to the vector dimensions of the word2vec word embeddings:
(1000,). Beautiful was never used in the vocabulary, so the embedding will have the first five elements equal to zero.

```{.python .input  n=11}
vocab.embedding['beautiful'].shape
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "(300,)"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

And here are the first five elements.

```{.python .input  n=12}
vocab.embedding['beautiful'][:5]
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "\n[0. 0. 0. 0. 0.]\n<NDArray 5 @cpu(0)>"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Let us check the shape of the embedding of the words 'hello' and 'world' from `vocab`.

```{.python .input  n=13}
vocab.embedding['hello', 'world'].shape
```

```{.json .output n=13}
[
 {
  "data": {
   "text/plain": "(2, 300)"
  },
  "execution_count": 13,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can access the first five elements of the embedding of 'hello' and 'world' and see that they are non-zero as they are in our vocabulary.

```{.python .input  n=14}
vocab.embedding['hello', 'world'][:5]
```

```{.json .output n=14}
[
 {
  "data": {
   "text/plain": "\n[[-1.64060e-02  5.17300e-03 -1.59800e-03  1.00507e-01 -7.56760e-02\n  -4.23100e-03 -4.55240e-02 -8.04060e-02  4.98800e-03  1.15879e-01\n  -9.97700e-03 -2.94130e-02 -4.93670e-02 -1.34500e-02  2.86400e-03\n   5.58700e-02  1.10110e-02  5.02540e-02  1.09967e-01 -7.74500e-02\n   1.13514e-01  5.20270e-02  6.47390e-02 -6.03040e-02  3.96120e-02\n  -2.20230e-02 -8.53600e-03  3.51780e-02  4.64110e-02  2.74920e-02\n   2.05450e-02 -9.09000e-03 -5.11410e-02 -6.29650e-02 -1.12330e-02\n  -6.88770e-02  7.98150e-02  3.67700e-03  5.55750e-02  9.40040e-02\n  -3.26650e-02 -5.79400e-02  6.53300e-02  3.99070e-02 -1.06420e-02\n   5.61660e-02 -9.34130e-02  1.44850e-02 -3.32560e-02  4.34550e-02\n  -1.31842e-01 -1.14550e-02  3.28130e-02  4.25680e-02 -3.20740e-02\n   7.92230e-02 -5.17320e-02  1.19426e-01  3.81340e-02 -8.39530e-02\n  -8.51360e-02  4.46370e-02 -6.35560e-02  7.13200e-03  5.58700e-02\n   1.34900e-03 -8.45440e-02 -1.10110e-02 -8.86830e-02  5.94180e-02\n   6.26690e-02  8.75010e-02 -6.20780e-02  2.05450e-02 -1.30660e-01\n  -3.32560e-02 -7.80410e-02 -7.05800e-03  3.42910e-02  7.00600e-02\n  -1.33760e-02  3.28130e-02 -8.75010e-02 -2.88220e-02 -3.13350e-02\n  -9.82900e-03  2.35010e-02 -4.04990e-02  6.94680e-02  1.90670e-02\n   2.52750e-02  8.86800e-03 -3.47340e-02  1.61700e-03 -3.93160e-02\n   7.61200e-03  2.66050e-02  7.41980e-02 -3.47340e-02 -8.98650e-02\n  -1.79731e-01 -8.92740e-02 -4.04990e-02  8.39530e-02 -1.26370e-02\n   3.51780e-02  8.57270e-02  7.30000e-04  4.19770e-02 -2.06900e-03\n  -9.10480e-02  4.90710e-02  3.54700e-03 -4.19770e-02  1.47805e-01\n   8.72000e-03  7.31600e-03  1.43370e-02  1.77370e-02 -7.21290e-02\n   8.35100e-03  1.81060e-02 -1.16770e-02  2.09880e-02  4.52280e-02\n  -3.29610e-02 -2.21710e-02  2.66050e-02  8.21800e-02  1.99540e-02\n  -1.14697e-01 -7.92230e-02 -3.99070e-02  2.89700e-02 -9.45950e-02\n   3.08910e-02  9.27500e-03  7.15380e-02  1.76400e-03  8.39530e-02\n   6.20780e-02 -5.40970e-02 -9.51860e-02 -4.80400e-03  4.22720e-02\n   4.10900e-02  1.18240e-02 -8.86830e-02  7.09460e-02 -1.02872e-01\n  -3.57690e-02  3.22210e-02 -5.58700e-02 -6.35600e-03  7.76000e-03\n   7.68590e-02  2.20230e-02  4.10900e-02 -4.18000e-04 -7.80410e-02\n  -8.75010e-02  3.05960e-02  5.82350e-02 -1.47810e-02  8.45440e-02\n  -1.01690e-01 -2.23190e-02  5.69000e-03 -3.32560e-02 -1.39680e-02\n   4.75930e-02  2.02490e-02 -1.03460e-02  4.93670e-02  2.61610e-02\n   1.35389e-01  6.13400e-03 -5.98600e-03  2.40920e-02  5.38010e-02\n  -1.32290e-02 -2.88200e-03  4.99580e-02  5.23230e-02  7.00600e-02\n  -1.29330e-02  7.13200e-03  5.55750e-02 -1.25930e-01 -5.28400e-03\n   5.08450e-02  1.43370e-02  4.31590e-02  2.57180e-02  1.02872e-01\n   4.49330e-02 -1.04055e-01 -4.28630e-02 -2.06930e-02 -4.43420e-02\n  -8.61000e-03  2.24660e-02 -2.32050e-02  3.84290e-02  1.77370e-02\n  -2.29100e-02 -1.01250e-02  7.15380e-02 -4.93670e-02  4.99580e-02\n   4.49330e-02 -7.33110e-02 -1.06420e-01 -9.28220e-02  1.48700e-03\n   5.35050e-02  1.40119e-01  4.31590e-02 -7.56760e-02 -7.86320e-02\n   1.32290e-02  1.05828e-01  1.80320e-02  2.30580e-02 -8.68400e-03\n  -2.92650e-02 -3.88000e-03  1.77370e-02 -6.94680e-02 -5.91220e-02\n  -3.69510e-02  6.08960e-02 -1.28295e-01  1.81800e-02  1.52535e-01\n   6.32610e-02  3.39950e-02 -1.83280e-02  1.00507e-01  2.24660e-02\n  -1.75150e-02  3.35520e-02 -1.96580e-02  1.70710e-02  5.32100e-03\n   2.54220e-02  4.16810e-02  7.80410e-02  5.08450e-02 -5.11410e-02\n   5.43200e-03  4.84800e-02  7.92230e-02  9.57780e-02 -7.50850e-02\n   1.62590e-02 -2.60140e-02  5.17320e-02 -1.19426e-01 -4.73000e-04\n  -2.21710e-02 -1.39680e-02 -4.90710e-02 -4.75930e-02  6.44430e-02\n  -4.78890e-02  1.33020e-02  8.63180e-02  3.62100e-03 -8.15880e-02\n  -1.32290e-02  1.12332e-01  1.41150e-02 -5.97130e-02  9.22300e-02\n  -1.10558e-01 -7.15380e-02  2.43880e-02 -1.28590e-02 -4.43420e-02\n  -1.88450e-02 -1.61850e-02 -1.71450e-02  5.73480e-02  1.12332e-01\n  -6.68080e-02  1.40410e-02  7.90800e-03 -3.47340e-02  8.04060e-02\n  -7.42700e-03  3.35520e-02 -7.61200e-03 -3.65080e-02  1.60370e-02\n   2.15800e-02  1.93000e-04 -1.10558e-01 -4.16810e-02 -3.90210e-02]\n [-2.80450e-02  2.99710e-02  9.84780e-02  5.78020e-02 -2.61180e-02\n   1.70200e-02  3.31830e-02 -6.20840e-02  3.10420e-02  6.63650e-02\n  -3.87490e-02 -3.01860e-02  8.99140e-02 -9.29120e-02 -4.83830e-02\n   4.19600e-02 -3.57520e-02  9.37680e-02  7.10750e-02  1.72340e-02\n  -3.42530e-02  3.36110e-02  3.98190e-02 -3.93910e-02  4.00330e-02\n   9.42000e-03 -1.14748e-01  2.65460e-02 -2.45120e-02 -5.95150e-02\n  -5.86000e-03 -2.30140e-02 -8.13510e-02  3.33970e-02  4.51710e-02\n  -1.33800e-02 -7.02190e-02 -3.55380e-02  5.99430e-02  6.29400e-02\n  -1.12930e-02  7.19320e-02 -1.98030e-02  1.26737e-01  9.37680e-02\n   1.44719e-01 -2.38200e-03 -5.29000e-04 -3.72500e-02  8.17790e-02\n  -7.66410e-02  3.38250e-02  4.22800e-03 -3.36110e-02  7.49290e-02\n   1.07897e-01  2.56900e-02 -1.08754e-01 -1.98030e-02  2.31210e-02\n   5.11660e-02  1.92670e-02  5.07370e-02 -2.42980e-02 -4.30300e-02\n  -1.92670e-02 -2.44050e-02  1.14000e-02  1.26400e-03 -2.41910e-02\n  -9.97620e-02  4.73120e-02  6.89340e-02  6.80780e-02  3.12560e-02\n  -3.87490e-02  1.04360e-02  4.11040e-02  9.31300e-03 -3.72500e-02\n   2.66530e-02 -4.47430e-02  1.42360e-02 -5.52330e-02  1.54140e-02\n  -4.15320e-02 -2.23720e-02 -7.92100e-02 -4.88000e-04  1.04900e-01\n   8.60610e-02 -2.46190e-02 -5.56610e-02 -6.37960e-02 -1.14748e-01\n  -7.40720e-02 -4.41010e-02 -4.92390e-02  1.49001e-01 -5.90870e-02\n  -4.04610e-02  1.25024e-01 -3.23260e-02  2.19430e-02 -7.62130e-02\n   2.57970e-02  1.79830e-02  2.80450e-02 -1.85180e-02 -2.66530e-02\n   2.65460e-02  9.95500e-03 -2.65460e-02  5.07370e-02  5.48050e-02\n  -3.89630e-02 -9.63370e-02 -1.57564e-01  3.89630e-02  3.58600e-03\n  -2.74020e-02 -6.20840e-02 -3.40390e-02  5.20220e-02  1.07040e-02\n  -8.26360e-02  6.20840e-02 -4.28160e-02 -1.17317e-01 -3.40390e-02\n  -3.15800e-03  4.53850e-02 -3.74640e-02  4.96670e-02  2.78310e-02\n  -8.52050e-02  9.29120e-02  2.49410e-02 -3.55380e-02 -2.82590e-02\n   1.23311e-01 -5.69460e-02  2.39770e-02  2.84730e-02  8.30640e-02\n   4.17460e-02  3.55380e-02  3.01860e-02 -2.61180e-02 -4.38870e-02\n   5.00950e-02  1.58420e-02  1.25770e-02 -4.41010e-02 -2.80450e-02\n  -3.61800e-02 -5.24500e-02  5.13800e-03  5.52330e-02 -6.46530e-02\n  -6.37960e-02  3.31830e-02  3.18980e-02 -1.21599e-01  2.69740e-02\n  -4.62420e-02  3.25400e-02 -8.77740e-02  3.08280e-02  3.27550e-02\n  -4.73120e-02  6.37960e-02  6.89340e-02 -8.22070e-02  4.66700e-02\n  -1.02331e-01  4.75260e-02 -5.20220e-02 -7.36440e-02 -8.13510e-02\n  -3.38250e-02 -1.20742e-01  3.27550e-02 -1.03616e-01  4.49570e-02\n  -4.64560e-02 -2.04450e-02 -2.76170e-02  4.85970e-02  1.14748e-01\n   8.99140e-02 -6.12000e-04  5.53900e-03 -8.56330e-02 -4.00330e-02\n   2.38700e-02 -5.00950e-02 -2.91150e-02 -6.80780e-02 -2.32921e-01\n  -4.54900e-03  9.63400e-03  3.74600e-03 -3.76780e-02 -4.64560e-02\n   4.00330e-02 -8.69170e-02 -1.64840e-02 -1.19030e-01  2.25860e-02\n  -4.30300e-02  2.86870e-02  8.83100e-03  4.94530e-02 -3.10420e-02\n   1.94810e-02 -8.86300e-02  4.94530e-02  7.76000e-03  2.49410e-02\n  -3.98190e-02 -3.68220e-02  1.25800e-03 -4.64560e-02 -5.13800e-02\n   1.83040e-02 -1.42360e-02 -1.03188e-01  2.49410e-02 -7.32160e-02\n   2.86870e-02 -3.12560e-02  6.93630e-02 -6.80780e-02  3.87490e-02\n   5.65180e-02 -5.22360e-02  3.27550e-02 -5.80700e-03 -4.53850e-02\n   7.40720e-02 -5.00950e-02 -1.75550e-02 -5.37350e-02  5.41630e-02\n   1.71270e-02  6.60000e-05 -1.10466e-01 -5.30920e-02 -7.06470e-02\n   7.15030e-02 -6.03710e-02  4.55990e-02  1.03290e-02 -1.10250e-02\n  -9.97620e-02  1.38080e-02  6.12800e-03 -1.03000e-03 -6.76500e-02\n   7.06500e-03 -7.19320e-02 -3.24000e-04  3.78930e-02  4.58140e-02\n   2.42980e-02  5.11660e-02  2.47260e-02 -5.22360e-02  4.47430e-02\n  -9.37680e-02  6.03710e-02  3.29690e-02 -6.50810e-02 -8.34900e-03\n   1.92700e-03 -1.45576e-01 -1.76600e-03  2.62250e-02 -1.68050e-02\n  -7.81400e-03 -1.12179e-01  2.22100e-03 -2.76170e-02  3.63940e-02\n   4.55990e-02 -5.35200e-02 -2.99710e-02  6.12270e-02  7.15030e-02\n  -5.29900e-03 -6.61000e-03 -1.23311e-01  1.55210e-02 -3.98190e-02\n  -6.20840e-02 -3.14700e-02  2.53690e-02  6.97910e-02 -2.71880e-02]]\n<NDArray 2x300 @cpu(0)>"
  },
  "execution_count": 14,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Creating Vocabulary from Pre-trained Word Embeddings

We can also create
vocabulary by using vocabulary of pre-trained word embeddings.
Below are a few pre-trained file names for Word2Vec specifically.

```{.python .input  n=17}
nlp.embedding.list_sources('word2vec')[:5]
```

```{.json .output n=17}
[
 {
  "data": {
   "text/plain": "['GoogleNews-vectors-negative300',\n 'freebase-vectors-skipgram1000-en',\n 'freebase-vectors-skipgram1000']"
  },
  "execution_count": 17,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

For simplicity of demonstration, we use a smaller word embedding file, such as
the 300-dimensional one.

```{.python .input  n=18}
word2vec = nlp.embedding.create('word2vec', source='GoogleNews-vectors-negative300')
```

Now we create vocabulary by using all the tokens from `word2vec`.

```{.python .input  n=19}
vocab = nlp.Vocab(nlp.data.Counter(word2vec.idx_to_token))
vocab.set_embedding(word2vec)
```

Below shows the size of `vocab` including a special unknown token.

```{.python .input  n=20}
len(vocab.idx_to_token)
```

```{.json .output n=20}
[
 {
  "data": {
   "text/plain": "3000004"
  },
  "execution_count": 20,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can access attributes of `vocab`.

```{.python .input  n=21}
print(vocab['beautiful'])
print(vocab.idx_to_token[2410306])
```

```{.json .output n=21}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "2410306\nbeautiful\n"
 }
]
```

We will continue to use these word2vec embeddings and its large vocab for the task of sentiment analysis, to get a better result than using the word2vec from our small-scale build-from-scratch and its relatively limited vocabulary. 

## Using the embeddings for sentiment analysis with a pre-trained language model

Let's use the embeddings we created above with a pre-trained language model for sentiment analysis. We'll look at the classic problem of sentiment analysis: taking an input consisting of a string of text and classifying its sentiment as positive or negative.

The weights of this model are initialized based on a pre-trained language model. Using pre-trained language model weights is one of the most common approaches for semi-supervised learning in NLP. In order to do a good job with large language modeling on a large corpus of text, our model must learn representations that contain information about the structure of natural language. Intuitively, by starting with these good features, versus simply random features, we're able to converge faster towards a superior model for our downstream task, and in this case, get to our results a lot faster! Either that or you can spend lots of cpu/gpu time retraining the model from scratch; but for the sake of brevity, we'll use the pre-trained weights.

With GluonNLP, we can quickly prototype the model and it's easy to customize. The building process consists of just three simple steps. For this demonstration we'll focus on movie reviews from the Large Movie Review Dataset, also known as the IMDB dataset. Given a movie review, our model will output a prediction of its sentiment, either positive or negative.


## Setup

Firstly, we must load the required modules. Please remember to have already installed GluonNLP and Spacy with its English language model. We set the random seed so the outcome can be relatively consistent.

```{.python .input  n=22}
import warnings
warnings.filterwarnings('ignore')

import random
import time
import multiprocessing as mp
import numpy as np

import mxnet as mx
from mxnet import nd, gluon, autograd

import gluonnlp as nlp

random.seed(123)
np.random.seed(123)
mx.random.seed(123)
```

## Sentiment analysis model with pre-trained language model encoder

So that we can easily transplant the pre-trained weights, we'll base our model architecture on the pre-trained language model (LM). Following the LSTM layer, we have one representation vector for each word in the sentence. Because we plan to make a single prediction (as opposed to one per word), we'll first pool our predictions across time steps before feeding them through a dense last layer to produce our final prediction (a single sigmoid output node).

![sa-model](samodel-v3.png)

Specifically, our model represents input words by their embeddings. Following the embedding layer, our model consists of a two-layer LSTM, followed by an average pooling layer, followed by a sigmoid output layer (all illustrated in the figure above).

Thus, given an input sequence, the memory cells in the LSTM layer will produce a representation sequence. This representation sequence is then averaged over all time steps resulting in a fixed-length sentence representation $h$. Finally, we apply a sigmoid output layer on top of $h$. We’re using the sigmoid activation function because we’re trying to predict if this text has positive or negative sentiment. A sigmoid activation function squashes the output values to the range [0,1], allowing us to interpret this output as a probability, making our lives relatively simpler.

Below we define our `MeanPoolingLayer` and basic sentiment analysis network's (`SentimentNet`) structure.

```{.python .input  n=23}
class MeanPoolingLayer(gluon.HybridBlock):
    """A block for mean pooling of encoder features"""
    def __init__(self, prefix=None, params=None):
        super(MeanPoolingLayer, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, data, valid_length): # pylint: disable=arguments-differ
        """Forward logic"""
        # Data will have shape (T, N, C)
        masked_encoded = F.SequenceMask(data,
                                        sequence_length=valid_length,
                                        use_sequence_length=True)
        agg_state = F.broadcast_div(F.sum(masked_encoded, axis=0),
                                    F.expand_dims(valid_length, axis=1))
        return agg_state


class SentimentNet(gluon.HybridBlock):
    """Network for sentiment analysis."""
    def __init__(self, dropout, prefix=None, params=None):
        super(SentimentNet, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = None # will set with lm embedding later
            self.encoder = None # will set with lm encoder later
            self.agg_layer = MeanPoolingLayer()
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dropout(dropout))
                self.output.add(gluon.nn.Dense(1, flatten=False))

    def hybrid_forward(self, F, data, valid_length): # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(data))  # Shape(T, N, C)
        agg_state = self.agg_layer(encoded, valid_length)
        out = self.output(agg_state)
        return out
```

## Defining the hyperparameters and initializing the model

### Hyperparameters

Our model is based on a standard LSTM model. We use a hidden layer size of 200. We use bucketing for speeding up the processing of variable-length sequences. We don't configure dropout for this model as it could be deleterious to the results.

```{.python .input  n=24}
dropout = 0
language_model_name = 'standard_lstm_lm_200'
pretrained = True
learning_rate, batch_size = 0.005, 32
bucket_num, bucket_ratio = 10, 0.2
epochs = 1
grad_clip = None
log_interval = 100
```

If your environment supports GPUs, keep the context value the same. If it doesn't, swap the `mx.gpu(0)` to `mx.cpu()`.

```{.python .input  n=27}
context = mx.cpu(0)
```

### Loading the pre-trained model

The loading of the pre-trained model, like in previous tutorials, is as simple as one line.

```{.python .input  n=28}
lm_model, _ = nlp.model.get_model(name=language_model_name,
                                      dataset_name='wikitext-2',
                                      pretrained=pretrained,
                                      ctx=context,
                                      dropout=dropout)
```

### Creating the sentiment analysis model from the loaded pre-trained model

In the code below, we already have acquireq a pre-trained model on the Wikitext-2 dataset using `nlp.model.get_model`. We then construct a SentimentNet object, which takes as input the embedding layer and encoder of the pre-trained model.

As we employ the pre-trained embedding layer and encoder, *we only need to initialize the output layer* using `net.out_layer.initialize(mx.init.Xavier(), ctx=context)`.

```{.python .input  n=29}
net = SentimentNet(dropout=dropout)
net.embedding = lm_model.embedding
net.encoder = lm_model.encoder
net.hybridize()
net.output.initialize(mx.init.Xavier(), ctx=context)
print(net)
```

```{.json .output n=29}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "SentimentNet(\n  (embedding): HybridSequential(\n    (0): Embedding(33278 -> 200, float32)\n  )\n  (encoder): LSTM(200 -> 200, TNC, num_layers=2)\n  (agg_layer): MeanPoolingLayer(\n  \n  )\n  (output): HybridSequential(\n    (0): Dropout(p = 0, axes=())\n    (1): Dense(None -> 1, linear)\n  )\n)\n"
 }
]
```

## The data pipeline

In this section, we describe in detail the data pipeline, from initialization to modifying it for use in our model.

### Loading the sentiment analysis dataset (IMDB reviews)

In the labeled train/test sets, out of a max score of 10, a negative review has a score of no more than 4, and a positive review has a score of no less than 7. Thus reviews with more neutral ratings are not included in the train/test sets. We labeled a negative review whose score <= 4 as 0, and a
positive review whose score >= 7 as 1. As the neural ratings are not
included in the datasets, we can use 5 as our threshold.

```{.python .input  n=33}
# The tokenizer takes as input a string and outputs a list of tokens.
tokenizer = nlp.data.SpacyTokenizer('en')

# `length_clip` takes as input a list and outputs a list with maximum length 500.
length_clip = nlp.data.ClipSequence(500)

# Helper function to preprocess a single data point
def preprocess(x):
    data, label = x
    label = int(label > 5)
    # A token index or a list of token indices is
    # returned according to the vocabulary.
    data = vocab[length_clip(tokenizer(data))]
    return data, label

# Helper function for getting the length
def get_length(x):
    return float(len(x[0]))

# Loading the dataset
train_dataset, test_dataset = [nlp.data.IMDB(root='data/imdb', segment=segment)
                               for segment in ('train', 'test')]
print('Tokenize using spaCy...')

```

```{.json .output n=33}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Downloading data/imdb/train.json from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/imdb/train.json...\nDownloading data/imdb/test.json from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/imdb/test.json...\nTokenize using spaCy...\n"
 }
]
```

Here we use the helper functions defined above to make pre-processing the dataset relatively stress-free and concise. As in a previous tutorial, `mp.Pool()` is leveraged to divide the work of preprocessing to multiple cores/machines.

```{.python .input  n=34}
def preprocess_dataset(dataset):
    start = time.time()
    with mp.Pool() as pool:
        # Each sample is processed in an asynchronous manner.
        dataset = gluon.data.SimpleDataset(pool.map(preprocess, dataset))
        lengths = gluon.data.SimpleDataset(pool.map(get_length, dataset))
    end = time.time()
    print('Done! Tokenizing Time={:.2f}s, #Sentences={}'.format(end - start, len(dataset)))
    return dataset, lengths

# Doing the actual pre-processing of the dataset
train_dataset, train_data_lengths = preprocess_dataset(train_dataset)
test_dataset, test_data_lengths = preprocess_dataset(test_dataset)
```

```{.json .output n=34}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Done! Tokenizing Time=22.89s, #Sentences=25000\nDone! Tokenizing Time=20.22s, #Sentences=25000\n"
 }
]
```

In the following code, we use `FixedBucketSampler`, which assigns each data sample to a fixed bucket based on its length. The bucket keys are either given or generated from the input sequence lengths and the number of buckets.

```{.python .input  n=35}
# Construct the DataLoader

def get_dataloader():

    # Pad data, stack label and lengths
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, ret_length=True),
        nlp.data.batchify.Stack(dtype='float32'))
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        train_data_lengths,
        batch_size=batch_size,
        num_buckets=bucket_num,
        ratio=bucket_ratio,
        shuffle=True)
    print(batch_sampler.stats())

    # Construct a DataLoader object for both the training and test data
    train_dataloader = gluon.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)
    test_dataloader = gluon.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        batchify_fn=batchify_fn)
    return train_dataloader, test_dataloader

# Use the pre-defined function to make the retrieval of the DataLoader objects simple
train_dataloader, test_dataloader = get_dataloader()
```

```{.json .output n=35}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "FixedBucketSampler:\n  sample_num=25000, batch_num=779\n  key=[59, 108, 157, 206, 255, 304, 353, 402, 451, 500]\n  cnt=[591, 1999, 5092, 5108, 3035, 2084, 1476, 1164, 871, 3580]\n  batch_size=[54, 32, 32, 32, 32, 32, 32, 32, 32, 32]\n"
 }
]
```

## Training the model

Now that all the data has been pre-processed and the model architecture has been loosely defined, we can define the helper functions for evaluation and training of the model.

### Evaluation using loss and accuracy

Here, we define a function `evaluate(net, dataloader, context)` to determine the loss and accuracy of our model in a concise way. The code is very similar to evaluation of other models in the previous tutorials. For more information and explanation of this code, please refer to the previous tutorial on [LSTM-based Language Models](https://gluon-nlp.mxnet.io/master/examples/language_model/language_model.html).

```{.python .input  n=36}
def evaluate(net, dataloader, context):
    loss = gluon.loss.SigmoidBCELoss()
    total_L = 0.0
    total_sample_num = 0
    total_correct_num = 0
    start_log_interval_time = time.time()

    print('Beginning Evaluation...')
    for i, ((data, valid_length), label) in enumerate(dataloader):
        data = mx.nd.transpose(data.as_in_context(context))
        valid_length = valid_length.as_in_context(context).astype(np.float32)
        label = label.as_in_context(context)
        output = net(data, valid_length)

        L = loss(output, label)
        pred = (output > 0.5).reshape(-1)
        total_L += L.sum().asscalar()
        total_sample_num += label.shape[0]
        total_correct_num += (pred == label).sum().asscalar()

        if (i + 1) % log_interval == 0:
            print('[Batch {}/{}] elapsed {:.2f} s'.format(
                i + 1, len(dataloader),
                time.time() - start_log_interval_time))
            start_log_interval_time = time.time()

    avg_L = total_L / float(total_sample_num)
    acc = total_correct_num / float(total_sample_num)

    return avg_L, acc
```

In the following code, we use FixedBucketSampler, which assigns each data sample to a fixed bucket based on its length. The bucket keys are either given or generated from the input sequence lengths and number of the buckets.

```{.python .input  n=37}
def train(net, context, epochs):
    trainer = gluon.Trainer(net.collect_params(), 'ftml',
                            {'learning_rate': learning_rate})
    loss = gluon.loss.SigmoidBCELoss()

    parameters = net.collect_params().values()

    # Training/Testing
    for epoch in range(epochs):
        # Epoch training stats
        start_epoch_time = time.time()
        epoch_L = 0.0
        epoch_sent_num = 0
        epoch_wc = 0
        # Log interval training stats
        start_log_interval_time = time.time()
        log_interval_wc = 0
        log_interval_sent_num = 0
        log_interval_L = 0.0

        for i, ((data, length), label) in enumerate(train_dataloader):
            L = 0
            wc = length.sum().asscalar()
            log_interval_wc += wc
            epoch_wc += wc
            log_interval_sent_num += data.shape[1]
            epoch_sent_num += data.shape[1]
            with autograd.record():
                output = net(data.as_in_context(context).T,
                             length.as_in_context(context)
                                   .astype(np.float32))
                L = L + loss(output, label.as_in_context(context)).mean()
            L.backward()
            # Clip gradient
            if grad_clip:
                gluon.utils.clip_global_norm(
                    [p.grad(context) for p in parameters],
                    grad_clip)
            # Update parameter
            trainer.step(1)
            log_interval_L += L.asscalar()
            epoch_L += L.asscalar()
            if (i + 1) % log_interval == 0:
                print(
                    '[Epoch {} Batch {}/{}] elapsed {:.2f} s, '
                    'avg loss {:.6f}, throughput {:.2f}K wps'.format(
                        epoch, i + 1, len(train_dataloader),
                        time.time() - start_log_interval_time,
                        log_interval_L / log_interval_sent_num, log_interval_wc
                        / 1000 / (time.time() - start_log_interval_time)))
                # Clear log interval training stats
                start_log_interval_time = time.time()
                log_interval_wc = 0
                log_interval_sent_num = 0
                log_interval_L = 0
        end_epoch_time = time.time()
        test_avg_L, test_acc = evaluate(net, test_dataloader, context)
        print('[Epoch {}] train avg loss {:.6f}, test acc {:.2f}, '
              'test avg loss {:.6f}, throughput {:.2f}K wps'.format(
                  epoch, epoch_L / epoch_sent_num, test_acc, test_avg_L,
                  epoch_wc / 1000 / (end_epoch_time - start_epoch_time)))
```

And finally, because of all the helper functions we've defined, training our model becomes simply one line of code!

```{.python .input}
train(net, context, epochs)
```

And testing it becomes as simple as feeding in the sample sentence like below:

```{.python .input}
net(
    mx.nd.reshape(
        mx.nd.array(vocab[['This', 'movie', 'is', 'amazing']], ctx=context),
        shape=(-1, 1)), mx.nd.array([4], ctx=context)).sigmoid()
```

Indeed, we can feed in any sentence and determine the sentiment with relative ease!
