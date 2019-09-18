# Word2Vec with GluonNLP and Sentiment Analysis

In this notebook, we'll demonstrate how to use pre-trained word embeddings for Word2Vec using GluonNLP.

To begin, let's first import a few packages that we'll need for this example:

```{.python .input  n=1}
import warnings
warnings.filterwarnings('ignore')

from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp
import re
```

Now that we've imported the packages, we'll demonstrate how to index words, attach pre-trained word embeddings for them, and use such embeddings in Gluon.

Recall in the count-based vector notebook, we created a `vocab` object. Here we'll do the same but with GluonNLP's advanced features we can do it much easier and have a `vocab` object that allows for more manipulation and extensibility. Let's assign a unique ID and word vector to each word in the vocabulary in just a few lines of code.

### Creating Vocabulary from Data Sets

To begin, suppose that we have a simple text data set consisting of newline-separated strings.

```{.python .input  n=5}
text = """
       hello world \n
       hello nice world \n
       hi world \n
       """
```

To start, let's implement a simple tokenizer to separate the words and then count the frequency of each word in the data set. We can use our defined tokenizer to count word frequency in the data set.

```{.python .input  n=6}
def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return filter(None, re.split(token_delim + '|' + seq_delim, source_str))
counter = nlp.data.count_tokens(simple_tokenize(text))
```

The obtained `counter` behaves like a Python dictionary whose key-value pairs consist of words and their frequencies, respectively.
We can then instantiate a `Vocab` object with the `counter` we just defined.

Because the `counter` object tracks word frequencies, we are able to specify arguments like `max_size` (maximum size) and `min_freq` (minimum frequency) to the `Vocab` constructor to restrict the size of the resulting vocabulary. Maximum size will restrict the max size of the vocabulary, and minimim frequency specifies the minimum frequency a word must occur before it's added to the `Vocab`.

Suppose that we want to build indices for all the keys in counter.
If we simply want to construct a  `Vocab` containing every word, then we can supply `counter` as the only argument. As simple as this:

```{.python .input  n=7}
vocab = nlp.Vocab(counter)
```

A `Vocab` object associates each word with an index. We can easily access words by their indices using the `vocab.idx_to_token` attribute.

```{.python .input  n=8}
for word in vocab.idx_to_token:
    print(word)
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "<unk>\n<pad>\n<bos>\n<eos>\nworld\nhello\nhi\nnice\n"
 }
]
```

Contrarily, we can also grab an index given a token using `vocab.token_to_idx`.

```{.python .input  n=9}
print(vocab.token_to_idx["<unk>"])
print(vocab.token_to_idx["world"])
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0\n4\n"
 }
]
```

In Gluon NLP, for each word, there are three representations: the index of where it occurred in the original input (idx), the embedding (or vector/vec), and the token (the actual word). At any point, we may use any of the following methods to switch between the three representations: `idx_to_vec`, `idx_to_token`, `token_to_idx`.

### Attaching word embeddings

Our next step will be to attach word embeddings to the words indexed by `vocab`.
In this example, we'll use pre-trained Word2Vec embeddings that we learned to train and construct from scratch in the previous notebook, pre-trained on the *freebase-vectors-skipgram1000-en* dataset.

First, we'll want to create a word embedding instance by calling `nlp.embedding.create`,
specifying the embedding type `word2vec` (an unnamed argument) and the source `source='freebase-vectors-skipgram1000-en'` (the named argument). This will take some time if you previously did not have the freebase-vectors dataset previously downloaded.

```{.python .input  n=11}
word2vec_simple = nlp.embedding.create('word2vec', source='freebase-vectors-skipgram1000-en')
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Embedding file freebase-vectors-skipgram1000-en-6086803e.npz is not found. Downloading from Gluon Repository. This may take some time.\nDownloading /Users/munshif/.mxnet/embedding/word2vec/freebase-vectors-skipgram1000-en-6086803e.npz from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/embeddings/word2vec/freebase-vectors-skipgram1000-en-6086803e.npz...\n"
 }
]
```

To attach the newly loaded word embeddings `word2vec_simple` to indexed words in `vocab`, we can simply call vocab's `set_embedding` method:

```{.python .input}
vocab.set_embedding(word2vec_simple)
```

To see other available sources of pretrained word embeddings using the word2vec algorithm,
we can call `text.embedding.list_sources`.

```{.python .input}
nlp.embedding.list_sources('word2vec')[:5]
```

The created vocabulary `vocab` includes four different words and a special
unknown token. Let us check the size of `vocab`.

```{.python .input}
len(vocab)
```

By default, the vector of any token that is unknown to `vocab` is a zero vector.
Its length is equal to the vector dimensions of the word2vec word embeddings:
(1000,).

```{.python .input}
vocab.embedding['beautiful'].shape
```

The first five elements of the vector of any unknown token are zeros.

```{.python .input}
vocab.embedding['beautiful'][:5]
```

Let us check the shape of the embedding of the words 'hello' and 'world' from `vocab`.

```{.python .input}
vocab.embedding['hello', 'world'].shape
```

We can access the first five elements of the embedding of 'hello' and 'world' and see that they are non-zero.

```{.python .input}
vocab.embedding['hello', 'world'][:, :5]
```

### Using Pre-trained Word Embeddings in Gluon

To demonstrate how to use pre-
trained word embeddings in Gluon, let us first obtain the indices of the words
'hello' and 'world'.

```{.python .input}
vocab['hello', 'world']
```

We can obtain the vectors for the words 'hello' and 'world' by specifying their
indices (5 and 4) and the weight or embedding matrix, which we get from calling `vocab.embedding.idx_to_vec` in
`gluon.nn.Embedding`. We initialize a new layer and set the weights using the layer.weight.set_data method. Subsequently, we pull out the indices 5 and 4 from the weight vector and check their first five entries.

```{.python .input}
input_dim, output_dim = vocab.embedding.idx_to_vec.shape
layer = gluon.nn.Embedding(input_dim, output_dim)
layer.initialize()
layer.weight.set_data(vocab.embedding.idx_to_vec)
layer(nd.array([5, 4]))[:, :5]
```

### Creating Vocabulary from Pre-trained Word Embeddings

We can also create
vocabulary by using vocabulary of pre-trained word embeddings, such as GloVe.
Below are a few pre-trained file names under the GloVe word embedding.

```{.python .input}
nlp.embedding.list_sources('glove')[:5]
```

For simplicity of demonstration, we use a smaller word embedding file, such as
the 50-dimensional one.

```{.python .input}
glove_6b50d = nlp.embedding.create('glove', source='glove.6B.50d')
```

Now we create vocabulary by using all the tokens from `glove_6b50d`.

```{.python .input}
vocab = nlp.Vocab(nlp.data.Counter(glove_6b50d.idx_to_token))
vocab.set_embedding(glove_6b50d)
```

Below shows the size of `vocab` including a special unknown token.

```{.python .input}
len(vocab.idx_to_token)
```

We can access attributes of `vocab`.

```{.python .input}
print(vocab['beautiful'])
print(vocab.idx_to_token[71424])
```
