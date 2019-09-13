# The Basics of Word Embeddings

So now that we understand the reason why NLP and representations for natural language concepts are difficult in the last chapter, let's take a look at some of the simple ways in which we can construct vector representations, on a word-level (i.e. for words). The vector representation of a word in NLP is called a **word embedding**.

Word embeddings are a very useful representation of vocabulary for machine learning purposes. The motive of word embeddings is to capture context, semantic, and syntactic similarity, and represent these aspects through a geometric relationship between the embedding vectors. Formulation of these embeddings is often harder as we need to isolate what we actually want to know about the words. Do we want to have embeddings that share a common part of speech appear closer together? Or do we want words with similar meanings appear geometrically closer than those without? How do we derive any of these relationships and how do we represent any of these features using simple means? 

All of these questions are answered in different manners according to the newest models and intuition as the field has developed, changing what word embeddings actually do for the down-stream task.


\frac{1}{6}

### Distributional vs. Distributed representations

Both of them are based on the distributional hypothesis that words occur in similar context tend to have similar meaning.

Distribution representation is the high-dimensional vector representation obtained from the rows of the word-context co-occurrence matrix, whose dimension size equals to the vocabulary size of the corpus.

Distributed representation is a low-dimensional vector representation, which can be obtained from a neural network based model (such as word2vec, Collobert and Weston embeddings, HLBL embeddings). The matrix factorization based model that perform matrix factorization on the word-context co-occurrence matrix (such as the Glove from Stanford using direct matrix factorization, the Latent Semantic Analysis using SVD factorization). The paper from Levy (Neural Word Embedding as Implicit Matrix Factorization) shows that the matrix factorization method and the neural network based method is somewhat equivalent.

Actually, as Yoav Goldberg showed in the presentation "word embeddings what, how and whither" that the distributed representation can be obtained from distributional representation based on matrix factorization.


### Count-based methods vs Prediction-based methods

With the steady growth of textual data, NLP methods are required that are able to process the data efficiently. The focus recently of word embeddings has been on efficient methods that are targeted to compute distributional models that are based on the distributional hypothesis of Harris (1951). This hypothesis claims that words occurring in similar contexts tend to have similar meanings. In order to implement this hypothesis, early approaches (Hindle, 1990; Grefenstette, 1994; Lin, 1997) represented words using count-based vectors of the context. However, such representations are very sparse, require a lot of memory and are not very efficient. In the last decades, methods have been developed that transform such sparse representations into dense representations mainly using matrix factorization. With word2vec (Mikolov et al., 2013), an efficient prediction-based method was introduced, which also represents words with a dense vector. However, also sparse and count-based methods have been proposed that allow an efficient computation, e.g. (Kilgarriff et al., 2004; Biemann and Riedl, 2013). A more detailed overview of semantic representations can be found in (Lund and Burgess, 1996; Turney and Pantel, 2010; Ferrone
and Zanzotto, 2017).

One of the first comparisons between count-based and prediction-based distributional models was performed by Baroni et al. (2014). For this, they consider various tasks and show that prediction-based word embeddings outperform sparse count-based methods and dense count-based methods used for computing distributional semantic models. The evaluation is performed on datasets for relatedness, analogy, concept categorization and selectional preferences. The majority of word pairs considered for the evaluation consists of noun pairs. However, Levy and Goldberg (2014b) showed that dense count-based methods, using PPMI weighted co-occurrences and SVD, approximates neural word embeddings. Levy et al. (2015) showed in an extensive study the impact of various parameters and show the best performing parameters for these methods. The study reports results for various datasets for word similarity and analogy. However, they do not evaluate the performance on local similarity ranking tasks and omit results for pure count-based semantic methods. Claveau and Kijak (2016) performed another comparison of various semantic representation using both intrinsic and extrinsic evaluations. They compare the performance of their count-based method to dense representations and prediction-based methods using a manually crafted lexicon, SimLex and an information retrieval task. They show that their method performs better on the manually crafted lexicon than using word2vec. For this task, they also show that a word2vec model computed on a larger dataset yields inferior results than models computed on a smaller corpus, which is contrary to previous findings, e.g. (Banko and Brill, 2001; Gorman and Curran, 2006; Riedl and Biemann, 2013). Based on the SimLex task and the extrinsic evaluation they show comparable performance to the word2vec model computed on a larger corpus.


--------------------------

```{.python .input}
import warnings
warnings.filterwarnings('ignore')

from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp
import re
```

Now we'll demonstrate how to index words,
attach pre-trained word embeddings for them,
and use such embeddings in Gluon.
First, let's assign a unique ID and word vector to each word in the vocabulary
in just a few lines of code.

To begin, suppose that we have a simple text data set consisting of newline-separated strings.

```{.python .input  n=1}
text = " hello world \n hello nice world \n hi world "
```

To start, let's implement a simple tokenizer to separate the words and then count the frequency of each word in the data set. We can use our defined tokenizer to count word frequency in the data set.

```{.python .input}
def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return filter(None, re.split(token_delim + '|' + seq_delim, source_str))

counter = nlp.data.count_tokens(simple_tokenize(text))
```

The obtained `counter` behaves like a Python dictionary whose key-value pairs consist of words and their frequencies, respectively.
We can then instantiate a `Vocab` object with a counter.

Because `counter` tracks word frequencies, we are able to specify arguments
such as `max_size` (maximum size) and `min_freq` (minimum frequency) to the `Vocab` constructor to restrict the size of the resulting vocabulary.

Suppose that we want to build indices for all the keys in counter.
If we simply want to construct a  `Vocab` containing every word, then we can supply `counter`  the only argument.

```{.python .input}
vocab = nlp.Vocab(counter)
```

A `Vocab` object associates each word with an index. We can easily access words by their indices using the `vocab.idx_to_token` attribute.

```{.python .input}
for word in vocab.idx_to_token:
    print(word)
```

Contrarily, we can also grab an index given a token using `vocab.token_to_idx`.

```{.python .input}
print(vocab.token_to_idx["<unk>"])
print(vocab.token_to_idx["world"])
```

In GluonNLP, for each word, there are three representations: the index of where it occurred in the original input (idx), the embedding (or vector/vec), and the token (the actual word). At any point, we may use any of the following methods to switch between the three representations: `idx_to_vec`, `idx_to_token`, `token_to_idx`.

Our next step will be to attach word embeddings to the words indexed by `vocab`.
In this example, we'll use *fastText* embeddings trained on the *wiki.simple* dataset.
First, we'll want to create a word embedding instance by calling `nlp.embedding.create`,
specifying the embedding type `word2vec` (an unnamed argument) and the source `source='freebase-vectors-skipgram1000'` (the named argument).

```{.python .input}
w2v = nlp.embedding.create('word2vec', source='freebase-vectors-skipgram1000')
```

To attach the newly loaded word embeddings `w2v` to indexed words in `vocab`, we can simply call vocab's `set_embedding` method:

```{.python .input}
vocab.set_embedding(w2v)
```

To see other available sources of pre-trained word embeddings using the *fastText* algorithm,
we can call `text.embedding.list_sources`.

```{.python .input}
nlp.embedding.list_sources('word2vec')
```

The created vocabulary `vocab` includes four different words and a special
unknown token. Let us check the size of `vocab`.

```{.python .input}
len(vocab)
```

By default, the vector of any token that is unknown to `vocab` is a zero vector.
Its length is equal to the vector dimensions of the word2vec word embeddings:
(300,).

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

To apply word embeddings, we need to define
cosine similarity. Cosine similarity determines the similarity between two vectors.

```{.python .input}
from mxnet import nd
def cos_sim(x, y):
    return nd.dot(x, y) / (nd.norm(x) * nd.norm(y))
```

The range of cosine similarity between two vectors can be between -1 and 1. The
larger the value, the larger the similarity between the two vectors.

```{.python .input}
x = nd.array([1, 2])
y = nd.array([10, 20])
z = nd.array([-1, -2])

print(cos_sim(x, y))
print(cos_sim(x, z))
```

## Evaluation of word embeddings

The goal of word embeddings is simple: a good embedding will provide a vector representation of a word such that its vector representation when compared with that of another word mirrors the total linguistic relationship between the two words. In the introduction section, we went over the number of linguistic relationships words can have with one another. The variability in expressiveness of these aforementioned characteristics will influence the "performance" of word embeddings.

The paper by Schnabel et al in 2015 provides a standard for evaluation methods with regards to unsupervised word embeddings, that, in a number of ways is still true today. We can break up evaluation of word embeddings into two distinct categories: Intrinsic and Extrinsic evaluation. Extrinsic evaluation refers to "using word embeddings as input features to a downstream task and measuring changes in performance metrics specific to that task" to evaluate the quality of the word embeddings. Examples of such downstream tasks range from part-of-speech (POS) tagging to named-entity recognition (NER) (Pennington et al., 2014). Extrinsic evaluation only provides a single way in specifying the "goodness" of an embedding, and it is relatively unclear how these measurements connect to other measures of "goodness."

Intrinsic evaluations test for syntactic and semantic relationships between specific words (Mikolov et al, 2013a; Baroni et al. 2014). The tasks typically require and involve a pre-selected set of query terms and semantically related target words, which are referred to as the "query inventory." The methods are then evaluated by compiling aggregate scores for each method such that the correlation coefficient serves as an absolute measure of quality. Query inventories were plethoric as psycholinguistics, information retrieval, and image analysis featured similar datasets, but the specificity of the queries led to poorly calibrated corpus statistics. In Schnabel et. al, they provided a newer model to constructing fair query inventories, picking words in an ad hoc fashion and selecting for diversity with respect to frequency, parts-of-speech, and abstractness.

Intrinsic evaluation is further dissected into four main categories: relatedness, analogies, categorization, and selectional preference. Relatedness refers to testing on datasets containing relatedness scores for pairs of words; the cosine similarity of the embeddings of the two words should have high correlation with the human relatedness scores. The second test is analogies, popularized by Mikolov et. al in 2013. The goal was to find the best term x for a given term y such that the relationship between x and y best matches that of the relationship between a and b. The third task is categorization; the goal being to recover a clustering of words into different categories. To do this, the corresponding word vectors of all words in the dataset are clustered and the purity of the returned clusters is computed with respect to the labeled dataset. Selectional preference is the last task; the goal being to determine how typical a noun is for a verb either as a subject or as an object, (e.g. people eat, but we rarely eat people).

### Word Similarity

Given an input word, we can find the nearest $k$ words from
the vocabulary (400,000 words excluding the unknown token) by similarity. The
similarity between any given pair of words can be represented by the cosine similarity
of their vectors.

We first must normalize each row, followed by taking the dot product of the entire
vocabulary embedding matrix and the single word embedding (`dot_prod`).
We can then find the indices for which the dot product is greatest (`topk`), which happens to be the indices of the most similar words.

```{.python .input}
def norm_vecs_by_row(x):
    return x / nd.sqrt(nd.sum(x * x, axis=1) + 1E-10).reshape((-1,1))

def get_knn(vocab, k, word):
    word_vec = vocab.embedding[word].reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(vocab.embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_vec)
    indices = nd.topk(dot_prod.reshape((len(vocab), )), k=k+1, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # Remove unknown and input tokens.
    return vocab.to_tokens(indices[1:])
```

Let us find the 5 most similar words to 'baby' from the vocabulary (size:
400,000 words).

```{.python .input}
get_knn(vocab, 5, 'baby')
```

We can verify the cosine similarity of the vectors of 'baby' and 'babies'.

```{.python .input}
cos_sim(vocab.embedding['baby'], vocab.embedding['babies'])
```

Let us find the 5 most similar words to 'computers' from the vocabulary.

```{.python .input}
get_knn(vocab, 5, 'computers')
```

Let us find the 5 most similar words to 'run' from the given vocabulary.

```{.python .input}
get_knn(vocab, 5, 'run')
```

Let us find the 5 most similar words to 'beautiful' from the vocabulary.

```{.python .input}
get_knn(vocab, 5, 'beautiful')
```

### Word Analogies

We can also apply pre-trained word embeddings to the word
analogy problem. For example, "man : woman :: son : daughter" is an analogy.
This sentence can also be read as "A man is to a woman as a son is to a daughter."

The word analogy completion problem is defined concretely as: for analogy 'a : b :: c : d',
given the first three words 'a', 'b', 'c', find 'd'. The idea is to find the
most similar word vector for vec('c') + (vec('b')-vec('a')).

In this example,
we will find words that are analogous from the 400,000 indexed words in `vocab`.
dkl

```{.python .input}
def get_top_k_by_analogy(vocab, k, word1, word2, word3):
    word_vecs = vocab.embedding[word1, word2, word3]
    word_diff = (word_vecs[1] - word_vecs[0] + word_vecs[2]).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(vocab.embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_diff)
    indices = nd.topk(dot_prod.reshape((len(vocab), )), k=k, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    return vocab.to_tokens(indices)
```

We leverage this method to find the word to complete the analogy 'man : woman :: son :'.

```{.python .input}
get_top_k_by_analogy(vocab, 1, 'man', 'woman', 'son')
```

Let us verify the cosine similarity between vec('son')+vec('woman')-vec('man')
and vec('daughter').

```{.python .input}
def cos_sim_word_analogy(vocab, word1, word2, word3, word4):
    words = [word1, word2, word3, word4]
    vecs = vocab.embedding[words]
    return cos_sim(vecs[1] - vecs[0] + vecs[2], vecs[3])

cos_sim_word_analogy(vocab, 'man', 'woman', 'son', 'daughter')
```

And to perform some more tests, let's try the following analogy: 'beijing : china :: tokyo : '.

```{.python .input}
get_top_k_by_analogy(vocab, 1, 'beijing', 'china', 'tokyo')
```

And another word analogy: 'bad : worst :: big : '.

```{.python .input}
get_top_k_by_analogy(vocab, 1, 'bad', 'worst', 'big')
```

And the last analogy: 'do : did :: go :'.

```{.python .input}
get_top_k_by_analogy(vocab, 1, 'do', 'did', 'go')
```

## Exercises
1. Rewrite the sentiment analysis notebook's code to deal with a different embedding type you personally implemented.
2.

## References

- https://medium.com/analytics-vidhya/introduction-to-natural-language-processing-part-1-777f972cc7b3
- https://gluon-nlp.mxnet.io/examples/word_embedding/word_embedding.html





### Subword-based and Character-based Embeddings

Learning continuous representations of words has a long history in natural language processing (Rumelhart et al., 1988). These representations are typically derived from large unlabeled corpora using co-occurrence statistics (Deerwester et al., 1990; Schütze, 1992; Lund and Burgess, 1996). A large body of work, known as distributional semantics, has studied the properties of these methods (Turney et al., 2010; Baroni and Lenci, 2010). In the neural network community, Collobert and Weston (2008) proposed to learn word embeddings using a feedforward neural network, by predicting a word based on the two words on the left and two words on the right. More recently, Mikolov et al. (2013b) proposed simple log-bilinear models to learn continuous representations of words on very large corpora efficiently.
Most of these techniques represent each word of the vocabulary by a distinct vector, without parameter sharing. In particular, they ignore the internal structure of words, which is an important limitation for morphologically rich languages, such as Turkish or Finnish. For example, in French or Spanish, most verbs have more than forty different inflected forms, while the Finnish language has fifteen cases for just nouns. These languages contain many word forms that occur rarely (or not at all) in the training corpus, making it difficult to learn good word representations. Because many word formations follow rules, it is possible to improve vector representations for morphologically rich languages by using character level information.

In recent years, many methods have been proposed to incorporate morphological information into word representations. To model rare words better, Alexandrescu and Kirchhoff (2006) introduced factored eural language models, where words are represented as sets of features. These features might include morphological information, and this technique as succesfully applied to morphologically rich languages, such as Turkish (Sak et al., 2010). Recently, several works have proposed different composition functions to derive representations of words from morphemes (Lazaridou et al., 2013; Luong t al., 2013; Botha and Blunsom, 2014; Qiu et l., 2014). These different approaches rely on a morphological decomposition of words.
