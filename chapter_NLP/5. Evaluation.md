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
