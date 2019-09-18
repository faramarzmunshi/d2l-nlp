# Count-Based Methods for Derivation of Word Embeddings

So now that we understand the reason why NLP and representations for natural language concepts are difficult in the last chapter, let's take a look at some of the simple ways in which we can construct vector representations, on a word-level (i.e. for words). The vector representation of a word in NLP is called a **word embedding**.

Word embeddings are a very useful representation of vocabulary for machine learning purposes. The motive of word embeddings is to capture context, semantic, and syntactic similarity, and represent these aspects through a geometric relationship between the embedding vectors. Formulation of these embeddings is often harder as we need to isolate what we actually want to know about the words. Do we want to have embeddings that share a common part of speech appear closer together? Or do we want words with similar meanings appear geometrically closer than those without? How do we derive any of these relationships and how do we represent any of these features using simple means? 

All of these questions are answered in different manners according to the newest models and intuition as the field has developed, changing what word embeddings actually do for the down-stream task.


## The simplest word embedding

By transforming textual data into numerical representations, we are able to train machines over complex variants of mathematical models that pose as intelligent understanding of language. The process of turning text into numbers is commonly known as “vectorization” or “embedding techniques”. These techniques are functions which map words onto vectors of real numbers. These vectors then combine to form a vector space, an algebraic model where all the rules of vector addition and measures of similarities apply.

Using the word embedding technique word2vec, researchers at Google were able to quantify word relationships in an algebraic model. This notebook goes into depth on the basics of count-based methods of quantifying these word embeddings.

Such a vector space is a very useful way of understanding language. We can find “similar words” in a vector space by finding inherent clusters of vectors. We can determine word relations using vector addition. We can measure how similar two words by measuring the angles between the vectors or by examining their dot product.
There are more ways of vectorizing text than mapping every word to a vector. We can also map **documents**, characters or groups of words to vectors as well.

“Document” is a term that's oft used in the NLP field. It refers to an "unbroken entity of text," usually one that is of interest to the analysis task. For example, if you are trying to create an algorithm to identify sentiment on movie reviews, each review would be its own document, and an analysis of the reviews would be considered a document-level analysis or a document-level sentiment analysis. 

Vectorizing documents makes it useful to compare text at the document level. At the document level, with enough data, we can derive word embeddings. Document level modeling is useful in many applications including topic modeling and text classification. Vectorizing groups of words helps us differentiate between the words with more than one semantic meaning. For example, “crash” can refer to a “car crash” or a “financial crash” or barging into a party uninvited.

All of these embedding techniques are reliant on the **distributional hypothesis**, the assumption that “words which are used and occur in the same contexts tend to purport similar meaning.”

### Count-based vectors

This is one of the simplest methods of embedding words into numerical vectors. It is not often used in practice due to its oversimplification of language, but often the first embedding technique to be taught in a classroom setting.

Let’s consider the following documents. If it helps, you can imagine that they are text messages shared between friends.

- Document 1: Hang ten!

- Document 2: I am fat.

- Document 3: She has ten.


The *vocabulary*, referred in this book as a Vocab object, we obtain from this set of documents is (hang, ten, I, am, fat, she, is). We will ignore punctuation (commas, periods, exclamation marks, all of which could be leveraged) for now, although depending on our use case it can also make a lot of sense to incorporate them into our vocabulary.

We can create a matrix representing the relationship between each term from our vocabulary and the document. Each element in the matrix represents how many times that term appears in that particular document.

|  -    | Document 1 | Document 2 | Document 3 |
|------|:----------:|------------|------------|
| Hang |      1     | 0          | 0          |
| ten  |      1     | 0          | 1          |
| I    |      0     | 1          | 0          |
| am   |      0     | 1          | 0          |
| fat  | 0          | 1          | 0          |
| she  | 0          | 0          | 1          |
| has  | 0          | 0          | 1          |

Using this matrix, we can obtain the vectors for each word as well as document. We can vectorize “ten” as `[1,0,1]` and the entirety of the second document can be vectorized as `[0,0,1,1,1,0,0]`.

Bag of words is not a good representation of words or documents, especially when using a relatively sparse/small vocabulary. Bag of words on the whole ignores order of words, relationships between words and produces very sparse vectors. We also see here from our small example that the words “I”, “am”, “fat” are mapped to the same word embedding. Using our hypothesis above, we can conclusively say that even though these are geometrically similar, linguistically they are not. Hence this is a "bad" representation.

### TF/IDF

This is another method of word embeddings that is slightly more advanced, based on the frequency method. But it is fundamentally divergent to the above count-based vectors because the method uses not just the number of occurrences of a word in a single document but in the entire corpus.

The most common words like 'he', 'they', 'a', 'the' etc. appear more frequently in comparison to the words which are "important" to the meaning of a document. For example, a document `A` on geurilla warfare is going to contain more occurences of the word “geurilla” in comparison to other documents. But common words like “the” etc. are also going to be present in higher frequency in basically every document.

Ideally, we want to scale down the importance or weight of common words that are oft-occurring in most documents and scale up the weight of words that appear less in a smaller subset of the documents.

TF-IDF works like this. It penalises common words by assigning lower weights and giving importance to words like "geurilla," which is key in particular documents.

Consider the two document's tables below:

**Document 1**

| Term     | Count |
|----------|:-----:|
| This     |   1   |
| is       |   1   |
| geurilla |   4   |
| warfare  |   2   |

**Document 2**

| Term     | Count |
|----------|:-----:|
| This     |   1   |
| is       |   2   |
| about    |   1   |
| TF-IDF   |   1   |


Let's define some terms:

TF stands for term frequency. This can be mathematically defined as


<center>$TF = \frac{\text{# of times term t appears in a document}}{\text{# of terms in the document}}$</center>

So, 


<center>$TF(This,Document1) = \frac{1}{8} $</center>


<center>$TF(This, Document2) = \frac{1}{5}$</center>

It denotes the contribution of the word to the document i.e words relevant to the document should be frequent. eg: A document about geurilla should contain the word ‘geurilla’ in large number.


<center>$IDF = log(\frac{N}{n})$</center>

where, `N` is the number of documents and `n` is the number of documents a term t has appeared in.

So, 


<center>$IDF(This) = log(\frac{2}{2}) = 0$</center>

Ideally, if a word has appeared in all the document, then that word is most likely not relevant to particular documents. But if it has appeared in a subset of documents then probably the word is of some relevance to the documents it is present in.

Let's compute IDF for the word ‘geurilla’.


<center>$IDF(geurilla) = log(\frac{2}{1}) = 0.301$</center>

Now, let's compare the TF-IDF for a common word like ‘this’ and the word ‘geurilla’ which seems to be of relevance to our first document.


<center>$TFIDF(This,Document1) = (\frac{1}{8}) * (0) = 0$</center>


<center>$TFIDF(This, Document2) = (\frac{1}{5}) * (0) = 0$</center>


<center>$TFIDF(geurilla, Document1) = (\frac{4}{8})*0.301 = 0.15$</center>

As, you can see for Document1 , TF-IDF method heavily penalises the word ‘This’ but assigns greater weight to ‘geurilla’. So, this may be understood as ‘geurilla’ is an important word for Document 1 from the context of the entire corpus.

## Implementation of TF-IDF

Now that we understand the basic mathematics behind this approach, let's implement simple TF-IDF computation for the given PTB corpus (Penn Tree Bank). Let's start with defining our corpus and preprocessing the text to make the vectorization as easy as possible.

```{.python .input}
import d2l
import zipfile
import math
import pprint

def read_ptb():
    with zipfile.ZipFile('../data/ptb.zip', 'r') as f:
        raw_text = f.read('ptb/ptb.train.txt').decode("utf-8")
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
'# sentences: %d' % len(sentences)

# Let's trim the number of sentences to make the calculations faster; if you don't do this
# the IDF calculations will take a long time
sentences = sentences[0:4000]
```

```{.python .input}
vocab = d2l.Vocab(sentences, min_freq=10)
'vocab size: %d' % len(vocab)
```

Now we have a large vocabulary or `vocab` object that contains every word in PTB that occurs more than 10 times and a specific index number for each. Let's look up the indices of hello and world.

```{.python .input}
print(vocab['hello'])
print(vocab['world'])
```

And we can go the other way, from token number to the actual word.

```{.python .input}
print(vocab.to_tokens(218))
print(sentences[2])
```

This vocab object will help us later on but an introduction to it is necessary for future notebooks.

We now define a function to return the documents and calculate the total word count of each document. In this case, each sentence is being defined as its own document. We define some helper functions here `count_words` and `get_doc` that return the number of words and the document level info in a nice dictionary with the number of words and a document id respectively.

```{.python .input}
def count_words(sentence):
    count = 0
    for word in sentence:
        count+=1
    return count
    
def get_doc(sentences):
    doc_info = []
    i = 0
    for sentence in sentences:
        i+=1
        count = count_words(sentence)
        temp = {'doc_id': i, 'doc_length': count}
        doc_info.append(temp)
    return doc_info
```

Next we create a frequency dictionary using the document info we've been given.

```{.python .input}
def create_frequency_dictionary(sentences):
    i = 0
    freq_dict_list = []
    for sentence in sentences:
        i+=1
        freq_dict = {}
        for word in sentence:
            word = word.lower()
            if word in freq_dict:
                freq_dict[word] +=1
            else:
                freq_dict[word] = 1
            temp = {'doc_id':i, 'freq_dict': freq_dict}
        freq_dict_list.append(temp)
    return freq_dict_list
```

And then finally, we compute TF, IDF and then TFIDF using the outputs of the TF and IDF computation functions defined below.

```{.python .input}
def computeTF(doc_info, freq_dict_list):
    TF_scores = []
    for temp_dict in freq_dict_list:
        doc_id = temp_dict['doc_id']
        for k in temp_dict['freq_dict']:
            temp = {'doc_id': doc_id,
                    'TF_score': temp_dict['freq_dict'][k]/doc_info[doc_id-1]['doc_length'],
                    'key': k}
            TF_scores.append(temp)
    return TF_scores

def computeIDF(doc_info, freq_dict_list):
    IDF_scores = []
    counter = 0
    for d in freq_dict_list:
        counter+=1
        for k in d['freq_dict'].keys():
            count = sum([k in temp_dict['freq_dict'] for temp_dict in freq_dict_list])
            temp = {'doc_id': counter,
                    'IDF_score': math.log(len(doc_info)/count),
                    'key': k}
            
            IDF_scores.append(temp)
    return IDF_scores
    
def computeTFIDF(TF_scores, IDF_scores):
    TFIDF_scores = []
    for j in IDF_scores:
        for i in TF_scores:
            if j['key'] == i['key'] and j['doc_id'] == i['doc_id']:
                temp = {'doc_id': j['doc_id'],
                        'TFIDF_score': j['IDF_score']*i['TF_score'],
                        'key': i['key']}
        TFIDF_scores.append(temp)
    return TFIDF_scores
```

Now let's see the outputs of each step for our given PTB corpus.

```{.python .input}
doc_info = get_doc(sentences)
pprint.pprint(doc_info[:10])
```

```{.python .input}
freq_dict_list = create_frequency_dictionary(sentences)
pprint.pprint(freq_dict_list[:10])
```

```{.python .input}
TF_scores = computeTF(doc_info, freq_dict_list)
pprint.pprint(TF_scores[:30])
```

```{.python .input}
IDF_scores = computeIDF(doc_info, freq_dict_list)
pprint.pprint(IDF_scores[:30])
```

```{.python .input}
TFIDF_scores = computeTFIDF(TF_scores, IDF_scores)
pprint.pprint(TFIDF_scores[:30])
```

We can clearly see that the words that occur once have a higher TFIDF score than those that occur more than once, not just in the same sentence, but across all the input sentences. The word embeddings for each word would be the following:

```{.python .input}
def get_embeddings(TFIDF_scores):
    embedding_dict = {}
    # Go through every key, document_id, and TFIDF combination
    for d in TFIDF_scores:
        doc_counter = d['doc_id'] - 1
        word = d['key']
        TFIDF = d['TFIDF_score']
        
        # Check if the word is in the embedding dictionary, if it isn't, zero pad until the document number
        # and append the TFIDF score on the end
        if word not in embedding_dict.keys():
            embedding_dict[word] = ([0] * doc_counter) + [TFIDF]
        
        # If the word is already in the embedding dictionary, zero pad until the document number and then append
        # the TFIDF score on the end
        else:
            embedding_dict[word] += [0] * (doc_counter-len(embedding_dict[word])) + [TFIDF]
    
    # Make sure that all the embeddings are the same length, and if they aren't, zero pad until they are
    for word in embedding_dict.keys():
        if len(embedding_dict[word]) <= doc_counter:
            embedding_dict[word] += [0] * (doc_counter - len(embedding_dict[word]) + 1)
    
    return embedding_dict

embedding_dict = get_embeddings(TFIDF_scores)
```

Let's make sure that all the keys have the same length vector.

```{.python .input}
for key in embedding_dict.keys():
    assert len(embedding_dict[key]) == 4000, len(embedding_dict[key])
```

And let's print a sample word embedding for the word "the" in this corpus.

```{.python .input}
embedding_dict['the']
```

In the next notebook, we'll look at a more effective way of calculating word embeddings using more of the idea of context.

## References:

- https://towardsdatascience.com/tfidf-for-piece-of-text-in-python-43feccaa74f8
- https://medium.com/analytics-vidhya/introduction-to-natural-language-processing-part-1-777f972cc7b3

