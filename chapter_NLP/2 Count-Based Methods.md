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

```{.python .input  n=1}
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

```{.python .input  n=2}
vocab = d2l.Vocab(sentences, min_freq=10)
'vocab size: %d' % len(vocab)
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "'vocab size: 1186'"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Now we have a large vocabulary or `vocab` object that contains every word in PTB that occurs more than 10 times and a specific index number for each. Let's look up the indices of hello and world.

```{.python .input  n=3}
print(vocab['hello'])
print(vocab['world'])
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0\n268\n"
 }
]
```

And we can go the other way, from token number to the actual word.

```{.python .input  n=4}
print(vocab.to_tokens(218))
print(sentences[2])
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "early\n['mr.', '<unk>', 'is', 'chairman', 'of', '<unk>', 'n.v.', 'the', 'dutch', 'publishing', 'group']\n"
 }
]
```

This vocab object will help us later on but an introduction to it is necessary for future notebooks.

We now define a function to return the documents and calculate the total word count of each document. In this case, each sentence is being defined as its own document. We define some helper functions here `count_words` and `get_doc` that return the number of words and the document level info in a nice dictionary with the number of words and a document id respectively.

```{.python .input  n=5}
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

```{.python .input  n=6}
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

```{.python .input  n=7}
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

```{.python .input  n=8}
doc_info = get_doc(sentences)
pprint.pprint(doc_info[:10])
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[{'doc_id': 1, 'doc_length': 24},\n {'doc_id': 2, 'doc_length': 15},\n {'doc_id': 3, 'doc_length': 11},\n {'doc_id': 4, 'doc_length': 23},\n {'doc_id': 5, 'doc_length': 34},\n {'doc_id': 6, 'doc_length': 27},\n {'doc_id': 7, 'doc_length': 23},\n {'doc_id': 8, 'doc_length': 32},\n {'doc_id': 9, 'doc_length': 9},\n {'doc_id': 10, 'doc_length': 15}]\n"
 }
]
```

```{.python .input  n=9}
freq_dict_list = create_frequency_dictionary(sentences)
pprint.pprint(freq_dict_list[:10])
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[{'doc_id': 1,\n  'freq_dict': {'aer': 1,\n                'banknote': 1,\n                'berlitz': 1,\n                'calloway': 1,\n                'centrust': 1,\n                'cluett': 1,\n                'fromstein': 1,\n                'gitano': 1,\n                'guterman': 1,\n                'hydro-quebec': 1,\n                'ipo': 1,\n                'kia': 1,\n                'memotec': 1,\n                'mlx': 1,\n                'nahb': 1,\n                'punts': 1,\n                'rake': 1,\n                'regatta': 1,\n                'rubens': 1,\n                'sim': 1,\n                'snack-food': 1,\n                'ssangyong': 1,\n                'swapo': 1,\n                'wachter': 1}},\n {'doc_id': 2,\n  'freq_dict': {'<unk>': 1,\n                'a': 1,\n                'as': 1,\n                'board': 1,\n                'director': 1,\n                'join': 1,\n                'n': 2,\n                'nonexecutive': 1,\n                'nov.': 1,\n                'old': 1,\n                'pierre': 1,\n                'the': 1,\n                'will': 1,\n                'years': 1}},\n {'doc_id': 3,\n  'freq_dict': {'<unk>': 2,\n                'chairman': 1,\n                'dutch': 1,\n                'group': 1,\n                'is': 1,\n                'mr.': 1,\n                'n.v.': 1,\n                'of': 1,\n                'publishing': 1,\n                'the': 1}},\n {'doc_id': 4,\n  'freq_dict': {'<unk>': 1,\n                'a': 1,\n                'and': 1,\n                'british': 1,\n                'chairman': 1,\n                'conglomerate': 1,\n                'consolidated': 1,\n                'director': 1,\n                'fields': 1,\n                'former': 1,\n                'gold': 1,\n                'industrial': 1,\n                'n': 1,\n                'named': 1,\n                'nonexecutive': 1,\n                'of': 2,\n                'old': 1,\n                'plc': 1,\n                'rudolph': 1,\n                'this': 1,\n                'was': 1,\n                'years': 1}},\n {'doc_id': 5,\n  'freq_dict': {'a': 3,\n                'ago': 1,\n                'among': 1,\n                'asbestos': 1,\n                'cancer': 1,\n                'caused': 1,\n                'cigarette': 1,\n                'deaths': 1,\n                'exposed': 1,\n                'filters': 1,\n                'form': 1,\n                'group': 1,\n                'has': 1,\n                'high': 1,\n                'it': 1,\n                'kent': 1,\n                'make': 1,\n                'more': 1,\n                'n': 1,\n                'of': 3,\n                'once': 1,\n                'percentage': 1,\n                'reported': 1,\n                'researchers': 1,\n                'than': 1,\n                'to': 2,\n                'used': 1,\n                'workers': 1,\n                'years': 1}},\n {'doc_id': 6,\n  'freq_dict': {'<unk>': 3,\n                'asbestos': 1,\n                'brief': 1,\n                'causing': 1,\n                'decades': 1,\n                'enters': 1,\n                'even': 1,\n                'exposures': 1,\n                'fiber': 1,\n                'is': 1,\n                'it': 2,\n                'later': 1,\n                'once': 1,\n                'researchers': 1,\n                'said': 1,\n                'show': 1,\n                'symptoms': 1,\n                'that': 1,\n                'the': 2,\n                'to': 1,\n                'unusually': 1,\n                'up': 1,\n                'with': 1}},\n {'doc_id': 7,\n  'freq_dict': {'<unk>': 4,\n                'cigarette': 1,\n                'cigarettes': 1,\n                'corp.': 1,\n                'filters': 1,\n                'in': 2,\n                'inc.': 1,\n                'its': 1,\n                'kent': 1,\n                'makes': 1,\n                'n': 1,\n                'new': 1,\n                'of': 1,\n                'stopped': 1,\n                'that': 1,\n                'the': 1,\n                'unit': 1,\n                'using': 1,\n                'york-based': 1}},\n {'doc_id': 8,\n  'freq_dict': {\"'s\": 1,\n                'a': 2,\n                'ago': 1,\n                'although': 1,\n                'appear': 1,\n                'attention': 1,\n                'bring': 1,\n                'england': 1,\n                'findings': 1,\n                'forum': 1,\n                'in': 1,\n                'journal': 1,\n                'latest': 1,\n                'likely': 1,\n                'medicine': 1,\n                'more': 1,\n                'new': 2,\n                'of': 1,\n                'preliminary': 1,\n                'problem': 1,\n                'reported': 1,\n                'results': 1,\n                'than': 1,\n                'the': 2,\n                'to': 2,\n                'today': 1,\n                'were': 1,\n                'year': 1}},\n {'doc_id': 9,\n  'freq_dict': {'<unk>': 2,\n                'a': 1,\n                'an': 1,\n                'is': 1,\n                'old': 1,\n                'said': 1,\n                'story': 1,\n                'this': 1}},\n {'doc_id': 10,\n  'freq_dict': {\"'re\": 1,\n                'about': 1,\n                'ago': 1,\n                'any': 1,\n                'anyone': 1,\n                'asbestos': 1,\n                'before': 1,\n                'having': 1,\n                'heard': 1,\n                'of': 1,\n                'properties': 1,\n                'questionable': 1,\n                'talking': 1,\n                'we': 1,\n                'years': 1}}]\n"
 }
]
```

```{.python .input  n=10}
TF_scores = computeTF(doc_info, freq_dict_list)
pprint.pprint(TF_scores[:30])
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[{'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'aer'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'banknote'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'berlitz'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'calloway'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'centrust'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'cluett'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'fromstein'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'gitano'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'guterman'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'hydro-quebec'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'ipo'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'kia'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'memotec'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'mlx'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'nahb'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'punts'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'rake'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'regatta'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'rubens'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'sim'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'snack-food'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'ssangyong'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'swapo'},\n {'TF_score': 0.041666666666666664, 'doc_id': 1, 'key': 'wachter'},\n {'TF_score': 0.06666666666666667, 'doc_id': 2, 'key': 'pierre'},\n {'TF_score': 0.06666666666666667, 'doc_id': 2, 'key': '<unk>'},\n {'TF_score': 0.13333333333333333, 'doc_id': 2, 'key': 'n'},\n {'TF_score': 0.06666666666666667, 'doc_id': 2, 'key': 'years'},\n {'TF_score': 0.06666666666666667, 'doc_id': 2, 'key': 'old'},\n {'TF_score': 0.06666666666666667, 'doc_id': 2, 'key': 'will'}]\n"
 }
]
```

```{.python .input  n=11}
IDF_scores = computeIDF(doc_info, freq_dict_list)
pprint.pprint(IDF_scores[:30])
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[{'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'aer'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'banknote'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'berlitz'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'calloway'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'centrust'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'cluett'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'fromstein'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'gitano'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'guterman'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'hydro-quebec'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'ipo'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'kia'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'memotec'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'mlx'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'nahb'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'punts'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'rake'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'regatta'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'rubens'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'sim'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'snack-food'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'ssangyong'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'swapo'},\n {'IDF_score': 8.294049640102028, 'doc_id': 1, 'key': 'wachter'},\n {'IDF_score': 8.294049640102028, 'doc_id': 2, 'key': 'pierre'},\n {'IDF_score': 0.6143360001356556, 'doc_id': 2, 'key': '<unk>'},\n {'IDF_score': 1.1324276371628403, 'doc_id': 2, 'key': 'n'},\n {'IDF_score': 3.5491175117387774, 'doc_id': 2, 'key': 'years'},\n {'IDF_score': 4.767689115485866, 'doc_id': 2, 'key': 'old'},\n {'IDF_score': 2.645075401940822, 'doc_id': 2, 'key': 'will'}]\n"
 }
]
```

```{.python .input  n=12}
TFIDF_scores = computeTFIDF(TF_scores, IDF_scores)
pprint.pprint(TFIDF_scores[:30])
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[{'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'aer'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'banknote'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'berlitz'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'calloway'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'centrust'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'cluett'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'fromstein'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'gitano'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'guterman'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'hydro-quebec'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'ipo'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'kia'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'memotec'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'mlx'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'nahb'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'punts'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'rake'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'regatta'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'rubens'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'sim'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'snack-food'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'ssangyong'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'swapo'},\n {'TFIDF_score': 0.3455854016709178, 'doc_id': 1, 'key': 'wachter'},\n {'TFIDF_score': 0.5529366426734685, 'doc_id': 2, 'key': 'pierre'},\n {'TFIDF_score': 0.04095573334237704, 'doc_id': 2, 'key': '<unk>'},\n {'TFIDF_score': 0.15099035162171204, 'doc_id': 2, 'key': 'n'},\n {'TFIDF_score': 0.2366078341159185, 'doc_id': 2, 'key': 'years'},\n {'TFIDF_score': 0.31784594103239106, 'doc_id': 2, 'key': 'old'},\n {'TFIDF_score': 0.1763383601293881, 'doc_id': 2, 'key': 'will'}]\n"
 }
]
```

We can clearly see that the words that occur once have a higher TFIDF score than those that occur more than once, not just in the same sentence, but across all the input sentences. The word embeddings for each word would be the following:

```{.python .input  n=13}
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

```{.python .input  n=14}
for key in embedding_dict.keys():
    assert len(embedding_dict[key]) == 4000, len(embedding_dict[key])
```

And let's print a sample word embedding for the word "the" in this corpus.

```{.python .input  n=15}
embedding_dict['the']
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "[0,\n 0.026450211167076187,\n 0.03606846977328571,\n 0,\n 0,\n 0.02938912351897354,\n 0.017250137717658383,\n 0.024797072969133926,\n 0,\n 0,\n 0,\n 0.05951297512592142,\n 0,\n 0.03606846977328571,\n 0.03839546782317511,\n 0.04959414593826785,\n 0.041763491316436085,\n 0.026450211167076187,\n 0.03967531675061428,\n 0.022041842639230154,\n 0.04959414593826785,\n 0.03967531675061428,\n 0.041763491316436085,\n 0.033062763958845234,\n 0.0233384216180084,\n 0.024797072969133926,\n 0,\n 0.03967531675061428,\n 0.06612552791769047,\n 0.024797072969133926,\n 0.03967531675061428,\n 0,\n 0,\n 0,\n 0,\n 0.03400741435766939,\n 0.026450211167076187,\n 0.034500275435316766,\n 0,\n 0,\n 0.03606846977328571,\n 0,\n 0.01983765837530714,\n 0,\n 0.041763491316436085,\n 0.020881745658218043,\n 0.03606846977328571,\n 0.03174025340049143,\n 0,\n 0.04959414593826785,\n 0.03051947442354945,\n 0,\n 0,\n 0,\n 0,\n 0.0466768432360168,\n 0.015870126700245714,\n 0.028339511964724486,\n 0,\n 0,\n 0.041763491316436085,\n 0.03967531675061428,\n 0.0233384216180084,\n 0.052900422334152375,\n 0.0610389488470989,\n 0.0466768432360168,\n 0,\n 0.024797072969133926,\n 0,\n 0,\n 0.027362287414216747,\n 0,\n 0.0466768432360168,\n 0.0233384216180084,\n 0.03400741435766939,\n 0.041043431121325115,\n 0.033062763958845234,\n 0,\n 0.03174025340049143,\n 0.08501853589417345,\n 0.0233384216180084,\n 0.047610380100737135,\n 0.05667902392944897,\n 0.01983765837530714,\n 0.0700152648540252,\n 0,\n 0.0350076324270126,\n 0.04408368527846031,\n 0.01983765837530714,\n 0,\n 0.016531381979422617,\n 0.027362287414216747,\n 0,\n 0,\n 0.03967531675061428,\n 0.020881745658218043,\n 0.03606846977328571,\n 0,\n 0,\n 0.0610389488470989,\n 0.01889300797648299,\n 0,\n 0,\n 0,\n 0,\n 0.022041842639230154,\n 0.012798489274391704,\n 0.01845363569796013,\n 0,\n 0,\n 0.03606846977328571,\n 0.01983765837530714,\n 0.041763491316436085,\n 0.03051947442354945,\n 0.04289223432498841,\n 0.01983765837530714,\n 0.01889300797648299,\n 0,\n 0,\n 0.00809700341849271,\n 0.047610380100737135,\n 0,\n 0,\n 0.03051947442354945,\n 0,\n 0.05175041315297515,\n 0.05472457482843349,\n 0.0233384216180084,\n 0.04959414593826785,\n 0.015870126700245714,\n 0.041763491316436085,\n 0,\n 0,\n 0.018034234886642856,\n 0,\n 0.03606846977328571,\n 0,\n 0.014169755982362243,\n 0.024797072969133926,\n 0.020881745658218043,\n 0.03606846977328571,\n 0.022041842639230154,\n 0,\n 0.02938912351897354,\n 0,\n 0.028339511964724486,\n 0.03174025340049143,\n 0.03967531675061428,\n 0.041763491316436085,\n 0.03606846977328571,\n 0.028339511964724486,\n 0.024797072969133926,\n 0.05667902392944897,\n 0.015870126700245714,\n 0,\n 0,\n 0.05667902392944897,\n 0,\n 0.024045646515523808,\n 0.024797072969133926,\n 0.03606846977328571,\n 0.03606846977328571,\n 0,\n 0.01469456175948677,\n 0,\n 0.02938912351897354,\n 0.03051947442354945,\n 0.03839546782317511,\n 0.01889300797648299,\n 0.04408368527846031,\n 0.045343219143559176,\n 0.03778601595296598,\n 0.051193957097566814,\n 0,\n 0.05951297512592142,\n 0,\n 0.020881745658218043,\n 0,\n 0.041043431121325115,\n 0.013681143707108373,\n 0.01889300797648299,\n 0,\n 0,\n 0.03606846977328571,\n 0.015870126700245714,\n 0.04959414593826785,\n 0.02975648756296071,\n 0.018034234886642856,\n 0.027362287414216747,\n 0.012022823257761904,\n 0.04408368527846031,\n 0.06612552791769047,\n 0,\n 0,\n 0.025596978548783407,\n 0.03606846977328571,\n 0.045343219143559176,\n 0.022041842639230154,\n 0.027362287414216747,\n 0.04408368527846031,\n 0,\n 0.009446503988241496,\n 0,\n 0.01983765837530714,\n 0,\n 0,\n 0,\n 0.033062763958845234,\n 0.033062763958845234,\n 0,\n 0,\n 0,\n 0,\n 0.010723058581247103,\n 0,\n 0.013225105583538094,\n 0.016531381979422617,\n 0.025596978548783407,\n 0,\n 0,\n 0,\n 0.012398536484566963,\n 0,\n 0.012022823257761904,\n 0.03967531675061428,\n 0,\n 0.022041842639230154,\n 0.01889300797648299,\n 0,\n 0.02938912351897354,\n 0.01983765837530714,\n 0,\n 0.03967531675061428,\n 0.03174025340049143,\n 0.03606846977328571,\n 0.028339511964724486,\n 0.026450211167076187,\n 0.016531381979422617,\n 0,\n 0,\n 0.0350076324270126,\n 0.05667902392944897,\n 0.034500275435316766,\n 0.03051947442354945,\n 0.03051947442354945,\n 0.04959414593826785,\n 0,\n 0.013225105583538094,\n 0.03967531675061428,\n 0.020881745658218043,\n 0,\n 0.027362287414216747,\n 0,\n 0,\n 0,\n 0.04408368527846031,\n 0.034500275435316766,\n 0.03606846977328571,\n 0,\n 0,\n 0.028339511964724486,\n 0,\n 0.015870126700245714,\n 0.03606846977328571,\n 0.01889300797648299,\n 0.04959414593826785,\n 0,\n 0.04959414593826785,\n 0.03967531675061428,\n 0.03606846977328571,\n 0.04408368527846031,\n 0.025596978548783407,\n 0.024045646515523808,\n 0.07213693954657142,\n 0.0350076324270126,\n 0.03051947442354945,\n 0,\n 0,\n 0.03778601595296598,\n 0.028339511964724486,\n 0.06612552791769047,\n 0.017250137717658383,\n 0.028339511964724486,\n 0,\n 0,\n 0.01889300797648299,\n 0.041763491316436085,\n 0.014169755982362243,\n 0.024797072969133926,\n 0,\n 0.026450211167076187,\n 0.04408368527846031,\n 0.04408368527846031,\n 0,\n 0,\n 0,\n 0.017250137717658383,\n 0.03174025340049143,\n 0.026450211167076187,\n 0.03967531675061428,\n 0,\n 0,\n 0.03778601595296598,\n 0.03967531675061428,\n 0.012398536484566963,\n 0.026450211167076187,\n 0.03606846977328571,\n 0.012398536484566963,\n 0.03216917574374131,\n 0.01889300797648299,\n 0.03400741435766939,\n 0.024797072969133926,\n 0.022671609571779588,\n 0.05667902392944897,\n 0,\n 0.05667902392944897,\n 0.013681143707108373,\n 0.0991882918765357,\n 0.06348050680098286,\n 0.03778601595296598,\n 0,\n 0.013681143707108373,\n 0.012022823257761904,\n 0.03606846977328571,\n 0.03400741435766939,\n 0.13225105583538094,\n 0.018034234886642856,\n 0.04959414593826785,\n 0.04959414593826785,\n 0.017250137717658383,\n 0.034500275435316766,\n 0,\n 0.013225105583538094,\n 0.04959414593826785,\n 0.02938912351897354,\n 0.03051947442354945,\n 0,\n 0.026450211167076187,\n 0.018034234886642856,\n 0,\n 0.04959414593826785,\n 0.052900422334152375,\n 0.045779211635324175,\n 0.028339511964724486,\n 0.02938912351897354,\n 0,\n 0.03606846977328571,\n 0,\n 0,\n 0,\n 0,\n 0,\n 0.04959414593826785,\n 0,\n 0.026450211167076187,\n 0.020881745658218043,\n 0.026450211167076187,\n 0.06612552791769047,\n 0.029030719573620203,\n 0.03839546782317511,\n 0.06264523697465413,\n 0.017250137717658383,\n 0.03967531675061428,\n 0.0466768432360168,\n 0.031322618487327064,\n 0.03051947442354945,\n 0.06612552791769047,\n 0,\n 0.10580084466830475,\n 0.0350076324270126,\n 0.041043431121325115,\n 0.04959414593826785,\n 0,\n 0.033062763958845234,\n 0.0233384216180084,\n 0.05175041315297515,\n 0,\n 0.027362287414216747,\n 0,\n 0.03778601595296598,\n 0,\n 0.07935063350122856,\n 0.033062763958845234,\n 0,\n 0.024797072969133926,\n 0,\n 0.04408368527846031,\n 0.026450211167076187,\n 0,\n 0,\n 0,\n 0.01469456175948677,\n 0.020881745658218043,\n 0,\n 0,\n 0.025596978548783407,\n 0.020881745658218043,\n 0.033062763958845234,\n 0,\n 0.03606846977328571,\n 0.016531381979422617,\n 0,\n 0.0116692108090042,\n 0.028339511964724486,\n 0,\n 0,\n 0.026450211167076187,\n 0.03967531675061428,\n 0.03606846977328571,\n 0,\n 0.026450211167076187,\n 0.03606846977328571,\n 0.01983765837530714,\n 0.033062763958845234,\n 0.0466768432360168,\n 0.020881745658218043,\n 0.0233384216180084,\n 0.03051947442354945,\n 0.052900422334152375,\n 0.04959414593826785,\n 0.018034234886642856,\n 0,\n 0,\n 0.024797072969133926,\n 0.0610389488470989,\n 0.07213693954657142,\n 0.028339511964724486,\n 0,\n 0,\n 0.03051947442354945,\n 0.020881745658218043,\n 0,\n 0,\n 0.020881745658218043,\n 0.024797072969133926,\n 0.026450211167076187,\n 0.028339511964724486,\n 0,\n 0,\n 0,\n 0,\n 0.02938912351897354,\n 0.03719560945370089,\n 0.015259737211774725,\n 0.022041842639230154,\n 0,\n 0.0116692108090042,\n 0,\n 0.03839546782317511,\n 0.03967531675061428,\n 0,\n 0.01889300797648299,\n 0,\n 0.025596978548783407,\n 0,\n 0,\n 0,\n 0.020881745658218043,\n 0.03560605349414102,\n 0.01889300797648299,\n 0.013681143707108373,\n 0,\n 0.026450211167076187,\n 0.024797072969133926,\n 0.033062763958845234,\n 0.042509267947086725,\n 0,\n 0.0350076324270126,\n 0,\n 0,\n 0.026450211167076187,\n 0.03051947442354945,\n 0,\n 0.017250137717658383,\n 0,\n 0.022671609571779588,\n 0,\n 0,\n 0,\n 0.05175041315297515,\n 0.058346054045021,\n 0.03051947442354945,\n 0.012022823257761904,\n 0.0233384216180084,\n 0.048091293031047616,\n 0.03967531675061428,\n 0.051193957097566814,\n 0.03174025340049143,\n 0.03967531675061428,\n 0.03967531675061428,\n 0.041763491316436085,\n 0.04408368527846031,\n 0.03967531675061428,\n 0.028339511964724486,\n 0.027362287414216747,\n 0.016531381979422617,\n 0,\n 0.020881745658218043,\n 0.027362287414216747,\n 0.01889300797648299,\n 0.022041842639230154,\n 0.052900422334152375,\n 0,\n 0.03606846977328571,\n 0.024797072969133926,\n 0.05175041315297515,\n 0,\n 0.03967531675061428,\n 0.06612552791769047,\n 0.03174025340049143,\n 0.022041842639230154,\n 0.08501853589417345,\n 0.028339511964724486,\n 0,\n 0,\n 0.03216917574374131,\n 0,\n 0.0233384216180084,\n 0.04959414593826785,\n 0,\n 0.016883113510899692,\n 0.0350076324270126,\n 0.022041842639230154,\n 0,\n 0.01889300797648299,\n 0,\n 0.07935063350122856,\n 0,\n 0.03606846977328571,\n 0,\n 0,\n 0.020881745658218043,\n 0,\n 0.03967531675061428,\n 0.034500275435316766,\n 0.022041842639230154,\n 0.015870126700245714,\n 0.028339511964724486,\n 0,\n 0.033062763958845234,\n 0.022041842639230154,\n 0.016531381979422617,\n 0.026450211167076187,\n 0.02938912351897354,\n 0,\n 0,\n 0,\n 0,\n 0,\n 0.015870126700245714,\n 0,\n 0.03967531675061428,\n 0.018034234886642856,\n 0.01983765837530714,\n 0,\n 0,\n 0.03400741435766939,\n 0.07935063350122856,\n 0.03606846977328571,\n 0.0933536864720336,\n 0.05951297512592142,\n 0.03400741435766939,\n 0.020346316282366297,\n 0.022041842639230154,\n 0.03719560945370089,\n 0,\n 0.022041842639230154,\n 0.06264523697465413,\n 0.03174025340049143,\n 0,\n 0.01983765837530714,\n 0,\n 0.013681143707108373,\n 0.010440872829109021,\n 0.028339511964724486,\n 0,\n 0.017250137717658383,\n 0.03606846977328571,\n 0.04959414593826785,\n 0,\n 0.04408368527846031,\n 0,\n 0.03967531675061428,\n 0.033062763958845234,\n 0.07935063350122856,\n 0,\n 0,\n 0.013681143707108373,\n 0.01889300797648299,\n 0.017250137717658383,\n 0.03967531675061428,\n 0.018034234886642856,\n 0,\n 0.03051947442354945,\n 0.0233384216180084,\n 0.03778601595296598,\n 0,\n 0,\n 0.014169755982362243,\n 0.050865790705915744,\n 0.027680453546940195,\n 0.04069263256473259,\n 0.033062763958845234,\n 0,\n 0,\n 0.03778601595296598,\n 0.015259737211774725,\n 0.015870126700245714,\n 0.014169755982362243,\n 0.0350076324270126,\n 0.04959414593826785,\n 0,\n 0,\n 0.014169755982362243,\n 0,\n 0,\n 0,\n 0.027362287414216747,\n 0.04408368527846031,\n 0.03967531675061428,\n 0.03606846977328571,\n 0,\n 0.033062763958845234,\n 0,\n 0.07935063350122856,\n 0,\n 0.047610380100737135,\n 0.020881745658218043,\n 0,\n 0.011020921319615077,\n 0.03606846977328571,\n 0,\n 0.0466768432360168,\n 0.033062763958845234,\n 0.026450211167076187,\n 0.022041842639230154,\n 0.0233384216180084,\n 0.0233384216180084,\n 0.042509267947086725,\n 0,\n 0.024797072969133926,\n 0,\n 0,\n 0.0233384216180084,\n 0.013681143707108373,\n 0.0466768432360168,\n 0,\n 0.03051947442354945,\n 0,\n 0.033062763958845234,\n 0.047610380100737135,\n 0.03051947442354945,\n 0.017250137717658383,\n 0,\n 0.026450211167076187,\n 0.012798489274391704,\n 0.016531381979422617,\n 0.015870126700245714,\n 0.0233384216180084,\n 0.012398536484566963,\n 0.0466768432360168,\n 0.020346316282366297,\n 0.0466768432360168,\n 0,\n 0.026450211167076187,\n 0.04959414593826785,\n 0.03719560945370089,\n 0.06900055087063353,\n 0,\n 0.028339511964724486,\n 0,\n 0.03174025340049143,\n 0.051193957097566814,\n 0,\n 0.024797072969133926,\n 0.03051947442354945,\n 0.03051947442354945,\n 0,\n 0.015259737211774725,\n 0,\n 0,\n 0.03606846977328571,\n 0.041763491316436085,\n 0.07557203190593197,\n 0.0700152648540252,\n 0.07935063350122856,\n 0.0466768432360168,\n 0.054102704659928565,\n 0.03778601595296598,\n 0.04408368527846031,\n 0,\n 0,\n 0,\n 0,\n 0,\n 0.01259533865098866,\n 0.025596978548783407,\n 0.0233384216180084,\n 0.017250137717658383,\n 0.03606846977328571,\n 0,\n 0.07213693954657142,\n 0.024797072969133926,\n 0,\n 0.03174025340049143,\n 0.026450211167076187,\n 0,\n 0,\n 0,\n 0,\n 0.013225105583538094,\n 0,\n 0.01889300797648299,\n 0.01469456175948677,\n 0.028339511964724486,\n 0,\n 0.05667902392944897,\n 0,\n 0.022041842639230154,\n 0.011020921319615077,\n 0.014169755982362243,\n 0.017250137717658383,\n 0,\n 0.01845363569796013,\n 0.0466768432360168,\n 0.022041842639230154,\n 0,\n 0.026450211167076187,\n 0,\n 0.024797072969133926,\n 0.010173158141183148,\n 0.015259737211774725,\n 0,\n 0.01889300797648299,\n 0.01983765837530714,\n 0.06612552791769047,\n 0,\n 0,\n 0.026450211167076187,\n 0.013681143707108373,\n 0.04408368527846031,\n 0.025596978548783407,\n 0,\n 0,\n 0,\n 0.01469456175948677,\n 0.01983765837530714,\n 0,\n 0.03051947442354945,\n 0,\n 0,\n 0,\n 0,\n 0,\n 0.0233384216180084,\n 0.033062763958845234,\n 0,\n 0,\n 0,\n 0,\n 0.026450211167076187,\n 0,\n 0.02938912351897354,\n 0,\n 0,\n 0.04408368527846031,\n 0.03967531675061428,\n 0.012022823257761904,\n 0.034500275435316766,\n 0,\n 0,\n 0,\n 0.04408368527846031,\n 0,\n 0.012398536484566963,\n 0.03400741435766939,\n 0.03967531675061428,\n 0.024797072969133926,\n 0,\n 0,\n 0.07935063350122856,\n 0.03606846977328571,\n 0.033062763958845234,\n 0.03606846977328571,\n 0,\n 0,\n 0.028339511964724486,\n 0,\n 0,\n 0.028339511964724486,\n 0.018034234886642856,\n 0,\n 0.05175041315297515,\n 0.031322618487327064,\n 0.03606846977328571,\n 0.04959414593826785,\n 0.048091293031047616,\n 0,\n 0,\n 0.028339511964724486,\n 0,\n 0.06612552791769047,\n 0.10820540931985713,\n 0,\n 0.026450211167076187,\n 0,\n 0.03606846977328571,\n 0.0233384216180084,\n 0,\n 0,\n 0,\n 0.013681143707108373,\n 0.026450211167076187,\n 0.03051947442354945,\n 0.054102704659928565,\n 0,\n 0.034500275435316766,\n 0.024797072969133926,\n 0.025596978548783407,\n 0,\n 0,\n 0.01889300797648299,\n 0,\n 0.01889300797648299,\n 0.007084877991181121,\n 0.018034234886642856,\n 0.01983765837530714,\n 0.024797072969133926,\n 0,\n 0.013681143707108373,\n 0.022041842639230154,\n 0.04408368527846031,\n 0.03967531675061428,\n 0.016531381979422617,\n 0,\n 0.033062763958845234,\n 0.027051352329964282,\n 0.04408368527846031,\n 0.021254633973543362,\n 0,\n 0,\n 0.04069263256473259,\n 0.01983765837530714,\n 0.01469456175948677,\n 0,\n 0,\n 0,\n 0.015259737211774725,\n 0.020881745658218043,\n 0,\n 0.03174025340049143,\n 0.034500275435316766,\n 0,\n 0.05667902392944897,\n 0.01983765837530714,\n 0.013225105583538094,\n 0.017250137717658383,\n 0.01983765837530714,\n 0.01889300797648299,\n 0.03400741435766939,\n 0,\n 0.03778601595296598,\n 0.0466768432360168,\n 0.024045646515523808,\n 0.07084877991181122,\n 0,\n 0.01983765837530714,\n 0.026450211167076187,\n 0.024797072969133926,\n 0.024045646515523808,\n 0.0610389488470989,\n 0.012798489274391704,\n 0.033062763958845234,\n 0.026450211167076187,\n 0,\n 0.04959414593826785,\n 0.020881745658218043,\n 0.01983765837530714,\n 0.018034234886642856,\n 0.014169755982362243,\n 0.026450211167076187,\n 0,\n 0.026450211167076187,\n 0.0466768432360168,\n 0.0610389488470989,\n 0.0466768432360168,\n 0.028339511964724486,\n 0.010723058581247103,\n 0.06612552791769047,\n 0.028339511964724486,\n 0.022041842639230154,\n 0.016531381979422617,\n 0.012798489274391704,\n 0.033062763958845234,\n 0.012798489274391704,\n 0.05667902392944897,\n 0,\n 0,\n 0,\n 0,\n 0.01469456175948677,\n 0,\n 0,\n 0.01983765837530714,\n 0,\n 0.034500275435316766,\n 0.04408368527846031,\n 0.033062763958845234,\n 0,\n 0.013225105583538094,\n 0.022041842639230154,\n 0.022041842639230154,\n 0,\n 0,\n 0,\n 0.024797072969133926,\n 0.02938912351897354,\n 0,\n 0,\n 0.018034234886642856,\n 0.017250137717658383,\n 0.01469456175948677,\n 0.041763491316436085,\n 0.024797072969133926,\n 0.06348050680098286,\n 0.025596978548783407,\n 0.01469456175948677,\n 0.026450211167076187,\n 0.07213693954657142,\n 0.01469456175948677,\n 0.027051352329964282,\n 0,\n 0,\n 0.015870126700245714,\n 0.03967531675061428,\n 0.034500275435316766,\n 0.04408368527846031,\n 0.02245772646261186,\n 0.042509267947086725,\n 0.015870126700245714,\n 0.052900422334152375,\n 0.033062763958845234,\n 0.024797072969133926,\n 0.028339511964724486,\n 0.047610380100737135,\n 0.01983765837530714,\n 0.026450211167076187,\n 0.022041842639230154,\n 0.07935063350122856,\n 0,\n 0.04408368527846031,\n 0,\n 0.017250137717658383,\n 0.06612552791769047,\n 0.010723058581247103,\n 0,\n 0,\n 0,\n 0,\n 0,\n 0.015259737211774725,\n 0.020881745658218043,\n 0,\n 0.012398536484566963,\n 0.0233384216180084,\n 0,\n 0.05667902392944897,\n 0.07213693954657142,\n 0.08501853589417345,\n 0,\n 0.03778601595296598,\n 0,\n 0.04959414593826785,\n 0.05667902392944897,\n 0,\n 0.03051947442354945,\n 0.011335804785889794,\n 0.03051947442354945,\n 0.041043431121325115,\n 0.061992682422834816,\n 0,\n 0.009446503988241496,\n 0.03839546782317511,\n 0.01983765837530714,\n 0.013225105583538094,\n 0.04408368527846031,\n 0.041763491316436085,\n 0,\n 0.033062763958845234,\n 0.023805190050368567,\n 0.03606846977328571,\n 0,\n 0.0991882918765357,\n 0.05951297512592142,\n 0.051193957097566814,\n 0,\n 0.03839546782317511,\n 0,\n 0,\n 0.034500275435316766,\n ...]"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

In the next notebook, we'll look at a more effective way of calculating word embeddings using more of the idea of context.

## References:

- https://towardsdatascience.com/tfidf-for-piece-of-text-in-python-43feccaa74f8
- https://medium.com/analytics-vidhya/introduction-to-natural-language-processing-part-1-777f972cc7b3
