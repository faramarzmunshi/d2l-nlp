## Setup

Let's first start by setting up our environment so we can use the textual corpora from Project Gutenberg! Because Python 3 removed the berkeley-db4 package from native Python, we have to reinstall it using homebrew. This also applies to the gzip package that made unzipping large files using python easier. If you don't have homebrew or you are not using a Mac, you can use "apt-get" on Linux OS.

```{.python .input}
!brew install berkeley-db4
!brew install zlib
!pip3 install gutenberg
```

Now that we have the required packages, we can play a bit with the Gutenberg corpora and take a look at some core concepts related to NLP.

# An Introduction to NLP and Core Linguistic Concepts

## Ambiguity

Firstly, let's cover the topic of ambiguity, and in this case, the specifics of *syntactic and lexical ambiguity*. **Ambiguity** can be referred to as the ability of having more than one meaning or being understood in more than one way. Natural languages are ambiguous, so computers are not able to understand language the way people do. Natural Language Processing (NLP) is concerned with the development of computational models of aspects of human language processing. 

Ambiguity can occur at various levels of NLP. Ambiguity could be **Lexical** (word-level), **Syntactic** (dealing with order of words), **Semantic** (dealing with meaning of words), **Pragmatic** (dealing with contextual meanings) etc. 

Lexicon and lexical simply refers to words or word-type. Syntax (other forms: syntactic) refers to the literal order of words and what can be derived from the order of these words. Semantics is dealing simply with the meaning of words without context, while pragmatics deals with meaning related to **context** and the derivation of meaning from context.

The terms and definitions we use in this introductory section are interwoven with linguistics basics. The relevant and important terms that will be repeated and referenced throughout this chapter are bolded. The italicized words and phrases are important, but not absolutely necessary to understand the rest of the content in this book.

Let's take a quick look at an example of *lexical ambiguity*. We are loading a **corpus** (plural: corpora/corpuses), or a body of text(s), from Project Gutenberg, and using the text to extract the context of a certain word we're looking up in the corpus. 

This function, `give_me_context`, is defined here to allow us to quickly and easily see the context for any given word given a corpus and the context window (the number of words on each side of the word) that we would like to see.

```{.python .input  n=2}
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from pprint import pprint

text = strip_headers(load_etext(2701)).strip()
print("Corpus name: \n{0}\n".format(text[0:46]))

print("Length of the loaded corpus: {0} words".format(len(text.split(" "))))

def give_me_context(text, word, context_window):
    text = text.casefold()
    space = " "
    count = 1
    li = []
    
    for word_counter, i in enumerate(text.split()):
        if i.casefold() == word.casefold():
            li.append("{0}. {1}\n".format(str(count),
                                                      space.join(
                                                          text.split()[word_counter-context_window:
                                                                       word_counter+context_window])))
            count+=1
                     
    if count==1:
        print("Couldn't find \"{0}\" in the corpus.".format(word.lower()))
    return li
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Corpus name: \nMOBY-DICK;\n\nor, THE WHALE.\n\nBy Herman Melville\n\nLength of the loaded corpus: 194853 words\n"
 }
]
```

Here we can see the length of the corpus is quite large. We're going to be using Moby-dick by Herman Melville to demonstrate some of the aforementioned ambiguities. 

Let's use the word "learned" in this example. Learned can either be the past tense of learn, or it can simply mean knowledgable. Let's see if we can decipher the two meanings given context.

```{.python .input  n=3}
l = give_me_context(text, "learned", 15)
pprint(l[0:7])
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "['1. \u2014_sir william davenant. preface to gondibert_. \u201cwhat spermacetti is, men '\n 'might justly doubt, since the learned hosmannus in his work of thirty years, '\n 'saith plainly, _nescio quid sit_.\u201d \u2014_sir t.\\n',\n '2. tarshish could have been no other city than the modern cadiz. that\u2019s the '\n 'opinion of learned men. and where is cadiz, shipmates? cadiz is in spain; as '\n 'far by water,\\n',\n '3. take sich dangerous weepons in their rooms at night. so, mr. queequeg\u201d '\n '(for she had learned his name), \u201ci will just take this here iron, and keep '\n 'it for you\\n',\n '4. say, that though the captain is very discreet and scientific here, yet, '\n 'for all his learned \u201cbinnacle deviations,\u201d \u201cazimuth compass observations,\u201d '\n 'and \u201capproximate errors,\u201d he knows very well, captain sleet,\\n',\n '5. though, upon the whole, i greatly admire and even love the brave, the '\n 'honest, and learned captain; yet i take it very ill of him that he should so '\n 'utterly\\n',\n '6. mystical, sympathetical feeling was in me; ahab\u2019s quenchless feud seemed '\n 'mine. with greedy ears i learned the history of that murderous monster '\n 'against whom i and all the others had\\n',\n '7. this glorious thing is utterly unknown to men ashore! never! but some '\n 'time after, i learned that goney was some seaman\u2019s name for albatross. so '\n 'that by no possibility could\\n']\n"
 }
]
```

Here we can clearly see the difference between these two words in example 5 and example 6. In example 5, the word "learned" describes Captain, referring to a knowledgeable Captain. In example 6, it refers to the narrator learning of the history of the "murderous monster," meaning it uses the past tense verb form of the word "learn." This is a clear example of lexical, or word-level, ambiguity. These two words are considered *homographs* or words that are spelled the same but have different meanings. This basically means, on a word by word basis, just looking at the single word without context, we cannot decipher the word **sense**, or meaning, of the given word. 

There are also other types of ambiguity as we have mentioned above, like syntactic ambiguity. This refers to the presence of two or more possible meanings within a single sentence or sequence of words; this can also be referred to as structural or grammatical ambiguity. A quick example from our corpus is demonstrated below with the word "mole":

```{.python .input  n=4}
l = give_me_context(text, "mole", 3)
print(l[0])
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "1. where that noble mole is washed\n\n"
 }
]
```

Looking at the first occurrence of the word "mole," in Moby Dick, we see that with limited context, the phrase "where that noble mole is washed" seems quite strange, and could be referring to a number of different things. 

Firstly, it could actually mean where an aristocratic mole, the animal, is bathed and washed, which in and of itself is quite strange (yet pleasing to think about). Or it could be referring to, as I know from previous knowledge reading the book, the city in which this entire narrative takes place, using a metaphor to refer to the city's downtown area that's been battered and shaped by the sea. Without further context, we cannot confirm or deny either of these suspicions. This is what we refer to as *syntactic ambiguity*.

The meaning of the sentence depends on an understanding of the context and the speaker's intent. As defined in linguistics, a sentence is an abstract entity—-a string of words divorced from non-linguistic context--in contrast to an **utterance**, which is a concrete example of a speech act in a specific context. The more closely conscious subjects stick to common words, idioms, phrasings, and topics, the more easily others can surmise their meaning; simultaneously, the further they stray from common expressions and topics, the wider the variations in interpretations.

Even with some context, there's ambiguity as to the meaning of the phrase/sentence.

Let's look further at the rest of the sentence regarding the "mole" for context and clarify what the meaning here of "mole" from the text actually is, and if our wish of having a royal mole being bathed will be satisfied by the classical text of Moby Dick:

```{.python .input  n=5}
l = give_me_context(text, "mole", 9)
pprint(l[0])
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "('1. its extreme downtown is the battery, where that noble mole is washed by '\n 'waves, and cooled by breezes,\\n')\n"
 }
]
```

Once we are presented with further context, the meaning becomes clearer. Sometimes context can resolve ambiguity, but the more limited the context window, like our first window of 3, the harder it becomes to derive meaning and resolve the issue of ambiguity.

But there are other times where not even context can help. In those situations, you have to have a previous knowledge base or understanding of the situation for you, or the machine, to understand the different meanings a sentence, word, or phrase. This is one of the largest problems for NLP; the existence of a lack of clear understanding of phrases and sentences without prior, previous, or contextual knowledge.

This also suggests that sentences do not have an intrinsic meaning, that there is no meaning associated with a sentence or word, and that either only represent an idea symbolically. "The dog sat on the carpet" is a sentence in English. If someone were to say to someone else, "The dog sat on the carpet," the act is itself an utterance. This implies that a sentence, term, expression or word cannot symbolically represent a single true meaning; such meaning is underspecified (which dog sat on which carpet?) and consequently, is potentially ambiguous. By contrast, the meaning of an utterance can be inferred through knowledge of its context, or **pragmatics**, leveraging both its linguistic and non-linguistic contexts (which may or may not be sufficient to resolve ambiguity). You may factually know from previous conversations it's your neighbour's dog, or the carpet referred to is the Persian carpet you keep in your bedroom, or even that the dog didn't actually sit normally, because the neighbour's dog has a hip problem and causes him to lay more than sit, yet the closest word to describe what the dog did is to use sit. All of these are ambiguous from the sentence "The dog sat on the carpet," and having a limited context can allow the sentence to become more and more ambiguous and answer less and less questions.

It's almost like having a vector in which an element is 3, but that 3 could sometimes be 4, 5, 6, 7, and sometimes 14, but you don't necessarily have the information to make that determination. This lack of knowledge, and the lack of determination makes ambiguity such a large issue in NLP and fundamentally a harder problem than some other parts of Deep Learning. 

## Compositionality

Compositionality is the other beast that presents itself as a basic roadblock in the field of NLP. It is highly related to ambiguity and the previous section. 

The requirements for a language to be **compositional** are the following:
    - The meaning of the sentence doesn't directly depend on
        - things said earlier in the conversation
        - the beliefs or intentions of the person uttering the sentence
        - objects that are salient or events in the environment when the sentence is spoken/written
        - the non-semantic character of the sentences simple parts like their shape or sound

Clearly this is almost impossible for any sort of natural language; languages like Python, though, are strictly compositional, if we consider lines of code to be sentences. But no natural language has this; the presence of puns, conversation topics about things in or around a room, references to older conversations or in-jokes, or even the belief and intention of the person saying the sentence, are all regular participants in a conversation or piece of text. So why is this such a big deal?

This issue rears its head as machines and some of the current models rely heavily on a language being compositional to be effective. For a model to derive meaning, the only access it has is to the words in the text it's been given, or a previous knowledge base it's been given. When looking at a small portion of text, the better the knowledge base, the better the understanding the model will have of the text. But in most situations, this knowledge graph and knowledge base is hard to construct or entirely lacking. Instead, a model largely has to depend on context and semantic relationships to derive meaning for words. Basically, the principle, also known as Frege's principle or the principle of compositionality, is the theory that "the meaning of a \[sentence or utterance\] is determined by the meanings of its constituent expressions and the rules used to combine them." Many NLP algorithms rely heavily on this principle and derive the "meaning" or their representations based upon the prior statement. 

Let's look at a sentence from the first passage of the second chapter of Moby Dick:

```{.python .input  n=6}
l = give_me_context(text, "tucked", 10)
pprint(l[0])
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "('1. i stuffed a shirt or two into my old carpet-bag, tucked it under my arm, '\n 'and started for cape horn\\n')\n"
 }
]
```

The entirety of the sentence's meaning is only based on the meaning of the constituents. There is no ambiguity, the language here is entirely compositional, specifically referring to the narrator putting one or two shirts in his bag and starting his journey to Cape Horn. There are some cases in which natural languages are strictly compositional. But then, take for example, the following:

```{.python .input  n=7}
l = give_me_context(text, "curbstone", 40)
pprint(l[0][177:])
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "('poor lazarus there, chattering his teeth against the curbstone for his '\n 'pillow, and shaking off his tatters with his shiverings, he might plug up '\n 'both ears with rags, and put a corn-cob into his mouth, and yet that would '\n 'not keep out the tempestuous euroclydon. euroclydon! says old\\n')\n"
 }
]
```

The sentence here references Lazarus, the disciple and miracle of Jesus Christ, and Euroclydon, a tempestuous northeast wind which blew in the Mediterranean in certain seasons. Without prior knowledge or a knowledge base, an algorithm would not have possibly known of these two terms, or been able to understand from context the meaning of this sentence. The principle of compositionality here, fails for the English language. 

And this is exactly why NLP is such a difficult sub-field of deep learning: the presence of ambiguity and the failing of compositionality in metaphorical and referential natural language. 

The natural next question then, is how do we attempt to create numeric representations of words, sentences, and documents on which we can train models keeping these two issues in mind? 

We start answering exactly that on a word level in the next notebook.
