# README (TOPICS/TOPIC LIST)

## Notebook 1: Linguistic concepts and intro
  - Outline and breakdown of notebooks and how they're structured and what you can find in each
  - Introduction to Syntax, semantics, and pragmatics (definitions, examples of the differences using project gutenburg)
    - include references to English syntax and a brief overview of the key terminologies with regards to English syntax with examples using code to extract them
  - Ambiguity (examples with project gutenburg text) and Compositionality (with code examples of compositionality, again using project gutenburg free texts)
  - Briefly explain how these cause complications for ML with regard to NL
  
#### Exercises:

  1. Summarize the types of ambiguity and give a brief explanation of each.

  2. Give an example of each type of ambiguity not included in this notebook from the text data loaded above.

  3. How does compositionality affect meaning and make natural language processing harder?
  
  4. Label all the parts of speech of the following sentence and identify the noun phrase, predicate, complement, and prepositional phrases along with their objects. ..
"Suzy was determined to eat the lamb chop; her fork and knife were primed and gripped tightly in her hands, after which she dove in and did not look up once from her meal."

  5. Give examples of every part of speech with an excerpt from the text we loaded in this section.


## Notebook 2: Word2Vec + Other basic word embedding models
  - Why do we use word embeddings?
  - Distributed vs Distributional embeddings
  - Walkthrough of getting to Word2Vec and GLoVe embeddings
  - Count-based methods of Word Embeddings
  - simple training walkthrough for word embeddings
  
#### Exercises:
  
    1. For your specific native language, if not English, how would word embeddings work? For symbolic languages (Chinese, Thai, Korean etc.)?
    
    2. What are possible avenues for improving these word embeddings?
    
    3. Is there any way we can reduce training time? Is it possible to paralellize this training of word embeddings?
    
    4. What is the basic purpose of word embeddings?
    

## Notebook 3: Evaluation of word embeddings and Improvements on Word Embeddings with the introduction of sub-words
  - Evaluation metrics
    - intrinsic vs Extrinsic + qualitative vs quantitative
    - word similarity and analogies
  - Logical walkthrough of why the evaluation metrics invalidate/date some of the simpler word embedding concepts
  - SVD, LSA, PCA as a solution to compact the word embeddings and dealing with the sparsity problem
 

#### Exercises:
  
    1. What are the differences and improvements of the word embeddings discussed in this notebook versus the ones discussed in the previous word embedding notebook?
    
     2. What are other evaluation metrics that we can use to understand the quality of word embeddings? How has that influenced the start of sub-word embeddings? How do those differ from normal word embeddings?

## Notebook 4: Sentence level embeddings and data preparation for NLP
  - Data preparation tools
    - tokenization
    - batching
    - lemmatization/stemming
    - word and sentence segmentation
  - BOW + Word2Vec vs TF-IDF + Word2Vec vs CNN + Word2Vec vs RNN+Word2Vec for the task of sentiment analysis
  - complete walkthrough of using all of these different methods for sentiment analysis and understanding sentence level context
  
  
#### Exercises:
  
    1. What are the basic tasks that need to be done for data preparation and how does this differ from other non-NLP ETL?
    
    2. What is a key difference between certain symbolic languages (Chinese, Japanese, Thai etc.) and English when it comes to data preparation and preprocessing?
    
    3. How can we improve this basic sentiment analysis model?
    
    4. What factors have we not considered using this more naive word embedding implementation? How can we improve the word embedding part? 
    
    5. Redo sentiment analysis with one of the models (BOW/CNN/TF-IDF/RNNs) but use a more complex word embedding; how does this affect the results?

## Notebook 5: Language Models
  - Smoothing
  - Char/BPE/Word level embeddings

#### Exercises:
    
    1.
    
    2.
    
    3. 
    
## **Extras**:
Glossary of NLP terms and NLP tasks with brief explanations of each, easily searchable

### **Chapter Exercises**
- What are word embeddings and how do they work?
- How can we simply represent an entire sentence using embeddings?
- What are the differences between syntax, semantics and pragmatics?
- What are subwords and character-based embeddings?
- What's the difference between a distributional and distributed representation?
- What's the difference between extrinsic and intrinsic evlauations?
- Model the steps for accomplishing machine translation, including steps for data preparation and the variety of subtasks needed to train a model.
- What do word similarity metrics and word analogy metrics for word embeddings show us?
- How do ambiguity and compositionality affect NLP?
- What are the different types of ambiguity?
- How can we use compositionality to our advantage?

