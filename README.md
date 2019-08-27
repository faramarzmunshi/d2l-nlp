# README (TOPICS/TOPIC LIST)

**Notebook 0**: Introduction
  - Syntax, semantics, and pragmatics (definitions, examples of the differences using project gutenburg)
  - Outline and breakdown of notebooks and how they're structured and what you can find in each
 
**Notebook 1**: Ambiguity and Compositionality
  - Ambiguity (examples with project gutenburg text) and Compositionality (with code examples of compositionality, again using project gutenburg free texts)
  - Briefly explain how these cause complications for ML with regard to NL
  
  Exercises:

    1. Summarize the types of ambiguity and give a brief explanation of each.
    
    2. Give an example of each type of ambiguity not included in this notebook from the text data loaded above.
    
    3. How does compositionality affect meaning and make natural language processing harder?


**Notebook 2**: Syntax
  - Lemmatization and Stemming with code and code walkthrough
  - Brief introduction to POS tagging, sentence breaking and other syntax tasks with examples using NLTK (using pre-built functions)
  
  Exercises:

    1. Code your own tool for sentence breaking using only out of the box Python functions.
    
    2. Briefly describe the difference between Lemmatization and Stemming.

**Notebook 3**: English Syntax and Dissecting English
  - Talk about English POS, give examples from Project Gutenberg using POS tagging
  - Make the section interactive but give examples and understanding of all types of POS and basic structure of sentences
    - interactive partitioning of things like noun phrases, complements, etc.

  Exercises:
  
    1. Label all the parts of speech of the following sentence and identify the noun phrase, predicate, complement, and prepositional phrases along with their objects. ..
    "Suzy was determined to eat the lamb chop; her fork and knife were primed and gripped tightly in her hands, after which she dove in and did not look up once from her meal."
    
    2. Give examples of every part of speech with an excerpt from the text we loaded in this section.
    
    3. How would you begin to break down this syntax and code it into a machine learning model?

**Notebook 4**: Data Preparation
  - Understanding data preparation for NLP
    - what needs to be done prior to a task + simple example of tokenization and word and sentence segmentation
    - mention of terminology extraction/grammar induction/parsing and how they're more difficult tasks
    
    
  Exercises:
  
    1. What is a key difference between certain symbolic languages (Chinese, Japanese, Thai etc.) and English when it comes to data preparation and preprocessing?
    
    2. Code a simple terminology extractor for this notebook. (Hint: all keywords are highlighted/bolded)

```diff
- **Notebook 5**: Semantics and Pragmatics (struggling to have a good code segment here and struggling on how to format this notebook)
  - - Semantics/Pragmatics 
    - - NLU, NLG, NLP (chatbot example, talk about intent and entities, and summarize the main semantic tasks of NLP)
    - - Talk about how the remaining semantic/pragmatic tasks fit into the aforementioned categories
```
  Exercises:
  
    1. 

**Notebook 6**: Word Embeddings
  - Basic word embeddings, CBOW, n-grams, basic statistical properties
  - count based methods, distributional representations
  - training a count-based method and simple distributional representations

  Exercises:
  
    1. 

**Notebook 7**: Parsing
  - Parsing
    - a worked through example with a combination of use of the aforementioned lemmatization/stemming and other preprocessing steps
    - a basic introduction to both dependency and constituency parsing

  Exercises:
  
    1. 

**Notebook 8**: Sentiment Analysis
  - Sentiment Analysis Case Study
    - using basic word embeddings based on frequency matrices 
    - Putting together Notebook 4 (data prep), Notebook 3 (Lemmatization and stemming), and notebook 5 (semantics/pragmatics) 

  Exercises:
  
    1. 

**Notebook 9**: Word Embeddings++ (training and evaluating word embeddings)
  - introduction to semantics/pragmatics and distributional vs distributed word embeddings
  - coverage of basics of mathematics behind dense word embeddings vs sparse
  - coverage of count-based methods vs prediction-based methods
  - mention of subword-based and character-based embeddings
  - Evaluation of Word Embeddings
    - word similarity and analogies
    - extrinsic vs intrinsic evaluation

  Exercises:
  
    1. 

**Notebook 10**: Sentiment Analysis++
  - Using new word embeddings and in-built features, demonstrate how sentiment analysis becomes demonstrably better with a better word representation
  - easier tokenization and preprocessing, and using pre-trained models to streamline the process
  - show how easy it becomes once all the pieces are in place, and contrast to the simple hand-written model in the first case study

  Exercises:

    1. 
  
  
**Extra**: Glossary of NLP terms and NLP tasks with brief explanations of each, easily searchable
