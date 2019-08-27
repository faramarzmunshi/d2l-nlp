# README (TOPICS/TOPIC LIST)

## **Notebook 0**: Introduction
  - Syntax, semantics, and pragmatics (definitions, examples of the differences using project gutenburg)
  - Outline and breakdown of notebooks and how they're structured and what you can find in each
 
## **Notebook 1**: Ambiguity and Compositionality
  - Ambiguity (examples with project gutenburg text) and Compositionality (with code examples of compositionality, again using project gutenburg free texts)
  - Briefly explain how these cause complications for ML with regard to NL
  
#### Exercises:

    1. Summarize the types of ambiguity and give a brief explanation of each.
    
    2. Give an example of each type of ambiguity not included in this notebook from the text data loaded above.
    
    3. How does compositionality affect meaning and make natural language processing harder?


## **Notebook 2**: Syntax
  - Lemmatization and Stemming with code and code walkthrough
  - Brief introduction to POS tagging, sentence breaking and other syntax tasks with examples using NLTK (using pre-built functions)
  
#### Exercises:

    1. Code your own tool for sentence breaking using only out of the box Python functions.
    
    2. Briefly describe the difference between Lemmatization and Stemming.

## **Notebook 3**: English Syntax and Dissecting English
  - Talk about English POS, give examples from Project Gutenberg using POS tagging
  - Make the section interactive but give examples and understanding of all types of POS and basic structure of sentences
    - interactive partitioning of things like noun phrases, complements, etc.

#### Exercises:
  
    1. Label all the parts of speech of the following sentence and identify the noun phrase, predicate, complement, and prepositional phrases along with their objects. ..
    "Suzy was determined to eat the lamb chop; her fork and knife were primed and gripped tightly in her hands, after which she dove in and did not look up once from her meal."
    
    2. Give examples of every part of speech with an excerpt from the text we loaded in this section.
    
    3. How would you begin to break down this syntax and code it into a machine learning model?

## **Notebook 4**: Data Preparation
  - Understanding data preparation for NLP
    - what needs to be done prior to a task + simple example of tokenization and word and sentence segmentation
    - mention of terminology extraction/grammar induction/parsing and how they're more difficult tasks
    
    
#### Exercises:
  
    1. What is a key difference between certain symbolic languages (Chinese, Japanese, Thai etc.) and English when it comes to data preparation and preprocessing?
    
    2. Code a simple terminology extractor for this notebook. (Hint: all keywords are highlighted/bolded)

```diff
!--------------------------------------------
```
## **Notebook 5**: Semantics and Pragmatics (struggling to have a good code segment here and struggling on how to format this notebook)
```diff
!--------------------------------------------
```
  - Semantics/Pragmatics 
    - NLU, NLG, NLP (chatbot example, talk about intent and entities, and summarize the main semantic tasks of NLP)
    - Talk about how the remaining semantic/pragmatic tasks fit into the aforementioned categories

#### Exercises:
  
    1. 

## **Notebook 6**: Word Embeddings
  - Basic word embeddings, CBOW, n-grams, basic statistical properties
  - count based methods, distributional representations
  - training a count-based method and simple distributional representations

#### Exercises:
  
    1. What are possible avenues for improving these word embeddings?
    2. Is there any way we can make training time lower? Is it possible to paralellize this training?
    3. What is the basic purpose of word embeddings?

## **Notebook 7**: Parsing
  - Parsing
    - a worked through example of building a simple parser with a combination of use of the aforementioned lemmatization/stemming and other preprocessing steps
    - a basic introduction to both dependency and constituency parsing

#### Exercises:
  
    1. Describe the difference between dependency and constituency parsing.
    2. Play with both the constituency parser and dependency parser demos here and here to understand the basics even better: 
          - https://demo.allennlp.org/dependency-parsing
          - https://demo.allennlp.org/constituency-parsing

## **Notebook 8**: Sentiment Analysis
  - Sentiment Analysis Case Study
    - using basic word embeddings based on frequency matrices 
    - using only out of the box python to implement this basic model

#### Exercises:
  
    1. How can we improve this basic sentiment analysis model?
    2. What factors have we not considered using this more naive word embedding implementation? How can we improve the word embedding part?
    3. What aspects have we neglected? What fundamental ideas are missing from this implementation?

## **Notebook 9**: Word Embeddings++ (training and evaluating word embeddings)
  - introduction to semantics/pragmatics and distributional vs distributed word embeddings
  - coverage of basics of mathematics behind dense word embeddings vs sparse
  - coverage of count-based methods vs prediction-based methods
  - mention of subword-based and character-based embeddings
  - Evaluation of Word Embeddings
    - word similarity and analogies
    - extrinsic vs intrinsic evaluation

#### Exercises:
  
    1. What are the differences and improvements of the word embeddings discussed in this notebook versus the ones discussed in the previous word embedding notebook?
    2. For your specific native language, if not English, how would word embeddings work? For symbolic languages (Chinese, Thai, Korean etc.)?
    3. What are other evaluation metrics that we can use to understand the quality of word embeddings? 

## **Notebook 10**: Sentiment Analysis++
  - Using new word embeddings and in-built features, demonstrate how sentiment analysis becomes demonstrably better with a better word representation
  - easier tokenization and preprocessing, and using pre-trained models to streamline the process
  - show how easy it becomes once all the pieces are in place, and contrast to the simple hand-written model in the first case study

#### Exercises:

    1. Explain how this sentiment analysis model is demonstrably better.
    2. What steps were we able to skip with better word representations?
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

