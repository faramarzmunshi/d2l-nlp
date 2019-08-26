# README (TOPICS/TOPIC LIST)

**Notebook 0**: Introduction
  - Syntax, semantics, and pragmatics (definitions, examples of the differences using project gutenburg)

**Notebook 1**: Ambiguity and Compositionality
  - Ambiguity (examples with project gutenburg text) and Compositionality (with code examples of compositionality, again using project gutenburg free texts)

**Notebook 2**: Syntax
  - Lemmatization and Stemming with code and code walkthrough
  - Brief introduction to POS tagging, sentence breaking and other syntax tasks with examples using NLTK (using pre-built functions)

**Notebook 3**: Data Preparation
  - Understanding data preparation for NLP
    - what needs to be done prior to a task + simple example of tokenization and word and sentence segmentation
    - mention of terminology extraction/grammar induction/parsing and how they're more difficult tasks

**Notebook 4**: Semantics and Pragmatics
  - Semantics/Pragmatics (struggling to have a good code segment here)
    - NLU, NLG, NLP (chatbot example, talk about intent and entities, and summarize the main semantic tasks of NLP)
    - Talk about how the remaining semantic/pragmatic tasks fit into the aforementioned categories

**Notebook 5**: Word Embeddings
  - Basic word embeddings, CBOW, n-grams, basic statistical properties
  - count based methods, distributional representations
  - training a count-based method and simple distributional representations

**Notebook 6**: Parsing
  - Parsing
    - a worked through example with a combination of use of the aforementioned lemmatization/stemming and other preprocessing steps
    - a basic introduction to both dependency and constituency parsing


**Notebook 7**: Sentiment Analysis
  - Sentiment Analysis Case Study
    - using basic word embeddings based on frequency matrices 
      - Putting together Notebook 4 (data prep), Notebook 3 (Lemmatization and stemming), and notebook 5 (semantics/pragmatics) 

**Notebook 8**: Word Embeddings++ (training and evaluating word embeddings)
  - introduction to semantics/pragmatics and distributional vs distributed word embeddings
  - coverage of basics of mathematics behind dense word embeddings vs sparse
  - coverage of count-based methods vs prediction-based methods
  - mention of subword-based and character-based embeddings
  - Evaluation of Word Embeddings
    - word similarity and analogies

**Notebook 9**: Sentiment Analysis++
  - Using new word embeddings and in-built features, demonstrate how sentiment analysis becomes demonstrably better with a better word representation
  - easier tokenization and preprocessing, and using pre-trained models to streamline the process
  - show how easy it becomes once all the pieces are in place, and contrast to the simple hand-written model in the first case study
  
**Extra**: Glossary of NLP terms and NLP tasks with brief explanations of each, easily searchable
