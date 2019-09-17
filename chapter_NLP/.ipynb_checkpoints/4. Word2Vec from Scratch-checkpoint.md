# Variations of Co-occurrence Matrix

Let’s say there are V unique words in the corpus. So Vocabulary size = V. The columns of the Co-occurrence matrix form the context words. The different variations of Co-Occurrence Matrix are-

A co-occurrence matrix of size V X V. Now, for even a decent corpus V gets very large and difficult to handle. So generally, this architecture is never preferred in practice.
A co-occurrence matrix of size V X N where N is a subset of V and can be obtained by removing irrelevant words like stopwords etc. for example. This is still very large and presents computational difficulties.
But, remember this co-occurrence matrix is not the word vector representation that is generally used. Instead, this Co-occurrence matrix is decomposed using techniques like PCA, SVD etc. into factors and combination of these factors forms the word vector representation.

Let me illustrate this more clearly. For example, you perform PCA on the above matrix of size VXV. You will obtain V principal components. You can choose k components out of these V components. So, the new matrix will be of the form V X k.

And, a single word, instead of being represented in V dimensions will be represented in k dimensions while still capturing almost the same semantic meaning. k is generally of the order of hundreds.

So, what PCA does at the back is decompose Co-Occurrence matrix into three matrices, U,S and V where U and V are both orthogonal matrices. What is of importance is that dot product of U and S gives the word vector representation and V gives the word context representation.



 

## Advantages of Co-occurrence Matrix

It preserves the semantic relationship between words. i.e man and woman tend to be closer than man and apple.
It uses SVD at its core, which produces more accurate word vector representations than existing methods.
It uses factorization which is a well-defined problem and can be efficiently solved.
It has to be computed once and can be used anytime once computed. In this sense, it is faster in comparison to others.
 

## Disadvantages of Co-Occurrence Matrix

It requires huge memory to store the co-occurrence matrix.
But, this problem can be circumvented by factorizing the matrix out of the system for example in Hadoop clusters etc. and can be saved.
 

2.2 Prediction based Vector
Pre-requisite: This section assumes that you have a working knowledge of how a neural network works and the mechanisms by which weights in an NN are updated. If you are new to Neural Network, I would suggest you go through this awesome article by Sunil to gain a very good understanding of how NN works.

So far, we have seen deterministic methods to determine word vectors. But these methods proved to be limited in their word representations until Mitolov etc. el introduced word2vec to the NLP community. These methods were prediction based in the sense that they provided probabilities to the words and proved to be state of the art for tasks like word analogies and word similarities. They were also able to achieve tasks like King -man +woman = Queen, which was considered a result almost magical. So let us look at the word2vec model used as of today to generate word vectors.

Word2vec is not a single algorithm but a combination of two techniques – CBOW(Continuous bag of words) and Skip-gram model. Both of these are shallow neural networks which map word(s) to the target variable which is also a word(s). Both of these techniques learn weights which act as word vector representations. Let us discuss both these methods separately and gain intuition into their working.

 

# 2.2.1 CBOW (Continuous Bag of words)
The way CBOW work is that it tends to predict the probability of a word given a context. A context may be a single word or a group of words. But for simplicity, I will take a single context word and try to predict a single target word.

Suppose, we have a corpus C = “Hey, this is sample corpus using only one context word.” and we have defined a context window of 1. This corpus may be converted into a training set for a CBOW model as follow. The input is shown below. The matrix on the right in the below image contains the one-hot encoded from of the input on the left.



The target for a single datapoint say Datapoint 4 is shown as below

Hey	this	is	sample	corpus	using	only	one	context	word
0	0	0	1	0	0	0	0	0	0
 

This matrix shown in the above image is sent into a shallow neural network with three layers: an input layer, a hidden layer and an output layer. The output layer is a softmax layer which is used to sum the probabilities obtained in the output layer to 1. Now let us see how the forward propagation will work to calculate the hidden layer activation.

Let us first see a diagrammatic representation of the CBOW model.



The matrix representation of the above image for a single data point is below.



The flow is as follows:

The input layer and the target, both are one- hot encoded of size [1 X V]. Here V=10 in the above example.
There are two sets of weights. one is between the input and the hidden layer and second between hidden and output layer.
Input-Hidden layer matrix size =[V X N] , hidden-Output layer matrix  size =[N X V] : Where N is the number of dimensions we choose to represent our word in. It is arbitary and a hyper-parameter for a Neural Network. Also, N is the number of neurons in the hidden layer. Here, N=4.
There is a no activation function between any layers.( More specifically, I am referring to linear activation)
The input is multiplied by the input-hidden weights and called hidden activation. It is simply the corresponding row in the input-hidden matrix copied.
The hidden input gets multiplied by hidden- output weights and output is calculated.
Error between output and target is calculated and propagated back to re-adjust the weights.
The weight  between the hidden layer and the output layer is taken as the word vector representation of the word.
We saw the above steps for a single context word. Now, what about if we have multiple context words? The image below describes the architecture for multiple context words.



Below is a matrix representation of the above architecture for an easy understanding.



The image above takes 3 context words and predicts the probability of a target word. The input can be assumed as taking three one-hot encoded vectors in the input layer as shown above in red, blue and green.

So, the input layer will have 3 [1 X V] Vectors in the input as shown above and 1 [1 X V] in the output layer. Rest of the architecture is same as for a 1-context CBOW.

The steps remain the same, only the calculation of hidden activation changes. Instead of just copying the corresponding rows of the input-hidden weight matrix to the hidden layer, an average is taken over all the corresponding rows of the matrix. We can understand this with the above figure. The average vector calculated becomes the hidden activation. So, if we have three context words for a single target word, we will have three initial hidden activations which are then averaged element-wise to obtain the final activation.

In both a single context word and multiple context word, I have shown the images till the calculation of the hidden activations since this is the part where CBOW differs from a simple MLP network. The steps after the calculation of hidden layer are same as that of the MLP as mentioned in this article – Understanding and Coding Neural Networks from scratch.

The differences between MLP and CBOW are  mentioned below for clarification:

The objective function in MLP is a MSE(mean square error) whereas in CBOW it is negative log likelihood of a word given a set of context i.e -log(p(wo/wi)), where p(wo/wi) is given as


wo : output word
wi: context words

2. The gradient of error with respect to hidden-output weights and input-hidden weights are different since MLP has  sigmoid activations(generally) but CBOW has linear activations. The method however to calculate the gradient is same as an MLP.

 

## Advantages of CBOW:

Being probabilistic is nature, it is supposed to perform superior to deterministic methods(generally).
It is low on memory. It does not need to have huge RAM requirements like that of co-occurrence matrix where it needs to store three huge matrices.
 

## Disadvantages of CBOW:

CBOW takes the average of the context of a word (as seen above in calculation of hidden activation). For example, Apple can be both a fruit and a company but CBOW takes an average of both the contexts and places it in between a cluster for fruits and companies.
Training a CBOW from scratch can take forever if not properly optimized.
 

# Skip – Gram model
Skip – gram follows the same topology as of CBOW. It just flips CBOW’s architecture on its head. The aim of skip-gram is to predict the context given a word. Let us take the same corpus that we built our CBOW model on. C=”Hey, this is sample corpus using only one context word.” Let us construct the training data.



The input vector for skip-gram is going to be similar to a 1-context CBOW model. Also, the calculations up to hidden layer activations are going to be the same. The difference will be in the target variable. Since we have defined a context window of 1 on both the sides, there will be “two” one hot encoded target variables and “two” corresponding outputs as can be seen by the blue section in the image.

Two separate errors are calculated with respect to the two target variables and the two error vectors obtained are added element-wise to obtain a final error vector which is propagated back to update the weights.

The weights between the input and the hidden layer are taken as the word vector representation after training. The loss function or the objective is of the same type as of the CBOW model.

The skip-gram architecture is shown below.



 

For a better understanding, matrix style structure with calculation has been shown below.



 

Let us break down the above image.

Input layer  size – [1 X V], Input hidden weight matrix size – [V X N], Number of neurons in hidden layer – N, Hidden-Output weight matrix size – [N X V], Output layer size – C [1 X V]

In the above example, C is the number of context words=2, V= 10, N=4

The row in red is the hidden activation corresponding to the input one-hot encoded vector. It is basically the corresponding row of input-hidden matrix copied.
The yellow matrix is the weight between the hidden layer and the output layer.
The blue matrix is obtained by the matrix multiplication of hidden activation and the hidden output weights. There will be two rows calculated for two target(context) words.
Each row of the blue matrix is converted into its softmax probabilities individually as shown in the green box.
The grey matrix contains the one hot encoded vectors of the two context words(target).
Error is calculated by substracting the first row of the grey matrix(target) from the first row of the green matrix(output) element-wise. This is repeated for the next row. Therefore, for n target context words, we will have n error vectors.
Element-wise sum is taken over all the error vectors to obtain a final error vector.
This error vector is propagated back to update the weights.
Advantages of Skip-Gram Model
Skip-gram model can capture two semantics for a single word. i.e it will have two vector representations of Apple. One for the company and other for the fruit.
Skip-gram with negative sub-sampling outperforms every other method generally.
 

This is an excellent interactive tool to visualise CBOW and skip gram in action. I would suggest you to really go through this link for a better understanding.

### Building the Neural Network

Word2vec trains a shallow neural network over data as structured using either Continuous Bag of Words or Skip Gram architecture. Instead of leveraging the model for predictive purposes, we use the hidden weights from the neural network to generate the word vectors.
Assuming a Continuous Bag of Words architecture with a fixed context window of 1 word, this is what the process would look like. First, the corpus.

> I like math
> I like programming
> Today is Friday
> Today is a good day

To make things even easier, we can require our context window to only include words which proceeds the target. We can assume that the context of words at the end of a sentence is the first word of the next sentence. Under such rules:

- like is the context of target I
- math is the context of target like
- programming is also the context of target like

Even with such a simple corpus, we can begin to recognize some patterns. “Math” and “programming” are both context to “like”. While this might not be picked up by the model, both of these words can be understood as things that I like.

#### Step 1
The first step is to one hot encode our classes like we did above with the 'I like turtles' example (the words in our vocabulary): I, like, math, programming, today, is, Friday, a, good, day

#### Step 2
Create a feed forward neural network with one hidden layer and an output layer using the softmax activation function. The data set used to train the network uses the one hot encoded context vector to predict the one hot encoded target vector.
The number of neurons in the hidden layer corresponds to the number of dimensions in the final word vectors.

#### Step 3
Obtain the weights of the hidden network. Each row in the weight matrix corresponds to the vector of each word in the vocabulary.

Realistically, this is not something that we do very often. Good word2vec models require a very large corpus in the billions of words. Fortunately, pre-trained models are easy to use and find. You can download the word2vec model trained over the 100 billion word Google News corpus on their website, or you can use GluonNLP to load a set of pre-trained word embedding.

Here, we'll show you how to create the model and train it, but, in the end, will use pre-built word embeddings that have been independently verified for accuracy for testing and understanding. 

```{.python .input}

```
