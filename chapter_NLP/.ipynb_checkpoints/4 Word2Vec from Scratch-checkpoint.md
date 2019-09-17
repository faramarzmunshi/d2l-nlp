# Word2Vec from Scratch

Recall content of the last section.  The core feature of the skip-gram model is the use of softmax operations to compute the conditional probability of generating context word $w_o$ based on the given central target word $w_c$.

$$\mathbb{P}(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}.$$

The logarithmic loss corresponding to the conditional probability is given as

$$-\log \mathbb{P}(w_o \mid w_c) =
-\mathbf{u}_o^\top \mathbf{v}_c + \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$


Because the softmax operation has considered that the context word could be any word in the dictionary $\mathcal{V}$, the loss mentioned above actually includes the sum of the number of items in the dictionary size. From the last section, we know that for both the skip-gram model and CBOW model, because they both get the conditional probability using a softmax operation, the gradient computation for each step contains the sum of the number of items in the dictionary size. For larger dictionaries with hundreds of thousands or even millions of words, the overhead for computing each gradient may be too high.  In order to reduce such computational complexity, we will introduce two approximate training methods in this section: negative sampling and hierarchical softmax. Since there is no major difference between the skip-gram model and the CBOW model, we will only use the skip-gram model as an example to introduce these two training methods in this section.



## Negative Sampling

Negative sampling modifies the original objective function. Given a context window for the central target word $w_c$, we will treat it as an event for context word $w_o$ to appear in the context window and compute the probability of this event from

$$\mathbb{P}(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

Here, the $\sigma$ function has the same definition as the sigmoid activation function:

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$

We will first consider training the word vector by maximizing the joint probability of all events in the text sequence. Given a text sequence of length $T$, we assume that the word at time step $t$ is $w^{(t)}$ and the context window size is $m$. Now we consider maximizing the joint probability

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} \mathbb{P}(D=1\mid w^{(t)}, w^{(t+j)}).$$

However, the events included in the model only consider positive examples. In this case, only when all the word vectors are equal and their values approach infinity can the joint probability above be maximized to 1. Obviously, such word vectors are meaningless. Negative sampling makes the objective function more meaningful by sampling with an addition of negative examples. Assume that event $P$ occurs when context word $w_o$ appears in the context window of central target word $w_c$, and we sample $K$ words that do not appear in the context window according to the distribution $\mathbb{P}(w)$ to act as noise words. We assume the event for noise word $w_k$($k=1, \ldots, K$) to not appear in the context window of central target word $w_c$ is $N_k$. Suppose that events $P$ and $N_1, \ldots, N_K$ for both positive and negative examples are independent of each other. By considering negative sampling, we can rewrite the joint probability above, which only considers the positive examples, as


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} \mathbb{P}(w^{(t+j)} \mid w^{(t)}),$$

Here, the conditional probability is approximated to be
$$ \mathbb{P}(w^{(t+j)} \mid w^{(t)}) =\mathbb{P}(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim \mathbb{P}(w)}^K \mathbb{P}(D=0\mid w^{(t)}, w_k).$$


Let the text sequence index of word $w^{(t)}$ at time step $t$ be $i_t$ and $h_k$ for noise word $w_k$ in the dictionary. The logarithmic loss for the conditional probability above is

$$
\begin{aligned}
-\log\mathbb{P}(w^{(t+j)} \mid w^{(t)})
=& -\log\mathbb{P}(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim \mathbb{P}(w)}^K \log\mathbb{P}(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim \mathbb{P}(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim \mathbb{P}(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$

Here, the gradient computation in each step of the training is no longer related to the dictionary size, but linearly related to $K$. When $K$ takes a smaller constant, the negative sampling has a lower computational overhead for each step.


## Hierarchical Softmax

Hierarchical softmax is another type of approximate training method. It uses a binary tree for data structure, with the leaf nodes of the tree representing every word in the dictionary $\mathcal{V}$.

![Hierarchical Softmax. Each leaf node of the tree represents a word in the dictionary. ](../img/hi-softmax.svg)


We assume that $L(w)$ is the number of nodes on the path (including the root and leaf nodes) from the root node of the binary tree to the leaf node of word $w$. Let $n(w,j)$ be the $j$th node on this path, with the context word vector $\mathbf{u}_{n(w,j)}$. We use Figure 12.3 as an example, so $L(w_3) = 4$. Hierarchical softmax will approximate the conditional probability in the skip-gram model as

$$\mathbb{P}(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o,j)) ]\!] \cdot \mathbf{u}_{n(w_o,j)}^\top \mathbf{v}_c\right),$$

Here the $\sigma$ function has the same definition as the sigmoid activation function, and $\text{leftChild}(n)$ is the left child node of node $n$. If $x$ is true, $[\![x]\!] = 1$; otherwise $[\![x]\!] = -1$.
Now, we will compute the conditional probability of generating word $w_3$ based on the given word $w_c$ in Figure 12.3. We need to find the inner product of word vector $\mathbf{v}_c$ (for word $w_c$) and each non-leaf node vector on the path from the root node to $w_3$. Because, in the binary tree, the path from the root node to leaf node $w_3$ needs to be traversed left, right, and left again (the path with the bold line in Figure 12.3), we get

$$\mathbb{P}(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3,1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3,2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3,3)}^\top \mathbf{v}_c).$$

Because $\sigma(x)+\sigma(-x) = 1$, the condition that the sum of the conditional probability of any word generated based on the given central target word $w_c$ in dictionary $\mathcal{V}$ be 1 will also suffice:

$$\sum_{w \in \mathcal{V}} \mathbb{P}(w \mid w_c) = 1.$$

In addition, because the order of magnitude for $L(w_o)-1$ is $\mathcal{O}(\text{log}_2|\mathcal{V}|)$, when the size of dictionary $\mathcal{V}$ is large, the computational overhead for each step in the hierarchical softmax training is greatly reduced compared to situations where we do not use approximate training.

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

In this section, we will train a skip-gram model defined in the last notebook.
First, import the packages and modules required for the experiment, and load the PTB data set.

```{.python .input  n=1}
import d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(512, 5, 5)
```

## The Skip-Gram Model

We will implement the skip-gram model by using embedding layers and mini-batch multiplication. These methods are also often used to implement other natural language processing applications.

### Embedding Layer

The layer in which the obtained word is embedded is called the embedding layer, which can be obtained by creating an `nn.Embedding` instance in Gluon. The weight of the embedding layer is a matrix whose number of rows is the dictionary size (`input_dim`) and whose number of columns is the dimension of each word vector (`output_dim`). We set the dictionary size to 20 and the word vector dimension to 4.

```{.python .input  n=2}
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "Parameter embedding0_weight (shape=(20, 4), dtype=float32)"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

The input of the embedding layer is the index of the word. When we enter the index $i$ of a word, the embedding layer returns the $i$th row of the weight matrix as its word vector. Below we enter an index of shape (2,3) into the embedding layer. Because the dimension of the word vector is 4, we obtain a word vector of shape (2,3,4).

```{.python .input  n=4}
x = nd.array([[1, 2, 3], [4, 5, 6]])
embed(x)
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "\n[[[ 0.01438687  0.05011239  0.00628365  0.04861524]\n  [-0.01068833  0.01729892  0.02042518 -0.01618656]\n  [-0.00873779 -0.02834515  0.05484822 -0.06206018]]\n\n [[ 0.06491279 -0.03182812 -0.01631819 -0.00312688]\n  [ 0.0408415   0.04370362  0.00404529 -0.0028032 ]\n  [ 0.00952624 -0.01501013  0.05958354  0.04705103]]]\n<NDArray 2x3x4 @cpu(0)>"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Mini-batch Multiplication

We can multiply the matrices in two mini-batches one by one, by the mini-batch multiplication operation `batch_dot`. Suppose the first batch contains $n$ matrices $\boldsymbol{X}_1, \ldots, \boldsymbol{X}_n$ with a shape of $a\times b$, and the second batch contains $n$ matrices $\boldsymbol{Y}_1, \ldots, \boldsymbol{Y}_n$ with a shape of $b\times c$. The output of matrix multiplication on these two batches are $n$ matrices $\boldsymbol{X}_1\boldsymbol{Y}_1, \ldots, \boldsymbol{X}_n\boldsymbol{Y}_n$ with a shape of $a\times c$. Therefore, given two NDArrays of shape ($n$, $a$, $b$) and ($n$, $b$, $c$), the shape of the mini-batch multiplication output is ($n$, $a$, $c$).

```{.python .input  n=5}
X = nd.ones((2, 1, 4))
Y = nd.ones((2, 4, 6))
nd.batch_dot(X, Y).shape
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "(2, 1, 6)"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Skip-gram Model Forward Calculation

In forward calculation, the input of the skip-gram model contains the central target word index `center` and the concatenated context and noise word index `contexts_and_negatives`. In which, the `center` variable has the shape (batch size, 1), while the `contexts_and_negatives` variable has the shape (batch size, `max_len`). These two variables are first transformed from word indexes to word vectors by the word embedding layer, and then the output of shape (batch size, 1, `max_len`) is obtained by mini-batch multiplication. Each element in the output is the inner product of the central target word vector and the context word vector or noise word vector.

```{.python .input  n=6}
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

Verify that the output shape should be (batch size, 1, `max_len`).

```{.python .input  n=7}
skip_gram(nd.ones((2,1)), nd.ones((2,4)), embed, embed).shape
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "(2, 1, 4)"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Training

Before training the word embedding model, we need to define the loss function of the model.

### Binary Cross Entropy Loss Function

According to the definition of the loss function in negative sampling, we can directly use Gluon's binary cross-entropy loss function `SigmoidBinaryCrossEntropyLoss`.

```{.python .input  n=8}
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
```

It is worth mentioning that we can use the mask variable to specify the partial predicted value and label that participate in loss function calculation in the mini-batch: when the mask is 1, the predicted value and label of the corresponding position will participate in the calculation of the loss function; When the mask is 0, the predicted value and label of the corresponding position do not participate in the calculation of the loss function. As we mentioned earlier, mask variables can be used to avoid the effect of padding on loss function calculations.

Given two identical examples, different masks lead to different loss values.

```{.python .input  n=9}
pred = nd.array([[.5]*4]*2)
label = nd.array([[1,0,1,0]]*2)
mask = nd.array([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask)
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "\n[0.724077  0.3620385]\n<NDArray 2 @cpu(0)>"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Initialize Model Parameters

We construct the embedding layers of the central and context words, respectively, and set the hyper-parameter word vector dimension `embed_size` to 100.

```{.python .input  n=10}
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

### Training

The training function is defined below. Because of the existence of padding, the calculation of the loss function is slightly different compared to the previous training functions.

```{.python .input  n=11}
def train(net, data_iter, lr, num_epochs, ctx=d2l.try_gpu()):
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, num_tokens
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_context(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum().asscalar(), l.size)
            if (i+1) % 50 == 0:
                animator.add(epoch+(i+1)/len(data_iter), metric[0]/metric[1])
    print('loss %.3f, %d tokens/sec on %s ' % (
        metric[0]/metric[1], metric[1]/timer.stop(), ctx))
```

Now, we can train a skip-gram model using negative sampling.

```{.python .input  n=12}
lr, num_epochs = 0.01, 5
train(net, data_iter, lr, num_epochs)
```

```{.json .output n=12}
[
 {
  "ename": "MXNetError",
  "evalue": "[15:20:55] src/ndarray/ndarray.cc:1285: GPU is not enabled\nStack trace:\n  [bt] (0) 1   libmxnet.so                         0x0000000119806d29 mxnet::op::NDArrayOpProp::~NDArrayOpProp() + 4473\n  [bt] (1) 2   libmxnet.so                         0x000000011ae6d02f mxnet::CopyFromTo(mxnet::NDArray const&, mxnet::NDArray const&, int, bool) + 4671\n  [bt] (2) 3   libmxnet.so                         0x000000011ad8fd99 mxnet::imperative::PushFComputeEx(std::__1::function<void (nnvm::NodeAttrs const&, mxnet::OpContext const&, std::__1::vector<mxnet::NDArray, std::__1::allocator<mxnet::NDArray> > const&, std::__1::vector<mxnet::OpReqType, std::__1::allocator<mxnet::OpReqType> > const&, std::__1::vector<mxnet::NDArray, std::__1::allocator<mxnet::NDArray> > const&)> const&, nnvm::Op const*, nnvm::NodeAttrs const&, mxnet::Context const&, std::__1::vector<mxnet::engine::Var*, std::__1::allocator<mxnet::engine::Var*> > const&, std::__1::vector<mxnet::engine::Var*, std::__1::allocator<mxnet::engine::Var*> > const&, std::__1::vector<mxnet::Resource, std::__1::allocator<mxnet::Resource> > const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::OpReqType, std::__1::allocator<mxnet::OpReqType> > const&)::'lambda'(mxnet::RunContext)::operator()(mxnet::RunContext) const + 217\n  [bt] (3) 4   libmxnet.so                         0x000000011ad7e74e mxnet::imperative::PushFComputeEx(std::__1::function<void (nnvm::NodeAttrs const&, mxnet::OpContext const&, std::__1::vector<mxnet::NDArray, std::__1::allocator<mxnet::NDArray> > const&, std::__1::vector<mxnet::OpReqType, std::__1::allocator<mxnet::OpReqType> > const&, std::__1::vector<mxnet::NDArray, std::__1::allocator<mxnet::NDArray> > const&)> const&, nnvm::Op const*, nnvm::NodeAttrs const&, mxnet::Context const&, std::__1::vector<mxnet::engine::Var*, std::__1::allocator<mxnet::engine::Var*> > const&, std::__1::vector<mxnet::engine::Var*, std::__1::allocator<mxnet::engine::Var*> > const&, std::__1::vector<mxnet::Resource, std::__1::allocator<mxnet::Resource> > const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::OpReqType, std::__1::allocator<mxnet::OpReqType> > const&) + 1230\n  [bt] (4) 5   libmxnet.so                         0x000000011ad7d05a mxnet::Imperative::InvokeOp(mxnet::Context const&, nnvm::NodeAttrs const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::OpReqType, std::__1::allocator<mxnet::OpReqType> > const&, mxnet::DispatchMode, mxnet::OpStatePtr) + 810\n  [bt] (5) 6   libmxnet.so                         0x000000011ad81991 mxnet::Imperative::Invoke(mxnet::Context const&, nnvm::NodeAttrs const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&) + 817\n  [bt] (6) 7   libmxnet.so                         0x000000011acc73ae SetNDInputsOutputs(nnvm::Op const*, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> >*, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> >*, int, void* const*, int*, int, int, void***) + 1582\n  [bt] (7) 8   libmxnet.so                         0x000000011acc80f0 MXImperativeInvokeEx + 176\n  [bt] (8) 9   _ctypes.cpython-37m-darwin.so       0x000000010fc9236f ffi_call_unix64 + 79\n\n",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mMXNetError\u001b[0m                                Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-12-c04f85570b58>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0mlr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_epochs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m0.01\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m5\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mtrain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnet\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata_iter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_epochs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;32m<ipython-input-11-88131a5051b8>\u001b[0m in \u001b[0;36mtrain\u001b[0;34m(net, data_iter, lr, num_epochs, ctx)\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0mtrain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnet\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata_iter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_epochs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0md2l\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtry_gpu\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m     \u001b[0mnet\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0minitialize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mctx\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mctx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mforce_reinit\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m     trainer = gluon.Trainer(net.collect_params(), 'adam',\n\u001b[1;32m      4\u001b[0m                             {'learning_rate': lr})\n\u001b[1;32m      5\u001b[0m     animator = d2l.Animator(xlabel='epoch', ylabel='loss',\n",
   "\u001b[0;32m/usr/local/lib/python3.7/site-packages/mxnet/gluon/block.py\u001b[0m in \u001b[0;36minitialize\u001b[0;34m(self, init, ctx, verbose, force_reinit)\u001b[0m\n\u001b[1;32m    503\u001b[0m             \u001b[0mWhether\u001b[0m \u001b[0mto\u001b[0m \u001b[0mforce\u001b[0m \u001b[0mre\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0minitialization\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mparameter\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0malready\u001b[0m \u001b[0minitialized\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    504\u001b[0m         \"\"\"\n\u001b[0;32m--> 505\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcollect_params\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0minitialize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minit\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mverbose\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mforce_reinit\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    506\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    507\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mhybridize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mactive\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/local/lib/python3.7/site-packages/mxnet/gluon/parameter.py\u001b[0m in \u001b[0;36minitialize\u001b[0;34m(self, init, ctx, verbose, force_reinit)\u001b[0m\n\u001b[1;32m    828\u001b[0m             \u001b[0minit\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_verbosity\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mverbose\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mverbose\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    829\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0m_\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mv\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 830\u001b[0;31m             \u001b[0mv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0minitialize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minit\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mforce_reinit\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mforce_reinit\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    831\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    832\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mzero_grad\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/local/lib/python3.7/site-packages/mxnet/gluon/parameter.py\u001b[0m in \u001b[0;36minitialize\u001b[0;34m(self, init, ctx, default_init, force_reinit)\u001b[0m\n\u001b[1;32m    406\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    407\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_deferred_init\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0minit\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdefault_init\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 408\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_finish_deferred_init\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    409\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    410\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mreset_ctx\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/local/lib/python3.7/site-packages/mxnet/gluon/parameter.py\u001b[0m in \u001b[0;36m_finish_deferred_init\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    300\u001b[0m                     initializer.InitDesc(self.name, {'__init__': init}), data)\n\u001b[1;32m    301\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 302\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_init_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    303\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    304\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_init_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx_list\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/local/lib/python3.7/site-packages/mxnet/gluon/parameter.py\u001b[0m in \u001b[0;36m_init_impl\u001b[0;34m(self, data, ctx_list)\u001b[0m\n\u001b[1;32m    312\u001b[0m             \u001b[0mdev_list\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mctx\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdevice_id\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    313\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 314\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_data\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcopyto\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mctx\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mctx\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_ctx_list\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    315\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_init_grad\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    316\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/local/lib/python3.7/site-packages/mxnet/gluon/parameter.py\u001b[0m in \u001b[0;36m<listcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m    312\u001b[0m             \u001b[0mdev_list\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mctx\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdevice_id\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    313\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 314\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_data\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcopyto\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mctx\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mctx\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_ctx_list\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    315\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_init_grad\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    316\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/local/lib/python3.7/site-packages/mxnet/ndarray/ndarray.py\u001b[0m in \u001b[0;36mcopyto\u001b[0;34m(self, other)\u001b[0m\n\u001b[1;32m   2091\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mother\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mContext\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2092\u001b[0m             \u001b[0mhret\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mNDArray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m_new_alloc_handle\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mother\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2093\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0m_internal\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_copyto\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mout\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mhret\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2094\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2095\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mTypeError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'copyto does not support type '\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mother\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/local/lib/python3.7/site-packages/mxnet/ndarray/register.py\u001b[0m in \u001b[0;36m_copyto\u001b[0;34m(data, out, name, **kwargs)\u001b[0m\n",
   "\u001b[0;32m/usr/local/lib/python3.7/site-packages/mxnet/_ctypes/ndarray.py\u001b[0m in \u001b[0;36m_imperative_invoke\u001b[0;34m(handle, ndargs, keys, vals, out)\u001b[0m\n\u001b[1;32m     90\u001b[0m         \u001b[0mc_str_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkeys\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     91\u001b[0m         \u001b[0mc_str_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ms\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0ms\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mvals\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 92\u001b[0;31m         ctypes.byref(out_stypes)))\n\u001b[0m\u001b[1;32m     93\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     94\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0moriginal_output\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/local/lib/python3.7/site-packages/mxnet/base.py\u001b[0m in \u001b[0;36mcheck_call\u001b[0;34m(ret)\u001b[0m\n\u001b[1;32m    251\u001b[0m     \"\"\"\n\u001b[1;32m    252\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mret\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 253\u001b[0;31m         \u001b[0;32mraise\u001b[0m \u001b[0mMXNetError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpy_str\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m_LIB\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mMXGetLastError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    254\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    255\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mMXNetError\u001b[0m: [15:20:55] src/ndarray/ndarray.cc:1285: GPU is not enabled\nStack trace:\n  [bt] (0) 1   libmxnet.so                         0x0000000119806d29 mxnet::op::NDArrayOpProp::~NDArrayOpProp() + 4473\n  [bt] (1) 2   libmxnet.so                         0x000000011ae6d02f mxnet::CopyFromTo(mxnet::NDArray const&, mxnet::NDArray const&, int, bool) + 4671\n  [bt] (2) 3   libmxnet.so                         0x000000011ad8fd99 mxnet::imperative::PushFComputeEx(std::__1::function<void (nnvm::NodeAttrs const&, mxnet::OpContext const&, std::__1::vector<mxnet::NDArray, std::__1::allocator<mxnet::NDArray> > const&, std::__1::vector<mxnet::OpReqType, std::__1::allocator<mxnet::OpReqType> > const&, std::__1::vector<mxnet::NDArray, std::__1::allocator<mxnet::NDArray> > const&)> const&, nnvm::Op const*, nnvm::NodeAttrs const&, mxnet::Context const&, std::__1::vector<mxnet::engine::Var*, std::__1::allocator<mxnet::engine::Var*> > const&, std::__1::vector<mxnet::engine::Var*, std::__1::allocator<mxnet::engine::Var*> > const&, std::__1::vector<mxnet::Resource, std::__1::allocator<mxnet::Resource> > const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::OpReqType, std::__1::allocator<mxnet::OpReqType> > const&)::'lambda'(mxnet::RunContext)::operator()(mxnet::RunContext) const + 217\n  [bt] (3) 4   libmxnet.so                         0x000000011ad7e74e mxnet::imperative::PushFComputeEx(std::__1::function<void (nnvm::NodeAttrs const&, mxnet::OpContext const&, std::__1::vector<mxnet::NDArray, std::__1::allocator<mxnet::NDArray> > const&, std::__1::vector<mxnet::OpReqType, std::__1::allocator<mxnet::OpReqType> > const&, std::__1::vector<mxnet::NDArray, std::__1::allocator<mxnet::NDArray> > const&)> const&, nnvm::Op const*, nnvm::NodeAttrs const&, mxnet::Context const&, std::__1::vector<mxnet::engine::Var*, std::__1::allocator<mxnet::engine::Var*> > const&, std::__1::vector<mxnet::engine::Var*, std::__1::allocator<mxnet::engine::Var*> > const&, std::__1::vector<mxnet::Resource, std::__1::allocator<mxnet::Resource> > const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::OpReqType, std::__1::allocator<mxnet::OpReqType> > const&) + 1230\n  [bt] (4) 5   libmxnet.so                         0x000000011ad7d05a mxnet::Imperative::InvokeOp(mxnet::Context const&, nnvm::NodeAttrs const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::OpReqType, std::__1::allocator<mxnet::OpReqType> > const&, mxnet::DispatchMode, mxnet::OpStatePtr) + 810\n  [bt] (5) 6   libmxnet.so                         0x000000011ad81991 mxnet::Imperative::Invoke(mxnet::Context const&, nnvm::NodeAttrs const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> > const&) + 817\n  [bt] (6) 7   libmxnet.so                         0x000000011acc73ae SetNDInputsOutputs(nnvm::Op const*, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> >*, std::__1::vector<mxnet::NDArray*, std::__1::allocator<mxnet::NDArray*> >*, int, void* const*, int*, int, int, void***) + 1582\n  [bt] (7) 8   libmxnet.so                         0x000000011acc80f0 MXImperativeInvokeEx + 176\n  [bt] (8) 9   _ctypes.cpython-37m-darwin.so       0x000000010fc9236f ffi_call_unix64 + 79\n\n"
  ]
 }
]
```

## Applying the Word Embedding Model

After training the word embedding model, we can represent similarity in meaning between words based on the cosine similarity of two word vectors. As we can see, when using the trained word embedding model, the words closest in meaning to the word "chip" are mostly related to chips.

```{.python .input  n=14}
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability.
    cos = nd.dot(W, x) / (nd.sum(W * W, axis=1) * nd.sum(x * x) + 1e-9).sqrt()
    topk = nd.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print('cosine sim=%.3f: %s' % (cos[i].asscalar(), (vocab.idx_to_token[i])))

get_similar_tokens('chip', 3, net[0])
```

```{.json .output n=14}
[
 {
  "ename": "RuntimeError",
  "evalue": "Parameter 'embedding1_weight' has not been initialized. Note that you should initialize parameters and create Trainer with Block.collect_params() instead of Block.params because the later does not include Parameters of nested child Blocks",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-14-4d3eed2289c9>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      8\u001b[0m         \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'cosine sim=%.3f: %s'\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mcos\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masscalar\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mvocab\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0midx_to_token\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      9\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 10\u001b[0;31m \u001b[0mget_similar_tokens\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'chip'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m3\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnet\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;32m<ipython-input-14-4d3eed2289c9>\u001b[0m in \u001b[0;36mget_similar_tokens\u001b[0;34m(query_token, k, embed)\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0mget_similar_tokens\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mquery_token\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0membed\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m     \u001b[0mW\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0membed\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mweight\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m     \u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mW\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mvocab\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mquery_token\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m     \u001b[0;31m# Compute the cosine similarity. Add 1e-9 for numerical stability.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m     \u001b[0mcos\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mW\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m/\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mnd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msum\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mW\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mW\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mnd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msum\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0;36m1e-9\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msqrt\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/local/lib/python3.7/site-packages/mxnet/gluon/parameter.py\u001b[0m in \u001b[0;36mdata\u001b[0;34m(self, ctx)\u001b[0m\n\u001b[1;32m    509\u001b[0m                                \u001b[0;34m\"because its storage type is %s. Please use row_sparse_data() \"\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m\\\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    510\u001b[0m                                \"instead.\" % (self.name, str(ctx), self._stype))\n\u001b[0;32m--> 511\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_check_and_get\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_data\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    512\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    513\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mlist_data\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/local/lib/python3.7/site-packages/mxnet/gluon/parameter.py\u001b[0m in \u001b[0;36m_check_and_get\u001b[0;34m(self, arr_list, ctx)\u001b[0m\n\u001b[1;32m    226\u001b[0m             \u001b[0;34m\"with Block.collect_params() instead of Block.params \"\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m\\\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    227\u001b[0m             \u001b[0;34m\"because the later does not include Parameters of \"\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m\\\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 228\u001b[0;31m             \"nested child Blocks\"%(self.name))\n\u001b[0m\u001b[1;32m    229\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    230\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_get_row_sparse\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0marr_list\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrow_id\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mRuntimeError\u001b[0m: Parameter 'embedding1_weight' has not been initialized. Note that you should initialize parameters and create Trainer with Block.collect_params() instead of Block.params because the later does not include Parameters of nested child Blocks"
  ]
 }
]
```

```{.python .input}

```
