r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. input X:
A. the shape of the tensor would be 64\*512\*64\*1024 B. we can represent a fully connected layer as Y=XW.
 So we would expect our dim's to be Y_{64\*512} = X_{64\*1024}W_{1024\*512}. So the Jacobin would end pu to be as described. 
 
B. as we know etch element (i,j) of the output matrix Y is equal to the sum from k=1 up to k=1024 of X_{i,k}\*W_{k,j}.
so the partial derivative of y_{i,j} with respect to x is equal to a matrix were only one row has non-zeros values = a sparce matrix. 
this is happening for each combination of {i,j}. so we eventually get a 4D tensor of many sparse matrices. 

C. we dont need to calculate the downstream gradient w.r.t the input dX.
 we saw in the lecture that we can represent the partial derivative as a sum of:
 dX = (dY/dX)\*(dL/dY) = (dY/dX)\*dY = sum_{k=1 to 64} [(dY_{k}/dX)\*dY_{k}] = (dY)W^T.
2. input W:
A .as before the dim would be 64\*512\*1024\*512. 

B .as part 1, but now we will get a matrix dY_{i,j}/dW, were only one column has non-zero values. so the over all matrix 
would still be a sparse one.

C. same answer as part 1, the equation will now be : 
dW = (dL/dY)*(dY/dX) = dY*(dY/dX) = sum_{k=1 to 64} [dY_{k}*(dY_{k}/dX)] = X^T*(dY).

"""

part1_q2 = r"""
**Your answer:**
backpropagation is not requires for the training of a gradient descent optimizer. 
we are using backpropagation as a tool for improving the performance of our calculations. 
we can calculate the numerical gradient of the entire network 
which is significantly slower - but results a same output as backprop.

"""
# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 1
    lr = 0.07
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.05
    lr_momentum = 0.005
    lr_rmsprop = 0.0002
    reg = 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 100
    #wstd=0.1
    lr = 0.005
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. the results that we can see in the graphs are somewhat matching our initial exception of what we would get from the 
experiment - although the results were more 'noise' then expected and because of that our conclusion is somewhat questionable.
with no dropouts (=0) we can see an increase in overfitting  on the train data with no clear increase in the accuracy of the test. 
this is as expected!
with dropout of 0.4 we can see that the accuracy is higher then the rest of the the configurations which is what we expected.
because of the dropout of part of the neurons we are able to mitigate the overfitting affect.
with dropout of 0.8, most of the neurons are dropt-out and we expected to have an randoms affect that we could absorve in the graph.
but due to the noise of the results  we can't really see this clearly  

2. as explained above the high dropout setting is dropping most of the neurons and it make for a more random behaviour - we can see that the test accuray is most of the time below the 0.4 dropout.
in the 0.4 dropout the results are better, we are meneging the oevrfitting affect in a better way. 
 

"""

part2_q2 = r"""
the answer is that its possible. Cross-Entropy Loss is penalizing the model over wrong predictions. 
so because of that incorrect prediction have more lasting effect on the loss value than correct predictions.
were there are wrong predictions over one class, and the rest of the predictions of the class are correct
, it is possible for the test loss to increase while the accuracy also increases. 
"""


part2_q3 = r"""
**Your answer:**
1. backpropagation is a tool for efficiently calculating the gradients of the model using the chain-rule - 
it doesn't optimises anything by itself.
 on the over hand, GD/SGD is a tool for solving optimization problems - by using the gradient of the model.
 
2. SGD and GD are both methods of solving an optimization problem using the gradients of the model. in GD we are passing 
over all of the samples before updating the prameters. were in SGD we are passing over some batch of samples (or even one sample at a time)
and then we are updating the paramters with respect to the loss of that singal batch. the main idea is that in SGD we
should be able to converge faster - as we changing our parmetes w.r.t a small sample eatch time - 
and thus "exploring the world in a better way". we would expect to get an overall worse approx. error with SGD.

3. in sgd we are expecting to overcome local minimum and saddle points - as each time we twick the parameters w.r.t a diff batch sample.
another reason is that because SGD is converging faster we prefer it when dealing with large Data sets. 

4. A. our DL model uses non-linear functions. as we know, given function f(x1+x2) != f(x1)+f(x2). and because of that proprety, 
using this suggestion(splitting the data to batches and than adding the loss) would result in a different results then GD.
 B. in each forward pass we still would need to store the result of that forward pass. so the accumulation of the total loss would be possible. 
 so its possible that in some point we will loss all free memory.

"""
# ==============

# ==============
# Part 3 answers


def part3_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 100
    hypers['seq_len'] = 64
    hypers['h_dim'] = 128
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.1
    hypers['learn_rate'] = 0.0001
    hypers['lr_sched_factor'] = 0.0001
    hypers['lr_sched_patience'] = 0.0001
    # ========================
    return hypers


def part3_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "You cant judge a book by its cover"
    temperature = 0.5
    # ========================
    return start_seq, temperature


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
