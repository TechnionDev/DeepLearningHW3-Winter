r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
    start_seq = "You can’t judge a book by its cover"
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
