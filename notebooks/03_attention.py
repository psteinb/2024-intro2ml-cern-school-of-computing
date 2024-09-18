# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     argv:
#     - python
#     - -m
#     - ipykernel_launcher
#     - -f
#     - '{connection_file}'
#     display_name: Python 3 (ipykernel)
#     env: null
#     interrupt_mode: signal
#     language: python
#     metadata:
#       debugger: true
#     name: python3
# ---

# %% [markdown]
"""
# Exploring a reweighting task

The following material is taken from the excellent lectures of Francois Fleuret (Univeristy of Geneva). Francois Fleuret introduces the self-attention mechanism and the transformer architecure in three parts:

- [first stream](https://fleuret.org/dlc/streaming/dlc-video-13-1-attention-memory-translation.mp4),
- [second stream](https://fleuret.org/dlc/streaming/dlc-video-13-2-attention-mechanisms.mp4) and
- [third](https://fleuret.org/dlc/streaming/dlc-video-13-3-transformers.mp4)

all of which are worthwhile watching. For more details, checkout the [website](https://fleuret.org/dlc/) of his lecture series.

I have asked permission of Francois to reuse some this material which was published under the Public Domain.
"""

# %% [markdown]
r"""
## A regression task

In the following, we again look at a regression task. The functions below produce a dataset which can be used to illustrate the use of the self-attention mechanism. The dataset exhibits two triangles and two boxes/rectangles on a 1D line. The dataset purely serves didactical purposes and only has loose connections to language processing.

In this notebook, we focus on

1. exploring the data set with a standard convnet.
2. preparing a network using the attention mechanism
3. compare the performance of the two
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from fleuret_data import generate_sequences, save_sequence_images

drng = np.random.default_rng(43)  # set the RNG seed for reproducible runs


# %%
# create train and test set
seq_length=64
train_input, train_targets, train_tr, train_bx = generate_sequences(
    12000, seq_length=seq_length, rng=drng
)
test_input, test_targets, test_tr, test_bx = generate_sequences(
    1000, seq_length=seq_length, rng=drng
)

# %%
test_input.shape, test_targets.shape, test_tr.shape, test_bx.shape

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

ax[0].plot(
    np.arange(test_input[0].shape[-1]) + 0.5,
    test_input[0].squeeze(),
    color="blue",
    label="input",
)
ax[0].set_title("input")
ax[1].plot(
    np.arange(test_targets[0].shape[-1]) + 0.5,
    test_targets[0].squeeze(),
    color="green",
    label="target",
)
ax[1].set_title("target")
fig.savefig("attention_dataset.svg")

# %% [markdown]
"""
You see two kinds of "objects" in the plots above: two box-like structures and two triangle-like structure. We now define a **regression task** which is meant to equalize the height of the boxes. The new height of the output boxes should be the mean height of the two input boxes, The height of the output triangles should be the mean of the two input triangles).
"""

# %% [markdown]
r"""
## Convolutional Network

First, we need to normalize the data into a dynamic range as $\vec{x} \in [0,1]$. Then, we create a regression model using convolutions only, which tries to accomplish the task above.

### Data Normalisation
"""

# %%
import torch
# set the seeds to make the notebook reproducible
np.random.seed(41)
torch.random.manual_seed(43)

# %%
# normalize the signal, zscale normalisation commented out for experimentation
x_min, x_max = train_input.min(), train_input.max()
loc = x_min  # np.mean(train_input)
scale = x_max - x_min  # np.std(train_input)
x_ = (train_input - loc) / scale

y_min, y_max = train_targets.min(), train_targets.max()
y_ = (train_targets - loc) / scale

x_test_ = (test_input - loc) / scale
y_test_ = (test_targets - loc) / scale


# %%
print(f"data shape check: {x_.shape, y_.shape, x_test_.shape, y_test_.shape}")

# %% [markdown]
"""
### A fully convolutional model

The term fully convolutional describes an architecture which consists exclusively of convolutional operations. This has the benefit as the model design is independent of the input data shape.

** Exercise 03.1 **

The code for the fully convolutional model is almost ready, but some minor mistakes and blanks have been introduced. Check if you can fix these errors so that a model of the following form would be printed without errors.

```
> print(RegressionFCN((1,seq_length), num_channels=64))
RegressionFCN(
  (layers): Sequential(
    (0): Conv1d(1, 64, kernel_size=(5,), stride=(1,), padding=(2,))
    (1): ReLU()
    (2): Conv1d(64, 64, kernel_size=(5,), stride=(1,), padding=(2,))
    (3): ReLU()
    (4): Conv1d(64, 64, kernel_size=(5,), stride=(1,), padding=(2,))
    (5): ReLU()
    (6): Conv1d(64, 64, kernel_size=(5,), stride=(1,), padding=(2,))
    (7): ReLU()
    (8): Conv1d(64, 1, kernel_size=(1,), stride=(1,))
  )
)
```
"""

# %%
class RegressionFCN(torch.nn.ModuleList):

    def __init__(self, inshape, num_channels=64, ksize=5, num_layers=4):
        ...

        self.layers = torch.nn.Sequential()
        num_inchannels = inshape[0]
        padding = ...
        for _ in range(8):
            self.layers.append(
                torch.nn.Conv1d(num_inchannels,
                                num_channels,
                                ksize,
                                stride=...,
                                padding=padding )
            )
            self.layers.append(
                ...
            )
            num_inchannels = num_channels
        self.layers.append(torch.nn.Conv1d(num_channels, inshape[0], 1))

    def forward(self, x):

        return self.layers(x)


# %% jupyter={"source_hidden": true}
# Solution for Exercise 03.1
class RegressionFCN(torch.nn.Module):

    def __init__(self, inshape, num_channels=64, ksize=5, num_layers=4):
        super().__init__()

        self.layers = torch.nn.Sequential()
        num_inchannels = inshape[0]
        padding = ksize // 2
        for _ in range(num_layers):
            self.layers.append(
                torch.nn.Conv1d(num_inchannels, num_channels, ksize,stride=1,padding=padding )
            )
            self.layers.append(
                torch.nn.ReLU()
            )
            num_inchannels = num_channels
        self.layers.append(torch.nn.Conv1d(num_channels, inshape[0], 1))

    def forward(self, x):

        return self.layers(x)


# %%
# prepare the dataset and loaders

train_ds = torch.utils.data.StackDataset(x_, y_)
test_ds = torch.utils.data.StackDataset(x_test_, y_test_)

train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=256)
test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=256)

def nparams(model):
    """ compute the number of parameters of this <model> """
    return sum(p.numel() for p in model.parameters())

# %%
# we test our model

plainfcn = RegressionFCN(x_.shape[-2:], num_channels=64)
print(plainfcn)
first_x, first_y = next(iter(train_loader))
output = plainfcn(first_x)
assert output.shape == first_y.shape
print(f"set up plain fcn with {nparams(plainfcn)} parameters")

# %%
# perform training
def train_regression(model, opt, crit, train_dataloader, test_dataloader, max_epochs, log_every=5):

    results = {"train_losses": [], "test_losses": []}
    ntrainsteps = len(train_dataloader)
    nteststeps = len(test_dataloader)
    train_loss, test_loss = torch.zeros((ntrainsteps,)), torch.zeros((nteststeps,))

    for epoch in range(max_epochs):
        model.train()
        # perform training for one epoch
        for idx, (X, y) in enumerate(train_dataloader):
            # forward pass
            y_hat = model(X)

            # compute loss
            loss = crit(y_hat, y)

            # compute gradient
            loss.backward()

            # apply weight update rule
            opt.step()

            # set gradients to 0
            opt.zero_grad()

            train_loss[idx] = loss.item()


        model.eval()

        with torch.no_grad():
            for idx, (X_test, y_test) in enumerate(test_dataloader):
                X_prime_test = model(X_test)
                loss_ = crit(X_prime_test, X_test)
                test_loss[idx] = loss_.item()

            results["train_losses"].append(train_loss.mean())
            results["test_losses"].append(test_loss.mean())

            if epoch % log_every == 0 or (epoch + 1) == max_epochs:
                print(
                    f"{epoch+1:02.0f}/{max_epochs} :: training loss {train_loss.mean():03.4f}; test loss {test_loss.mean():03.4f}"
                )
    return results

# %%
optim = torch.optim.AdamW(plainfcn.parameters(), lr=1e-3)
crit  = torch.nn.MSELoss()
max_epochs = 15
fcnresults = train_regression(plainfcn, optim, crit, train_loader, test_loader, max_epochs,2)

# %%
# let's make our lives a bit easier
def plot_history(history, metrics=["train_losses", "test_losses"], metric_label="metric", draw_legend=True):
    """
    Plot the training history

    Args:
        history (keras History object that is returned by model.fit())
        metrics(str, list): Metric or a list of metrics to plot
    """

    f, ax = plt.subplots(1,1)
    for k in history.keys():
        ax.plot(history[k], label=k)
    ax.set_xlabel("epochs")
    ax.set_ylabel(metric_label)
    if draw_legend:
        ax.legend()

    return f, ax

# %%
f,ax = plot_history(fcnresults)
f.savefig("attention_fcnmodel_losses.svg")


# # %%
# let's visualise some example predictions
xaxis = np.arange(0, x_test_.shape[-1], 1)
first5_x_test, first5_y_test = next(iter(test_loader))
pred5 = plainfcn(first5_x_test[:5,...]) # predict only first 5 samples

f, ax = plt.subplots(1,5, figsize=(10,2), sharex=True, sharey=True)

for col in range(5):
    labl = first5_y_test[col:col+1].detach().squeeze().numpy()
    pred = pred5[col:col+1].detach().squeeze().numpy()
    ax[col].plot(labl, color="green", label="label")
    ax[col].plot(pred, color="red", label="prediction")
    ax[col].set_title(f"test set item {col}")
    ax[col].set_ylabel("intensity / a.u.")
    ax[col].set_xlabel("sample / a.u.")
    if col == 4:
        ax[col].legend()

f.savefig("attention_plainfcn_pred5.svg")

# %% [markdown]
r"""
The above is not a great model, actually it doesn't really well! But we expected no less as the test loss didn't decrease any further than `0.0085` while the training loss decreased further and further.
"""

# %% [markdown]
"""
# Attention is all you need

Attention and self-attention are very powerful transformations. In this section, we will write our own Attention layer.

The idea of attention was published in 2014 by A. Graves in "Neural Turing Machines", see [here](https://arxiv.org/abs/1410.5401)
It was picked up again in 2017 by A. Vaswani et al in "Attention is all you need", see [here](https://arxiv.org/abs/1706.03762). This paper coined the term Transformer architecture which relies strongly on self-attention layers. This architecture proved very successful in the language processing domain. Today, all large language models rely on the attention mechanism.

A nice visualizer to help you grasp the idea of attention can be found [here](https://poloclub.github.io/transformer-explainer/).

Even though, we will not do any language processing here, the dataset and task at hand is enough to demonstrate the effectiveness, by which attention can capture long-range dependencies in the data.
"""

# %% [markdown]
r"""
## Attention

The attention mechanism relies on three ingredients. But in our example task, we only have the incoming feature maps from a convolution. So we will model a triple bifurcation of the incoming data.

We first create queries, keys and value tensors, but applying a convolution on our input `x`. In practice, this is often done with linear layers, but for the sake of argument and processing speed (on weaker hardware) we pick convolutions.

$$ \mathbf{Q} = conv(\vec{x}), \mathbf{K} = conv(\vec{x}), \mathbf{V} = conv(\vec{x}) $$

We want to construct the convolutions in such a fashion, that $\mathbf{Q}\in\mathbb{R}^{TxD}$ where $T$ is an arbitrary dimension we can set and $D$ is the length of our sequence.

We further want $\mathbf{K}\in\mathbb{R}^{T'xD}$ where $T'$ is an arbitrary dimension we can set and $D$ is again the length of our sequence.

Last, we also want $\mathbf{V}\in\mathbb{R}^{T'xD'}$ where $T'$ is the same value as for $\mathbf{K}$ and $D'$ is for the sake of the example equal to $D$.

Here is a sketch of what we are trying to achieve:

<div style="display: block;margin-left: auto;margin-right: auto;width: 50%;"><img src="img/attention_sketch_qkv_cropped.png" alt="Establishing Q, K ,V"></div>
"""


# %% [markdown]
r"""
** Exercise 03.2 **

Write a `torch.nn.Module` which fulfills the requirements above. The model should establish three convolutions in its constructor. Also, the `forward` function should contain the execution of these 3 convolutions to create $\mathbf{Q}, \mathbf{K}, \mathbf{V}$.

Feel free to start from this template below:
``` python

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        query_channels: int,
        key_channels: int,
        out_dimension: int = 0,
    ):
        '''

        constructor to set up self-attention module

        Parameters
        ----------
        in_channels :
            depend on the data
        query_channels :
            equivalent to T
        key_channels :
            equivalent to T_prime
        out_dimension :
            equivalent to D_prime, not needed here
        '''
        super().__init__()

        # we want to establish queries Q, keys K and values V
        # instead of using Linear layers, we opt for Conv1D as they use less
        # parameters and hence are less memory intensive
        ...

    def forward(self, x):
        ...

```
"""

# %% [markdown]
r"""
The actual attention mechanism is two lines of math, but it may take a while to digest it. We start from considering the fact, that we have three matrices now: $\mathbf{Q},\mathbf{K},\mathbf{V}$.

We first need to calculate the attention matrix, sometimes also called the attention map. The attention map in our case computes a paired dot-product between all rows in $\mathbf{Q}$ and all rows in $\mathbf{K}$ (i.e. all columns in $K^{T}$. In other words, we compute all n-to-n distances of vectors in either matrix. Note, the normalisation by $\sqrt{D}$ is injected for numerical stability according to the authors.

$$ \mathbf{A}' = \frac{Q K^{T}}{\sqrt{D}} $$

To make these meaningful, the attention map is completed by a row-wise application of the softmax function:

$$ \mathbf{A} = softmax_{row}(\mathbf{A}') = softmax_{row}(\frac{QK^{T}}{\sqrt{D}}) $$

To illustrate the above, here is a sketch of what is going on:

<div style="display: block;margin-left: auto;margin-right: auto;width: 40%;"><img src="img/attention_sketch_details_smaller.png" alt="Computing the Attention Map A"></div>
"""

# %% [markdown]
r"""
To finish up, we obtain the output of attention by weighting the attention map with $\mathbf{V}$.

$$ \mathbf{Y} = \mathbf{A}\mathbf{V} $$

The matrix $\mathbf{Y}$ is the output of our Attention layer.

Note, the operation described above is called *self-attention* because we obtained $\mathbf{V}$ from the same input as $\mathbf{Q}$ and $\mathbf{K}$. This stands in contrast to *cross-attention* where $\mathbf{V}$ is constructed from a different output than $\mathbf{Q}$ and $\mathbf{K}$. Here is an illustration to show, what is going on:

<div style="display: block;margin-left: auto;margin-right: auto;width: 40%;"><img src="img/attention_sketch_schemes_cropped.png" alt="Self- and Cross-Attention"></div>
"""

# %% [markdown]
r"""
** Exercise 03.3 **

Given the formulae above, reuse the SelfAttention class that you already started and complete it! Use the following template as a guide:
"""

# %%
class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        query_channels: int,
        key_channels: int,
        out_dimension: int = 0,
    ):
        """

        constructor to set up self-attention module

        Parameters
        ----------
        in_channels :
            depend on the data
        query_channels :
            equivalent to T
        key_channels :
            equivalent to T_prime
        out_dimension :
            equivalent to D_prime, not needed here
        """
        super().__init__()

        # we want to establish queries Q, keys K and values V
        # instead of using Linear layers, we opt for Conv1D as they use less
        # parameters and hence are less memory intensive
        self.conv_Q = torch.nn.Conv1d(
            in_channels, query_channels, kernel_size=1, bias=False
        )
        self.conv_K = torch.nn.Conv1d(
            in_channels, key_channels, kernel_size=1, bias=False
        )
        self.conv_V = torch.nn.Conv1d(
            in_channels, key_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        # we receive a NxCxD tensor x

        # run the convolutions on our inputs
        Q = self.conv_Q(x)  # produces a NxTxD tensor
        K = self.conv_K(x)  # produces a NxT_primexD tensor
        V = self.conv_V(x)  # produces a NxT_primexD tensor as D_prime = D

        K_t = ...  # transpose K to give a NxDxT_prime matrix
        A_ = ...  # perform a matrix multiply Q*K_t, results in a NxTxT_prime matrix
        A = ...  # perform a row-wise softmax, results in a NxTxT_prime matrix

        y = ...  # perform a matrix multiply A*Y, results in TxT_prime * T_primexD = TxD matrix

        return y


# %% jupyter={"source_hidden": true}
# *Solution 03.3*
class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        query_channels: int,
        key_channels: int,
        out_dimension: int = 0,
    ):
        """

        constructor to set up self-attention module

        Parameters
        ----------
        in_channels :
            depend on the data
        query_channels :
            equivalent to T
        key_channels :
            equivalent to T_prime
        out_dimension :
            equivalent to D_prime, not needed here
        """
        super().__init__()

        # we want to establish queries Q, keys K and values V
        # instead of using Linear layers, we opt for Conv1D as they use less
        # parameters and hence are less memory intensive
        self.conv_Q = torch.nn.Conv1d(
            in_channels, query_channels, kernel_size=1, bias=False
        )
        self.conv_K = torch.nn.Conv1d(
            in_channels, key_channels, kernel_size=1, bias=False
        )
        self.conv_V = torch.nn.Conv1d(
            in_channels, key_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        # we receive a NxCxD tensor x

        # run the convolutions on our inputs
        Q = self.conv_Q(x)  # produces a NxTxD tensor
        K = self.conv_K(x)  # produces a NxT_primexD tensor
        V = self.conv_V(x)  # produces a NxT_primexD tensor as D_prime = D

        K_t = torch.transpose(K, -1, -2)  # convert to NxDxT_prime
        A_ = torch.matmul(Q, K_t)  # results in NxTxT_prime
        A = torch.nn.functional.softmax(A_, dim=2)  # results in NxTxT_prime

        y = torch.matmul(A, V)  # results in TxT_prime * T_primexD = TxD

        return y



# %%
# now we want to include that into a model
class CustomAttn(torch.nn.Module):

    def __init__(self, inshape=x_test_.shape[-2:], num_channels=64, ksize=5):

        super().__init__()

        self.layers = torch.nn.Sequential()
        num_inchannels = inshape[0]
        self.num_channels = num_channels
        padding = ksize // 2
        self.layers.append(
                torch.nn.Conv1d(num_inchannels, num_channels, ksize,stride=1,padding=padding )
            )
        self.layers.append(
                torch.nn.ReLU()
        )
        self.layers.append(
                torch.nn.Conv1d(num_channels, num_channels, ksize,stride=1,padding=padding )
            )
        self.layers.append(
                torch.nn.ReLU()
        )
        self.layers.append(
                SelfAttention(num_channels,num_channels,num_channels)
        )
        self.layers.append(
            torch.nn.Conv1d(num_channels, num_channels, ksize,stride=1,padding=padding )
        )
        self.layers.append(
            torch.nn.Conv1d(num_channels, num_inchannels, 1,stride=1,padding=0 )
        )

    def forward(self, x):

        return self.layers(x)

# test our model
attmodel = CustomAttn(x_test_.shape[1:], num_channels=64)
output = attmodel(first_x)
assert output.shape == first_y.shape
print(attmodel)
print(f"set up custom attention model with {nparams(attmodel)} parameters")

# %%
# train the model
attoptim = torch.optim.AdamW(attmodel.parameters(), lr=1e-3)
attcrit  = torch.nn.MSELoss()
max_epochs = 15
attresults = train_regression(attmodel, attoptim, attcrit, train_loader, test_loader, max_epochs,2)

# %%
f,ax = plot_history(attresults)
f.savefig("attention_attmodel_losses.svg")

# %% [markdown]
"""
## Have a look at the results

In the loss curve above, you see that the attention based model draws much quicker to `0` in the training set than what the FCN above could do. This trend will increase, the longer you train. The interesting observation here is that both models have the same amount of parameters. Hence, the attention mechanism must bring more flexibility along to model the data.

Next, we want to visualize some output sequences and check if the model has truly grasped long range dependencies.
"""

# %%
# obtain predictions

test_input, test_targets = torch.from_numpy(x_test_[:32,...]), torch.from_numpy(y_test_[:32,...])
test_outputs = attmodel(test_input)
ctest_outputs = plainfcn(test_input)

# required for easier plotting
test_input = test_input.detach().cpu()
test_outputs = test_outputs.detach().cpu()
ctest_outputs = ctest_outputs.detach().cpu()
test_targets = test_targets.detach().cpu()

# for the correct plot fo attention, we need to adapt to scaling issues
size_scale = attmodel.num_channels / seq_length

# plot outputs
for k in range(15):
    save_sequence_images(
        f"attention_attmodel_test_Y_{k:03.0f}.svg",
        [
            (test_input[k, 0], "blue", "Input"),
            (test_outputs[k, 0], "orange", "Output"),
        ],
        seq_length=seq_length,
        seq_height_max=1.
    )

    save_sequence_images(
        f"attention_fcnmodel_test_Y_{k:03.0f}.svg",
        [
            (test_input[k, 0], "blue", "Input"),
            (ctest_outputs[k, 0], "orange", "Output"),
        ],
        seq_length=seq_length,
        seq_height_max=1.
    )

# %% [markdown]
r"""
# Wrap-Up

<div style="display: block;margin-left: auto;margin-right: auto;width: 30%;"><img src="https://ar5iv.labs.arxiv.org/html/1706.03762/assets/Figures/ModalNet-21.png" alt="Transformer Architecture"></div>

The image above illustrates schematically how a transformer model was setup originally. Find the three instances of Multi-Head Attention. This is where attention is used. Note also, that we see two times self-attention (bottom) and one cross-attention where is $\mathbf{Q}$ and $\mathbf{K}$ is produced from an input and $\mathbf{Y}$ from another (top).
"""
