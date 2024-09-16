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

The following material is taken from the excellent lectures of Francois Fleuret (Univeristy of Geneva). Francois Fleuret introduces the self-attention mechanism in three parts: [first](https://fleuret.org/dlc/streaming/dlc-video-13-1-attention-memory-translation.mp4), [second](https://fleuret.org/dlc/streaming/dlc-video-13-2-attention-mechanisms.mp4) and [third](https://fleuret.org/dlc/streaming/dlc-video-13-3-transformers.mp4) - all of which are worthwhile watching. I have asked permission of Francois to reuse some of his material.

## A regression task

In the following, we again look at a regression task. The functions below produce a dataset which can be used to illustrate the use of the attention mechanism. The dataset exhibits 2 triangles and 2 boxes/rectangles on a 1D line. In this notebook, we focus on

1. exploring the data set with a standard convnet.
2. preparing a network using the attention mechanism
3. compare the performance of the two
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from fleuret_data import generate_sequences

drng = np.random.default_rng(43)  # set the RNG seed for reproducible runs


# %%
# create train and test set
train_input, train_targets, train_tr, train_bx = generate_sequences(
    15000, seq_length=64, rng=drng
)
test_input, test_targets, test_tr, test_bx = generate_sequences(
    1000, seq_length=64, rng=drng
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
    color="red",
    label="target",
)
ax[1].set_title("target")
fig.savefig("attention_dataset.svg")

# %% [markdown]
"""
You see two kinds of "objects" in the signal plotted above: two box-like structures and two triangle-like structure. We now define a **regression task** which is meant to equalize the height of the boxes (new height should be the average height of the two input boxes) and the height of the triangles (new height of the triangles should be the mean of the two input triangles).
"""

# %% [markdown]
r"""
## Convolutional Network

First, we need to normalize the data into a dynamic range as $\vec{x} \in [0,1]$. Then, we like to create a regression model using convolutions only, which tries to accomplish the task above.

### Data Normalisation
"""

# %%
import torch
# set the seeds to make the notebook reproducible
np.random.seed(41)
torch.random.manual_seed(43)

# %%
# normalize the signal, zscale if required
x_min, x_max = train_input.min(), train_input.max()
x_ = (train_input - x_min) / (x_max - x_min)

y_min, y_max = train_targets.min(), train_targets.max()
y_ = (train_targets - y_min) / (y_max - y_min)

x_test_ = (test_input - x_min) / (x_max - x_min)
y_test_ = (test_targets - y_min) / (y_max - y_min)


# %%
print(f"data shape check: {x_.shape, y_.shape, x_test_.shape, y_test_.shape}")

# %% [markdown]
"""
### A fully convolutional model

The term fully convolutional describes an architecture which consists exclusively of convolutional operations. This has the benefit as the model design is independent of the input data shape.
"""

# %%
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
# prepare the data
train_ds = torch.utils.data.StackDataset(x_, y_)
test_ds = torch.utils.data.StackDataset(x_test_, y_test_)

train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=64)

# %%
# test our model

plainfcn = RegressionFCN(x_.shape[-2:])
print(plainfcn)
first_x, first_y = next(iter(train_loader))
output = plainfcn(first_x)
assert output.shape == first_y.shape

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
optim = torch.optim.AdamW(plainfcn.parameters(), lr=5e-4)
crit  = torch.nn.MSELoss()
max_epochs = 10
fcnresults = train_regression(plainfcn, optim, crit, train_loader, test_loader, max_epochs,2)

# %%

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
f.savefig("attention_plainfcn_losses.svg")


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
"""
# The above is not a great model, actually it doesn't work at all! But we expected no less as the test loss didn't decrease any further than `0.0085` while the training loss decreased further and further.
"""

# %% [markdown]
"""
# Your own Attention Layer

Attention and self-attention are very powerful transformations. In this section, we will write our own Attention layer.
"""

# %%

class SelfAttention(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, key_channels
    ):
        super().__init__()

        # we want to establish queries Q, keys K and values V
        # instead of using Linear layers, we opt for Conv1D as they use less
        # parameters and hence are less memory intensive
        self.conv_Q = torch.nn.Conv1d(in_channels,
                                      key_channels,
                                      kernel_size=1,
                                      bias=False
                                      )
        self.conv_K = torch.nn.Conv1d(in_channels,
                                      key_channels,
                                      kernel_size=1,
                                      bias=False
                                      )
        self.conv_V = torch.nn.Conv1d(in_channels,
                                      out_channels,
                                      kernel_size=1,
                                      bias=False
                                      )

    def forward(self, x):
        # run the convolutions on our inputs
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)

        # TODO: perform a tensor transpose
        #       you want to transpose the very last dimension with the second to last
        K_t = torch.transpose(K, -1, -2)

        # TODO: perform a matrix multiplication of Q*K_t
        A_ = torch.matmul(Q,K_t)

        # TODO: perform a row-wise softmax of A_
        A = torch.nn.functional.softmax(A_,dim=-2)

        # TODO: perform a matrix multiplication of A*V
        y = torch.matmul(A,V)

        return y


# # %%
class CustomAttn(torch.nn.Module):

    def __init__(self, inshape=x_test_.shape[-2:], num_channels=64, ksize=5):

        super().__init__()

        self.layers = torch.nn.Sequential()
        num_inchannels = inshape[0]
        padding = ksize // 2
        self.layers.append(
                torch.nn.Conv1d(num_inchannels, num_channels, ksize,stride=1,padding=padding )
            )
        self.layers.append(
                torch.nn.ReLU()
        )
        self.layers.append(
                SelfAttention(num_channels,num_channels,num_channels)
        )
        self.layers.append(
            torch.nn.ReLU()
        )
        self.layers.append(
            torch.nn.Conv1d(num_channels, num_channels, ksize,stride=1,padding=padding )
        )
        self.layers.append(
            torch.nn.Conv1d(num_channels, num_inchannels, 1,stride=1,padding=0 )
        )

    def forward(self, x):

        return self.layers(x)


attmodel = CustomAttn(x_test_.shape[1:])
output = attmodel(first_x)
assert output.shape == first_y.shape
print(attmodel)

# %%
attoptim = torch.optim.AdamW(plainfcn.parameters(), lr=1e-3)
attcrit  = torch.nn.MSELoss()
max_epochs = 10
attresults = train_regression(attmodel, attoptim, attcrit, train_loader, test_loader, max_epochs,2)

# chistory = cmodel.fit(
#     x, y, validation_data=(x_test, y_test), batch_size=128, epochs=10, verbose=1
# )

# %%
f,ax = plot_history(attresults)
f.savefig("attention_attmodel_losses.svg")

# %% [markdown]
"""
## Create a model with attention

The idea of attention was published in 2014 by A. Graves in "Neural Turing Machines", see [here](https://arxiv.org/abs/1410.5401)
It was picked up again in 2017 by A. Vaswani et al in "Attention is all you need", see [here](https://arxiv.org/abs/1706.03762). This paper coined the term Transformer model which relies strongly on self-attention layers.
A nice visualizer to help you grasp the idea of attention can be found [here](https://poloclub.github.io/transformer-explainer/).
"""

# # %%
class CustomTorchAttn(torch.nn.Module):

    def __init__(self, inshape=x_test_.shape[-2:], num_channels=64, ksize=5):

        super().__init__()

        self.head_layers = torch.nn.Sequential()
        num_inchannels = inshape[0]
        padding = ksize // 2
        self.head_layers.append(
                torch.nn.Conv1d(num_inchannels, num_channels, ksize,stride=1,padding=padding )
            )
        self.head_layers.append(
                torch.nn.ReLU()
        )
        self.head_layers.append(
                torch.nn.Conv1d(num_channels, num_channels, ksize,stride=1,padding=padding )
            )
        self.head_layers.append(
                torch.nn.ReLU()
        )
        self.attn_layer = torch.nn.MultiheadAttention(num_channels,
                                                      num_heads=1,
                                                      kdim=num_channels,
                                                      vdim=num_channels,
                                                      batch_first=True)


        self.tail_layers = torch.nn.Sequential()
        self.tail_layers.append(
            torch.nn.Conv1d(num_channels, num_channels, ksize,stride=1,padding=padding )
        )
        self.tail_layers.append(
            torch.nn.ReLU()
        )

        self.tail_layers.append(
            torch.nn.Conv1d(num_channels, num_inchannels, 1,stride=1,padding=0 )
        )

    def forward(self, x):

        Q_ = self.head_layers(x)
        K_ = self.head_layers(x)
        V_ = self.head_layers(x)

        V = V_.swapaxes(-1, -2)
        K = K_.swapaxes(-1, -2)
        Q = Q_.swapaxes(-1, -2)

        embedded, _ = self.attn_layer(Q, K, V, need_weights=False)
        value = self.tail_layers(embedded)

        return value

tattmodel = CustomTorchAttn(x_test_.shape[1:])
output = tattmodel(first_x)
assert output.shape == first_y.shape
print(tattmodel)

# %%
tattoptim = torch.optim.AdamW(plainfcn.parameters(), lr=1e-3)
tattcrit  = torch.nn.MSELoss()
max_epochs = 10
attresults = train_regression(tattmodel, tattoptim, tattcrit, train_loader, test_loader, max_epochs,2)

# %%
f,ax = plot_history(attresults)
f.savefig("attention_tattmodel_losses.svg")

# # We reuse the model idea from above
# def create_attn(inshape=x.shape[-2:], channels=64, ksize=5):
#     "a fully convolutional network (fcn) to regress the signal using selfattention from keras"

#     inputs = keras.layers.Input(shape=inshape)
#     x = keras.layers.Conv1D(
#         channels, ksize, strides=1, padding="same", activation="relu"
#     )(inputs)
#     x = keras.layers.Conv1D(
#         channels, ksize, strides=1, padding="same", activation="relu"
#     )(x)
#     # TODO: Keras also has a built-in Attention Layer, find it
#     #       and use in in similar fashion as our own custom attention above
#     #       (note, we want to use one attention head)
# ...
# ...
# ...
# ...
# ...
#     x = keras.layers.Conv1D(
#         channels, ksize, strides=1, padding="same", activation="relu"
#     )(x)
#     outputs = keras.layers.Conv1D(1, ksize, strides=1, padding="same")(x)

#     return keras.Model(
#         inputs=inputs, outputs=outputs, name="fcn-regression-selfattention"
#     )


# # %%

# amodel = create_attn(x.shape[1:])
# amodel.summary()  # a simple model

# # %% [markdown]
# # The keras built-in attention layer uses Linear layers internally. This gives rise to the large number of parameters in the multi-head attention layer above even though we only want to use 1 head.
# #
# # Let's compile the model and see the effect of this change.

# # %%
# amodel.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())

# # %%
# ahistory = amodel.fit(
#     x, y, validation_data=(x_test, y_test), batch_size=128, epochs=15, verbose=1
# )


# # %% [markdown]
# # -

# # %%
# plot_histories(
#     [history, ahistory, chistory],
#     ["loss", "val_loss"],
#     "vanilla,self-attention, custom-attention".split(","),
# )
