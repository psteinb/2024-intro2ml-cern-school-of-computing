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
# # Exploring a reweighting task
#
# The following functions produce a dataset which can be used to illustrate the use of attention. In this notebook, we focus on exploring the data set with a standard convnet. The dataset exhibits 2 triangles and 2 boxes/rectangles on a 1D line. Think of this example being very similar to measuring the deposited energy xray radiation and a photon beam when traversing matter. The xrays deposit their energy continuously with some attentuation upto very high depth into the solid state object. Particles (like protons) exhibit a behavior called a Bragg peak, i.e. at a specific depth almost all of the dose is deposited and the beam does not traverse further.
#
# See also:
# ![https://en.wikipedia.org/wiki/Bragg_peak](https://en.wikipedia.org/wiki/Bragg_peak#/media/File:BraggPeak-en.svg)
#

# %%
import matplotlib.pyplot as plt
import numpy as np

from fleuret_data import generate_sequences

drng = np.random.default_rng(43)  # set the RNG seed for reproducible runs


# %%
# create train and test set
train_input, train_targets, train_tr, train_bx = generate_sequences(
    15000, seq_length=128, rng=drng
)
test_input, test_targets, test_tr, test_bx = generate_sequences(
    1000, seq_length=128, rng=drng
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


# %% [markdown]
# You see two kinds of "objects" in the signal above: two box-like structures and two triangle-like structure. We define a **regression task** which is meant to equalize the height of the boxes (new height should be the average height of the two input boxes) and the height of the triangles (new height of the triangles should be the mean of the two input triangles).

# %% [markdown]
# ## Convolutional Network
#
# In the following, we like to create a regression model using convolutions only, which tries to accomplish the task above.

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
x_.shape, y_.shape, x_test_.shape, y_test_.shape

# %%
# keras being shipped with tensorflow is unable to digest this data
# because it does we have to reshape it
x = np.swapaxes(x_, -2, -1)
y = np.swapaxes(y_, -2, -1)

x_test = np.swapaxes(x_test_, -2, -1)
y_test = np.swapaxes(y_test_, -2, -1)

x.shape, y.shape, x_test.shape, y_test.shape

# %% [markdown]
# # creating the model

# %%
class RegressionFCN(torch.nn.Module):

    def __init__(self, inshape, channels=64, ksize=5):
        super().__init__()



def create_fcn(inshape=x.shape[-2:], channels=64, ksize=5):
    "a fully convolutional network (fcn) to regress the signal"

    inputs = keras.layers.Input(shape=inshape)
    x = keras.layers.Conv1D(
        channels, ksize, strides=1, padding="same", activation="relu"
    )(inputs)
    x = keras.layers.Conv1D(
        channels, ksize, strides=1, padding="same", activation="relu"
    )(x)
    x = keras.layers.Conv1D(
        channels, ksize, strides=1, padding="same", activation="relu"
    )(x)
    x = keras.layers.Conv1D(
        channels, ksize, strides=1, padding="same", activation="relu"
    )(x)
    outputs = keras.layers.Conv1D(1, ksize, strides=1, padding="same")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="fcn-regression")


# %%

plainfcn = create_fcn(x.shape[1:])
plainfcn.summary()  # a simple model

# %%
plainfcn.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())

# %%
history = plainfcn.fit(
    x, y, validation_data=(x_test, y_test), batch_size=128, epochs=15, verbose=1
)

# %%
import seaborn as sns
import pandas as pd

def plot_history(history, metrics, draw_legend=True):
    """
    Plot the training history

    Args:
        history (keras History object that is returned by model.fit())
        metrics(str, list): Metric or a list of metrics to plot
    """
    history_df = pd.DataFrame.from_dict(history.history)
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("metric")
    if not draw_legend:
        plt.legend().set_visible(False)


# %%

plot_history(history, ["loss", "val_loss"])

# %%
pred5 = plainfcn.predict(x_test[:5, ...])
pred5_ = np.swapaxes(pred5, -2, -1)  # useful for plotting
pred5.shape, pred5_.shape


# %%
xaxis = np.arange(0, x_.shape[-1], 1)
print(xaxis.shape)

plt.plot(xaxis, y_test_[0:1, 0, ...].squeeze(), color="green", label="label")
plt.plot(xaxis, pred5_[0:1, 0, ...].squeeze(), color="red", label="prediction")
plt.legend()


# %% [markdown]
# The above is not a great model, actually it doesn't work at all! But we expected no less as the loss didn't decrease any further than `0.0033`.

# %%

def plot_histories(histories, metrics, hist_labels=[], draw_legend=True):
    """
    Plot the training progression of several histories

    Args:
        histories (list of keras History objects that was returned by model.fit())
        metrics(str, list): Metric or a list of metrics to plot
        hist_labels(list): list of strings describing each history
    """
    assert len(histories) == len(hist_labels)

    cols = {}
    for hidx in range(len(histories)):
        hist = pd.DataFrame.from_dict(histories[hidx].history)
        lab = hist_labels[hidx]
        for m in metrics:
            prefix = f"{lab}_{m}"
            col = hist[m]
            cols[prefix] = col

    dataframe = pd.DataFrame.from_dict(cols)
    sns.lineplot(data=dataframe)
    plt.xlabel("epochs")
    plt.ylabel("metric")
    if not draw_legend:
        plt.legend().set_visible(False)


# %% [markdown]
# # Your own Attention Layer
#
# In this section, we will write our own Attention layer.

# %%
from keras import ops


class SelfAttention(keras.layers.Layer):
    def __init__(
        self, in_channels, out_channels, key_channels, data_format="channels_last"
    ):
        super().__init__()

        # we want to establish queries Q, keys K and values V
        # instead of using Linear layers, we opt for Conv1D as they use less
        # parameters
        self.conv_Q = keras.layers.Conv1D(
            filters=key_channels, kernel_size=1, data_format=data_format, use_bias=False
        )
        self.conv_K = keras.layers.Conv1D(
            filters=key_channels, kernel_size=1, data_format=data_format, use_bias=False
        )
        self.conv_V = keras.layers.Conv1D(
            filters=out_channels, kernel_size=1, data_format=data_format, use_bias=False
        )

    def call(self, inputs):
        # run the convolutions on our inputs
        Q = self.conv_Q(inputs)
        K = self.conv_K(inputs)
        V = self.conv_V(inputs)

        # You will need to use the Keras OPS API for the operations below
        # https://keras.io/api/ops/

        # TODO: perform a tensor transpose
        #       you want to transpose the very last dimension with the second to last
...

        # TODO: perform a matrix multiplication of Q*K_t
...

        # TODO: perform a row-wise softmax of A_
...

        # TODO: perform a matrix multiplication of A*V
...

        return y


# %%


def custom_attn(inshape=x.shape[-2:], channels=64, ksize=5):
    "a fully convolutional network (fcn) to regress the signal using selfattention"

    inputs = keras.layers.Input(shape=inshape)
    x = keras.layers.Conv1D(
        channels, ksize, strides=1, padding="same", activation="relu"
    )(inputs)
    x = keras.layers.Conv1D(
        channels, ksize, strides=1, padding="same", activation="relu"
    )(x)
    # TODO: Use the SelfAttention class that we wrote above
...
    x = keras.layers.Conv1D(
        channels, ksize, strides=1, padding="same", activation="relu"
    )(x)
    outputs = keras.layers.Conv1D(1, ksize, strides=1, padding="same")(x)

    return keras.Model(
        inputs=inputs, outputs=outputs, name="fcn-regression-custom-attention"
    )


# %%

cmodel = custom_attn(x.shape[1:])
cmodel.summary()  # a simple model

# %%
cmodel.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())

# %%
chistory = cmodel.fit(
    x, y, validation_data=(x_test, y_test), batch_size=128, epochs=10, verbose=1
)


# %% [markdown]
# ## Create a model with attention
#
# The idea of attention was published in 2014 by A. Graves in "Neural Turing Machines", see https://arxiv.org/abs/1410.5401
# It was picked up again in 2017 by A. Vaswani et al in "Attention is all you need", see https://arxiv.org/abs/1706.03762
#
#

# %%


# We reuse the model idea from above
def create_attn(inshape=x.shape[-2:], channels=64, ksize=5):
    "a fully convolutional network (fcn) to regress the signal using selfattention from keras"

    inputs = keras.layers.Input(shape=inshape)
    x = keras.layers.Conv1D(
        channels, ksize, strides=1, padding="same", activation="relu"
    )(inputs)
    x = keras.layers.Conv1D(
        channels, ksize, strides=1, padding="same", activation="relu"
    )(x)
    # TODO: Keras also has a built-in Attention Layer, find it
    #       and use in in similar fashion as our own custom attention above
    #       (note, we want to use one attention head)
...
...
...
...
...
    x = keras.layers.Conv1D(
        channels, ksize, strides=1, padding="same", activation="relu"
    )(x)
    outputs = keras.layers.Conv1D(1, ksize, strides=1, padding="same")(x)

    return keras.Model(
        inputs=inputs, outputs=outputs, name="fcn-regression-selfattention"
    )


# %%

amodel = create_attn(x.shape[1:])
amodel.summary()  # a simple model

# %% [markdown]
# The keras built-in attention layer uses Linear layers internally. This gives rise to the large number of parameters in the multi-head attention layer above even though we only want to use 1 head.
#
# Let's compile the model and see the effect of this change.

# %%
amodel.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())

# %%
ahistory = amodel.fit(
    x, y, validation_data=(x_test, y_test), batch_size=128, epochs=15, verbose=1
)


# %% [markdown]
# -

# %%
plot_histories(
    [history, ahistory, chistory],
    ["loss", "val_loss"],
    "vanilla,self-attention, custom-attention".split(","),
)
