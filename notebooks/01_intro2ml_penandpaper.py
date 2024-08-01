# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A multi-layer perceptron with pen and paper
#
# Multi-layer Perceptrons (MLP) were some of the first Machine Learning (ML) architectures in use.
#
# This notebook will guide you through how to
# - perform a forward pass through a very simple MLP
# - how to backpropagate through the network
# - how to perform the same operations with [pytorch](https://pytorch.org)

# %% [markdown]
# ## A simple MLP
#
# To start off, we want to construct a very simple MLP with one input unit, one hidden unit and one ouput unit. We will keep everything lightweight and one dimensional.
#
# <div style="display: block;margin-left: auto;margin-right: auto;width: 75%;"><img src="img/01_1D_mlp.svg" alt="linear MLP"></div>
#
# $$ \text{Input}\,x \rightarrow \sigma (w \cdot x' + b) \rightarrow \text{Output}\,y' \rightarrow \mathcal{L}(y,y') $$
#
# In the above, the network receives an input datum $x$. This is used as input to the first hidden unit $\sigma (w \cdot x' + b)$. The hidden unit consists of 3 ingredients:
# - the weight $w$
# - the bias $b$
# - the activation function $\sigma$
#
# The output of the hidden unit will be considered to be the output of the forward pass. In order to establish a learning process, this prediction $y'$ will be compared with a label $y$ using the loss the function $\mathcal{L}(y,y')$.

# %% [markdown]
# **Exercise**
#
# <div style="display: block;margin-left: auto;margin-right: auto;width: 75%;"><img src="img/01_1D_mlp_filled.svg" alt="linear MLP with values"></div>
#
# Take pen and paper. Compute a full forward pass using the following values:
#
# - input $x = 2$
# - weight $w = .5$
# - bias $b = 1$
# - label $y = 1.5$
#
# Use the [ReLU function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) for $\sigma$ and the mean squared error ([MSE](https://en.wikipedia.org/wiki/Mean_squared_error)) for the loss function.

# %% [markdown] jupyter={"source_hidden": true}
# > **Solution**
# > 1. Compute $w \cdot x' + b$: We obtain `2`.
# > 2. Apply the ReLU: We obtain `y'=2` again (2 is larger than 0 and hence $f_{ReLU}(x=2)=2$).
# > 3. Compute the loss: $\mathcal{L}(y,y') = (y-y')^2 = (2 - 1.5)^2 = \frac{1}{4}$ 

# %%
