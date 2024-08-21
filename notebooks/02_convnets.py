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
# # Convolutional Neural Networks
#
# As we covered Multi-layer Perceptrons (MLP) in the last exercise. We want to explore using `pytorch` for ML more. For this, we also will start looking into another popular network architecture called convolutional neural networks.
#
# We start this exploration with a small motivating Gedankenexperiment.
#
# ## 02.1 A MLP for my Phone
#
# Take out your phone. Check in the picture galery some of the last images you took. Try to find out how many pixels such an image has! You should arrive with a tuple that denominates height and width of such an image.
#
# Imagine you want to design an MLP network to categorize the pictures you took on your camera. We convert each image into a flat vector before feeding it into the network. We assume you like to use three hidden layers with 128 hidden neurons each. Then (for sake of the example) you want to predict a single category, i.e. your last layer has a size of 1. In this scenario, compute how many parameters such an MLP might have.

# %% [markdown] jupyter={"source_hidden": true}
# ### 02.1 Solution
#
# On my phone, I go to "Gallery", then select a picture. Then I "click" on properties (some phones call this information) and see that it has 2160 pixels in width and 3840 pixels in height (`2160x3840`). We ignore for the time being that most of these images are encoded as RGB or RGBA or even something else.
#
# When such an image is flattened, it produces a vector of size `8294400`. For an MLP, every hidden neuron connects with every other hidden neuron. Thus, our MLP would be structured like this:
#
# ```
# input (output shape: 8294400) -> hidden (output shape: 128) -> hidden (output shape: 128) -> hidden (output shape: 128) -> output (output shape: 1)
# ```
# That means, that for the first hidden layer, we require $128 \cdot 8,294,400 = 1,061,683,200$ parameters. Every hidden layer consists of the weight matrix and a bias vector. The bias vector has the same size as the number of hidden neurons. With this information, we can calculate the number of parameters:
#
# 1. layer: $128 \cdot 8,294,400 = 1,061,683,200$ weights, $128$ bias terms, $1,061,683,200+128=1,061,683,328$ parameters
# 2. layer: $128 \cdot 128 = 16384$ weights, $128$ bias terms, $16384+128=16512$ parameters
# 3. layer: $128 \cdot 128 = 16384$ weights, $128$ bias terms, $16384+128=16512$ parameters
# 4. layer: $128 \cdot 1 = 128$ weights, $1$ bias term, $128+1=129$ parameters
#
# In sum, this network would have $1061716481$ parameters. As each trainable parameter in a pytorch model is typically a `float32` number. This would result in the model to be of size $1,061,716,481 \cdot 4 \text{Byte} = 4,246,865,924 \text{Byte} \approx 3.9 \text{GiB} \approx 4.2 \text{GB}$. Such a model would already exceed some GPU's memory. So we better look for a way to have neural networks with smaller number of parameters.

# %% [markdown]
# ## Loading 1D Training Data
#
# To explore convolutions, we will start out considering a one dimensional sequence. The sequence will be taken from the [MNIST1D](https://github.com/greydanus/mnist1d) dataset. The advantage of this dataset is, that it is small and can serve well to demonstrate key concepts of machine learning.
#
# <div style="display: block;margin-left: auto;margin-right: auto;width: 75%;"><img src="https://github.com/greydanus/mnist1d/raw/master/static/overview.png" alt="MNIST1D overview"></div>

# %%
from mnist1d.data import get_dataset_args, make_dataset

defaults = get_dataset_args()
data = make_dataset(defaults)
X, y = data['x'], data['y']

X.shape, y.shape

# %% [markdown]
# As we are interested in supervised learning, we rely on pairs of input data `X` and labels `y`. Here, the labels `y` refer to the number which each sequence in `X` resembles. Let's have a look at these sequences.

# %%
import matplotlib.pyplot as plt

f, ax = plt.subplots(2, 5, figsize=(14, 5), sharex=True, sharey=True)

for sample in range(10):
    col = sample % 5
    row = sample // 5
    ax[row, col].plot(X[sample, ...])
    label = y[sample]
    ax[row, col].set_title(f"label {label}")
    if row == 1:
        ax[row, col].set_xlabel(f"samples / a.u.")
    if col == 0:
        ax[row, col].set_ylabel(f"intensity / a.u.")

f.suptitle("MNIST1D examples")
f.savefig("mnist1d_default_first10.svg")

# %% [markdown]
# You can tell from the above, that the signals are far from easy to distinguish. This gives rise for the need for a flexible network architecture. But let's not get ahead of ourselves. Before we dive into designing a network, we will have to create a dataloader first. In `pytorch`, having a `DataLoader` for your data is the first hurdle you have to overcome to get started.

# %%
from typing import Callable

import torch
from sklearn.model_selection import train_test_split


class MNIST1D(torch.utils.data.Dataset):

    def __init__(self,
                 train:bool = True,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 mnist1d_args: dict = get_dataset_args(),
                 seed: int = 42):

        super().__init__()

        self.is_training = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = make_dataset(mnist1d_args)
        X_train, X_test, y_train, y_test = train_test_split(data['x'],
                                                            data['y'],
                                                            test_size=0.1,
                                                            random_state=seed)
        if train:
            self.X = X_train
            self.y = y_train
        else:
            self.X = X_test
            self.y = y_test

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index: int):

        X = self.X[index, ...]
        if self.transform is not None:
            X = self.transform(X)
        y = self.y[index, ...]
        if self.target_transform is not None:
            y = self.target_transform(y)
        return X, y

# %% [markdown]
# In `pytorch`, the Dataset class has to comply to 3 requirements:
# - it has to inherit from torch.utils.data.Dataset
# - it has to define a `__len__` function so that we can later call `len(mydataset)`
# - it has to define a `__getitem__` function so that we can later call `mydataset[12]` to obtain sample 12
# For more details, see the `pytorch` documentation on [Creating a Custom Dataset for your files](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files) or [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

# %%
training_data = MNIST1D()
test_data = MNIST1D(train=False)

nsamples = len(training_data)+len(test_data)
assert nsamples == 4000, f"number of samples for MNIST1D is not 4000 but {nsamples}"

# %% [markdown]
# In order to use the dataset for training, we need to create a DataLoader. A DataLoader orchestrates how the data is loaded and provided for the compute device that you intend to use. Note how we can set how many MNIST1D sequences at once will be provided to the compute device. This number, called the batch size, is set to `64` in the example below.

# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_X, train_y = next(iter(train_dataloader))
print("obtained first batch of training data and labels with shapes",train_X.shape, train_y.shape)

# %% [markdown]
# ## Convolutions in 1D
#
# Up until here, we have made great strides to load our dataset. We now want to explore what convolutions are and how to use them to construct a convolutional neural network (CNN).
#
# A convolution is a mathematical operation. Here, we only consider convolutions on a discrete signal $x \in \mathbb{R}^{n}$ using a convolution kernel $w \in \mathbb{R}^{m}$ where $m << n$ usually. For a fixed offset $i$ in the output signal $y$, a convolution $y = x \ast w$ is defined by:
# $$ y_{i} = \sum_{k=0}^{k-1} x_{i+m-k} \cdot w_{k} $$
#
# ###
