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
"""
# Autoencoders

An autoencoder is a type of artificial neural network used for learning efficient codings of input data. It's essentially a network that attempts to replicate its input (encoding) as its output (decoding), but the network is designed in such a way that it must learn an efficient representation (compression) for the input data in order to map it back to itself.

The importance of autoencoders lies in their ability to learn the underlying structure of complex data, making them valuable tools for scientific data analysis. Here's how:

1. Dimensionality Reduction: Autoencoders can be used to reduce the dimensionality of high-dimensional data while preserving its essential characteristics. This is particularly useful in cases where the high dimensionality makes computations slow or the data overfitting occurs.

2. Denoising: By training autoencoders on noisy versions of the data, they can learn to remove noise from the original data, making it cleaner and easier to analyze.

3. Anomaly Detection: The encoder part of the autoencoder can be used to represent the input data in a lower-dimensional space. Any data point that is far from the rest in this space can be considered an anomaly, as it doesn't fit the pattern learned by the autoencoder during training.

4. Generative Modeling: Autoencoders can be used as generative models, allowing them to generate new data that are similar to the original data. This can be useful in various scientific applications, such as creating synthetic data or for exploring the data space.

5. Feature Learning: Autoencoders can learn useful features from raw data, which can then be used as inputs for other machine learning models, improving their performance.

In summary, autoencoders are a powerful tool for scientific data analysis due to their ability to learn the underlying structure of complex data.
"""

# %% [markdown]
"""
## An autoencoder for denoising

In the next cells, we will face a situation in which the quality of the data is rather poor. There is a lot of noise added to the dataset which is hard to handle. We will set up an autoencoder to tackle the task of **denoising**.

First, let's prepare a dataset, which is contains a signal we are interested in and the noise.
"""

# %%

import numpy as np
import torch

np.random.seed(13)
torch.random.manual_seed(12)

# %%
from mnist1d.data import get_dataset_args, make_dataset

# disable noise for a clear reference
clean_config = get_dataset_args()
clean_config.iid_noise_scale = 0
clean_config.corr_noise_scale = 0
clean = make_dataset(clean_config)
cleanX, cleany = clean['x'], clean['y']

# use iid noise only for the time being
arguments = get_dataset_args()
arguments.iid_noise_scale = .1
arguments.corr_noise_scale = 0
data = make_dataset(arguments)

X, y = data['x'], data['y']

# %% [markdown]
"""
Now, let's plot the data which we would like to use.
"""

# %%
import matplotlib.pyplot as plt

f, ax = plt.subplots(2, 5, figsize=(14, 5), sharex=True, sharey=True)

for sample in range(10):
    col = sample % 5
    row = sample // 5
    ax[row, col].plot(X[sample, ...], label="noisy")
    ax[row, col].plot(cleanX[sample, ...], label="clean", color="green")
    label = y[sample]
    ax[row, col].set_title(f"label {label}")
    if row == 1:
        ax[row, col].set_xlabel(f"samples / a.u.")
    if col == 0:
        ax[row, col].set_ylabel(f"intensity / a.u.")
    if col == 4 and row == 0:
        ax[row, col].legend()

f.suptitle("MNIST1D examples")
f.savefig("mnist1d_noisy_first10.svg")

# %% [markdown]
"""
As we can see, the data is filled with jitter. Furthermore, it is interesting to note, that our dataset is still far from trivial. Have a look at all signals which are assigned to label `6`. Could you make them out by eye?

## Designing an autoencoder

The [autoencoder architecture](https://en.wikipedia.org/wiki/Autoencoder) is well illustrated on wikipedia. We reproduce [the image](https://commons.wikimedia.org/wiki/File:Autoencoder_schema.png) by [Michaela Massi](https://commons.wikimedia.org/w/index.php?title=User:Michela_Massi&action=edit&redlink=1) here for convenience:
<div style="display: block;margin-left: auto;margin-right: auto;width: 75%;"><img src="https://upload.wikimedia.org/wikipedia/commons/3/37/Autoencoder_schema.png" alt="autoencoder schematic from wikipedia by Michaela Massi, CC-BY 4.0"></div>

The architecture consists of three parts:
1. **the encoder** on the left: this small network ingests the input data `X` and compresses it into a smaller shape
2. the **code** in the center: this is the "bottleneck" which holds the **latent representation** of your input data
3. **the decoder** on the right: which reconstructs the output from the latent code

The task of the autoencoder is to reconstruct the input as best as possible. This task is far from easy, as the autoencoder is forced to shrink the data into the latent space.
"""

# %%
from utils import count_params


class MyEncoder(torch.nn.Module):

    def __init__(self, nlayers: int = 3, nchannels=16):

        super().__init__()
        self.layers = torch.nn.Sequential()

        for i in range(nlayers-1):
            inchannels = 1 if i == 0 else nchannels
            # convolve and shrink input width by 2x
            self.layers.append(torch.nn.Conv1d(in_channels=inchannels,
                                               out_channels=nchannels,
                                               kernel_size=5,
                                               padding=2,
                                               stride=2))
            self.layers.append(torch.nn.ReLU())

        # convolve and keep input width
        self.layers.append(torch.nn.Conv1d(in_channels=nchannels, out_channels=1,
                                           kernel_size=3, padding=1))
        self.layers.append(torch.nn.ReLU())

        # flatten and add a linear tail
        self.layers.append(torch.nn.Flatten())

    def forward(self, x):

        return self.layers(x)

# %%
enc = MyEncoder()
print(f"constructed encoder with {count_params(enc)} parameters")

# convert input data to torch.Tensor
Xt = torch.from_numpy(X)
# convolutions in torch require an explicit channel to be present in the data
Xt = Xt.unsqueeze(1)
# convert to float
Xt = Xt.float()
# extract only first 8 samples for testing
Xtest = Xt[:8, ...]

latent_h = enc(Xtest)

print(Xtest.shape, Xtest.dtype, "->", latent_h.shape, latent_h.dtype)

# %% [markdown]
"""
The encoder has been constructed. Now, we need to add a decoder object to reconstruct from the latent space.
"""

# %%
class MyDecoder(torch.nn.Module):

    def __init__(self, nlayers: int = 3, nchannels=16):

        super().__init__()
        self.layers = torch.nn.Sequential()

        for i in range(nlayers-1):
            inchannels = 1 if i == 0 else nchannels
            # deconvolve/Upsample and grow input width by 2x
            self.layers.append(torch.nn.ConvTranspose1d(in_channels=inchannels,
                                                        out_channels=nchannels,
                                                        kernel_size=5,
                                                        padding=2,
                                                        stride=2,
                                                        output_padding=1))
            self.layers.append(torch.nn.ReLU())

        # convolve and keep input width
        self.layers.append(torch.nn.Conv1d(in_channels=nchannels, out_channels=1,
                                           kernel_size=3, padding=1))

    def forward(self, x):

        return self.layers(x)

# %%
dec = MyDecoder()
print(f"constructed decoder with {count_params(dec)} parameters")

Xt_prime = dec(latent_h.unsqueeze(1))
assert Xt_prime.shape == Xtest.shape, f"{Xt_prime.shape} != {Xtest.shape}"
print(f"decoder is ready to train!")
