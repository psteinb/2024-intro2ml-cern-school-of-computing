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

# %%

import numpy as np
import torch

from utils import count_params

np.random.seed(13)
torch.random.manual_seed(12)

# %%
from mnist1d.data import get_dataset_args, make_dataset

# disable noise for a clear reference
clean_config = get_dataset_args()
clean_config.iid_noise_scale = 0
clean_config.corr_noise_scale = 0
clean_config.seed = 40
clean = make_dataset(clean_config)
cleanX, cleany = clean["x"], clean["y"]

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
    ax[row, col].plot(cleanX[sample, ...], label="clean", color="green")
    label = cleany[sample]
    ax[row, col].set_title(f"label {label}")
    if row == 1:
        ax[row, col].set_xlabel(f"samples / a.u.")
    if col == 0:
        ax[row, col].set_ylabel(f"intensity / a.u.")
    if col == 4 and row == 0:
        ax[row, col].legend()

f.suptitle("MNIST1D examples")
f.savefig("mnist1d_cleanonly_first10.svg")

# %%


class MyEncoder(torch.nn.Module):
    def __init__(self, nlayers: int = 3, nchannels=16):
        super().__init__()
        self.layers = torch.nn.Sequential()

        for i in range(nlayers - 1):
            inchannels = 1 if i == 0 else nchannels
            # convolve and shrink input width by 2x
            self.layers.append(
                torch.nn.Conv1d(
                    in_channels=inchannels,
                    out_channels=nchannels,
                    kernel_size=5,
                    padding=2,
                    stride=2,
                )
            )
            self.layers.append(torch.nn.ReLU())

        # convolve and keep input width
        self.layers.append(
            torch.nn.Conv1d(
                in_channels=nchannels, out_channels=1, kernel_size=3, padding=1
            )
        )

        # flatten and add a linear tail
        self.layers.append(torch.nn.Flatten())

    def forward(self, x):
        # convolutions in torch require an explicit channel dimension to be
        # present in the data in other words:
        # inputs of size (nbatch, 40) do not work,
        # inputs of size (nbatch, 1, 40) do work
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=1)
        return self.layers(x)


# %%
class MyDecoder(torch.nn.Module):
    def __init__(self, nlayers: int = 3, nchannels=16):
        super().__init__()
        self.layers = torch.nn.Sequential()

        for i in range(nlayers - 1):
            inchannels = 1 if i == 0 else nchannels
            # deconvolve/Upsample and grow input width by 2x
            self.layers.append(
                torch.nn.ConvTranspose1d(
                    in_channels=inchannels,
                    out_channels=nchannels,
                    kernel_size=5,
                    padding=2,
                    stride=2,
                    output_padding=1,
                )
            )
            self.layers.append(torch.nn.ReLU())

        # convolve and keep input width
        self.layers.append(
            torch.nn.Conv1d(
                in_channels=nchannels, out_channels=1, kernel_size=3, padding=1
            )
        )

    def forward(self, x):
        # convolutions in torch require an explicit channel dimension to be
        # present in the data in other words:
        # inputs of size (nbatch, 40) do not work,
        # inputs of size (nbatch, 1, 40) do work
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=1)

        return self.layers(x)


# %%
class MyAutoencoder(torch.nn.Module):
    def __init__(self, nlayers: int = 3, nchannels=16):
        super().__init__()

        self.enc = MyEncoder(nlayers, nchannels)
        self.dec = MyDecoder(nlayers, nchannels)

    def forward(self, x):
        # construct the latents
        h = self.enc(x)

        # perform reconstruction
        x_prime = self.dec(h)

        return x_prime


# %% [markdown]
"""
## Training an autoencoder

Training the autoencoder works in the same line as training for regression from the last episode.

1. create the dataset
2. create the loaders
3. setup the model
4. setup the optimizer
5. loop through epochs
"""

# %%
from torch.utils.data import DataLoader
from utils import MNIST1D

training_data = MNIST1D(mnist1d_args=clean_config)
test_data = MNIST1D(mnist1d_args=clean_config, train=False)

nsamples = len(training_data) + len(test_data)
assert nsamples == 4000, f"number of samples for MNIST1D is not 4000 but {nsamples}"

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

autoemodel = MyAutoencoder()
learning_rate = 1e-3
max_epochs = 30
log_every = 5

optimizer = torch.optim.AdamW(autoemodel.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()  # our loss function


# %%
# write the training loop
def train_autoencoder(
    model, opt, crit, train_dataloader, test_dataloader, max_epochs, log_every=5
):
    results = {"train_losses": [], "test_losses": []}
    ntrainsteps = len(train_dataloader)
    nteststeps = len(test_dataloader)
    train_loss, test_loss = torch.zeros((ntrainsteps,)), torch.zeros((nteststeps,))

    for epoch in range(max_epochs):
        # perform training for one epoch
        for idx, (X, _) in enumerate(train_dataloader):
            # forward pass
            X_prime = model(X)

            # compute loss
            loss = crit(X_prime, X)

            # compute gradient
            loss.backward()

            # apply weight update rule
            opt.step()

            # set gradients to 0
            opt.zero_grad()

            train_loss[idx] = loss.item()

        for idx, (X_test, _) in enumerate(test_dataloader):
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


print(f"Initialized autoencoder with {count_params(autoemodel)} parameters")
results = train_autoencoder(
    autoemodel,
    optimizer,
    criterion,
    train_dataloader,
    test_dataloader,
    max_epochs,
    log_every,
)
# %%
f, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(results["train_losses"], color="b", label="train")
ax[0].plot(results["test_losses"], color="orange", label="test")
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("avergage MSE Loss / a.u.")
ax[0].set_yscale("log")
ax[0].set_title("Loss")
ax[0].legend()


index = 0
# perform prediction again
last_x, last_y = test_data[index]
last_x_prime = autoemodel(last_x.unsqueeze(0))

# prepare tensors for plotting
last_in = last_x.detach().squeeze().numpy()
last_out = last_x_prime.detach().squeeze().numpy()

ax[1].plot(last_in, color="b", label="test input")
ax[1].plot(last_out, color="orange", label="test prediction")
ax[1].set_xlabel("samples / a.u.")
ax[1].set_ylabel("intensity / a.u.")
ax[1].set_title(f"Conv-based Autoencoder, label = {last_y.detach().item()}")
ax[1].legend()

f.savefig("representationlearning-autoencoder-loss.svg")

# %% [markdown]
"""
# Representation Learning

Effective Machine Learning is often about finding a good and flexible model that can represent high-dimensional data well. The autoencoder can be such an architecture depending on its design and the input data. In practice, the community has started to use the latent representation for all kinds of applications. But you should be aware, that the created representations can be task specific.

## Classifying MNIST1D

Similar to [MNIST](https://yann.lecun.com/exdb/mnist/), `mnist1d` can be used for the task of classification. In other words, given an input sequence, we only want to predict the class label `[0,1,...,9]` that the image belongs to. Classification has been one of the driving forces behind progress in machine learning since [ImageNet 2012]() - for better or worse. In science, classification is used rarely.
"""


# %%
# taken from https://github.com/greydanus/mnist1d/blob/dc46206f1e1ad7249c96e3042efca0955a6b5d35/notebooks/models.py#L36C1-L54C65
class ConvBase(torch.nn.Module):
    def __init__(self, output_size, channels=25, linear_in=10):
        super(ConvBase, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, channels, 5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv1d(channels, channels, 3, stride=1, padding=1)
        self.linear = torch.nn.Linear(
            linear_in * channels, output_size
        )  # flattened channels -> 10 (assumes input has dim 50)

    def forward(self, x, verbose=False):  # the print statements are for debugging
        x = x.view(-1, 1, x.shape[-1])
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = h3.view(h3.shape[0], -1)  # flatten the conv features
        return self.linear(h3)  # a linear classifier goes on top


# %%
from sklearn.metrics import accuracy_score as accuracy


def train_classifier(
    model, opt, crit, train_dataloader, test_dataloader, max_epochs, log_every=5
):
    results = {
        "train_losses": [],
        "test_losses": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }
    ntrainsteps = len(train_dataloader)
    nteststeps = len(test_dataloader)
    train_loss, test_loss = torch.zeros((ntrainsteps,)), torch.zeros((nteststeps,))
    train_acc, test_acc = np.zeros((ntrainsteps,)), np.zeros((nteststeps,))

    for epoch in range(max_epochs):
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
            train_acc[idx] = accuracy(
                y.detach().cpu().numpy(), y_hat.argmax(-1).cpu().numpy()
            )

        for idx, (X_test, y_test) in enumerate(test_dataloader):
            y_hat_test = model(X_test)
            loss_ = crit(y_hat_test, y_test)
            test_loss[idx] = loss_.item()
            test_acc = accuracy(
                y_test.detach().cpu().numpy(), y_hat_test.argmax(-1).cpu().numpy()
            )

        results["train_losses"].append(train_loss.mean())
        results["test_losses"].append(test_loss.mean())
        results["train_accuracy"].append(np.mean(train_acc))
        results["test_accuracy"].append(np.mean(test_acc))

        if epoch % log_every == 0 or (epoch + 1) == max_epochs:
            print(
                f"{epoch+1:02.0f}/{max_epochs} :: training loss {train_loss.mean():03.4f}; test loss {test_loss.mean():03.4f} "
                f"training acc {np.mean(train_acc):01.4f}; test acc {np.mean(test_acc):01.4f}"
            )
    return results


# %%
# we reuse the dataloaders from above
classmodel = ConvBase(10, channels=32)
print(f"Initialized ConvBase model with {count_params(classmodel)} parameters")
classopt = torch.optim.AdamW(classmodel.parameters(), lr=1e-3)
classcrit = torch.nn.CrossEntropyLoss()

classifier_results = train_classifier(
    classmodel, classopt, classcrit, train_dataloader, test_dataloader, max_epochs=30
)

# %%
f, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(classifier_results["train_losses"], color="b", label="train")
ax[0].plot(classifier_results["test_losses"], color="orange", label="test")
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("avergage MSE Loss / a.u.")
ax[0].set_yscale("log")
ax[0].set_title("Loss")
ax[0].legend()

ax[1].plot(classifier_results["train_accuracy"], color="b", label="train")
ax[1].plot(classifier_results["test_accuracy"], color="orange", label="test")
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("avergage Accuracy / a.u.")
ax[1].set_title("Accuracy")
ax[1].legend()

f.savefig("representationlearning-classifier-loss.svg")

# %% [markdown]
r"""
We have trained two networks:
- an autoencoder on a reconstruction task
- a classifier on a classification task

In practice, users are often interested in using the embeddings of either. The question, we want to answer now: are the embeddings the same?

At this point, we have to honor the fact, that we are dealing with a 10-dim space in either case. Thus, we have to choose a good visualisation method (or any other method to check) how similar, the embeddings actually are.

*Exercise 05.2*

Perform the study above by fitting a 2-component PCA from `sklearn` on the embedding spaces of the test set! Fix the errors in the visible code snippet first. Then move on to visualize the first 2 components of the PCA.

Bonus: If you feel like it, feel free to experiment with other techniques than PCA.
"""

# %%
# disable autodiff computations
classmodel.eval()
autoemodel.eval()

alldata_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
alltest_x, alltest_y = next(iter(alldata_loader))

allembed_class = ...
allembed_autoe = ...

assert ...


# %% jupyter={"source_hidden": true}
classmodel.eval()
autoemodel.eval()

alldata_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
alltest_x, alltest_y = next(iter(alldata_loader))

allembed_class = classmodel(alltest_x)
allembed_autoe = autoemodel.enc(alltest_x)

assert allembed_autoe.shape == allembed_class.shape


# %% jupyter={"source_hidden": true}
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_class = pca.fit(allembed_class.detach().numpy()).transform(
    allembed_class.detach().numpy()
)
X_autoe = pca.fit(allembed_autoe.detach().numpy()).transform(
    allembed_autoe.detach().numpy()
)

assert X_class.shape == X_autoe.shape

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
lw = 2

ax[0,0].scatter(X_class[..., 0], X_class[..., 1])
ax[0,0].set_title("PCA of classifier embeddings")

ax[0,1].scatter(X_autoe[..., 0], X_autoe[..., 1])
ax[0,1].set_title("PCA of autoencoder embeddings")

ax[1,0].scatter(X_class[..., 0], X_class[..., 1], c = alltest_y.detach().numpy())
ax[1,0].set_title("PCA of classifier embeddings")

ax[1,1].scatter(X_autoe[..., 0], X_autoe[..., 1], c = alltest_y.detach().numpy())
ax[1,1].set_title("PCA of autoencoder embeddings")

fig.savefig("representationlearning-pca-comparison.svg")
# %% [markdown]
r"""
From the above we learn, that the geometries which each of the two architectures populate in the embedding space tend to be quite different. Hence, the effect of this must be taken into account in practice. Moreover, we also see how clearly different either model differentiates the dataset.
"""
