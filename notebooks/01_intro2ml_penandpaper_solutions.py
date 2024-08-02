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
# # Solutions for exercise 01.4
#

# %%
import torch

x = torch.Tensor([[2.]])
y = torch.Tensor([[5.]])

# %% [markdown]
# The central building block of `pytorch` is a `torch.Tensor` object. The API of `Tensor` is very similar to that of a `numpy.ndarray`. That makes it easier to switch between libraries.
#

# %% [markdown]
# ## Our model
#
# We start out by defining the input data and the outputs. We will use the data of Group 1 as an example.
# To define a neural network, the mechanics of pytorch require us to define a class. This class needs to be based of `torch.nn.Module`. Within the class, we have to define the `forward` function which is effectively the forward pass of our model.

# %%


class f_prime_model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        #for more complicated models, this constructor will be rather complicated
        self.hidden0 = torch.nn.Linear(in_features=1, out_features=1)
        self.relu0 = torch.nn.ReLU()
        self.hidden1 = torch.nn.Linear(in_features=1, out_features=1)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        """ forward pass of our model, using x as the input data """
        h = self.hidden0(x)
        h_ = self.relu0(h)
        y_hat_ = self.hidden1(h_)
        y_hat = self.relu1(y_hat_)

        return y_hat


def one_iteration(model,x,y,h0w=4.,h0b=2.,h1w=4.,h1b=2., verbose=False):

    with torch.no_grad():
        model.hidden0.weight.fill_(h0w)
        model.hidden0.bias.fill_(h0b)
        model.hidden1.weight.fill_(h1w)
        model.hidden1.bias.fill_(h1b)

    if verbose:
        print("model parameters before weight update")
        for param in model.named_parameters():
            name, value = param
            print(name, value.item())

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters())
    y_hat = model(x)
    loss = loss_fn(y_hat, y) #loss function computed (computational graph is internally established)
    loss.backward() #backpropagate through loss function
    opt.step() #weight updated step in model.paramters()

    print("model parameters after weight update")
    for param in model.named_parameters():
        name, value = param
        print(name, value.item())

    y_hat_prime = model(x)
    print(f"y_hat {y_hat.item()} -> {y_hat_prime.item()}")

    return model, loss_fn, opt


# %%
print("Group 1")
one_iteration(f_prime_model(),
              x,
              y,
              4.,2.,0.75,.5,verbose=True)
print()
# %%
print("Group 2")
one_iteration(f_prime_model(),
              x,
              y,
              2.,.25,5.,0.,verbose=True)
print()

# %%
print("Group 3")
one_iteration(f_prime_model(),
              x,
              y,
              -4.,2.,0.75,-.5,verbose=True)
print()

# %%
print("Group 4")
one_iteration(f_prime_model(),
              x,
              y,
              2.,-.25,-5,0.,verbose=True)
print()

# %%
print("Group 5")
one_iteration(f_prime_model(),
              x,
              y,
              1.,1.,1.,1.,verbose=True)
print()
