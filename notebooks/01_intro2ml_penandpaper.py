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
# A multi-layer perceptron with pen and paper

Multi-layer Perceptrons (MLP) were some of the first Machine Learning (ML) architectures in use.

This notebook will guide you through how to
- perform a forward pass through a very simple MLP
- how to backpropagate through the network
- how to perform the same operations with [pytorch](https://pytorch.org)
"""

# %% [markdown]
r"""
## A FeedForward MLP

Let's recap the essential ingredients of a feedforward neural network:

<div style="display: block;margin-left: auto;margin-right: auto;width: 75%;"><img src="img/01_plain_mlp.svg" alt="plain MLP"></div>

In the above, the network receives an input datum $x$. This is used as input to the first hidden unit $\sigma (\boldsymbol{\omega} \cdot \vec{x} + \vec{\beta})$. The hidden unit consists of 3 ingredients:
- a weight matrix $\boldsymbol{\omega}$
- a bias vector $\vec{\beta}$
- an activation function $\sigma$

The output of the hidden unit will be considered to be the input of the last layer. This in turn produces the output $\vec{\hat{y}}$ of the forward pass. In order to establish a learning process, this prediction $\hat{y}$ will be compared with a label $y$ using the loss the function $\mathcal{L}(y,\hat{y})$.
"""

# %% [markdown]
"""
> **A FeedForward MLP with all details**
>
> For you to dive into all details, here is a pictorial representation of a MLP with all math operations spelled out.
>
> <div style="display: block;margin-left: auto;margin-right: auto;width: 75%;"><img src="img/01_detailed_mlp.svg" alt="detailed MLP"></div>
>
> To read through it, take your hand to block out the right hand part of the network. Then follow the flow of datum $x_0$ through the hidden layers until the output layers. Once you are done following $x_0$, have a look at $x_1$. You will notice that the operations are the same, except the input.
>
> Some comments to the above:
> - the figure omits the display of all matrix elements of $W^{I}$ and $W^{II}$ to remain clear for the reader
> - the last layer does not explicitely state which activation is used. Depending on the use case, this activation function can be a [ReLU](https://en.wikipedia.org/wiki/ReLU) or [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) (for regression) or a [softmax](https://en.wikipedia.org/wiki/Softmax_function) function (for classification).
"""

# %% [markdown]
r"""
**Exercise 01.1**

<div style="display: block;margin-left: auto;margin-right: auto;width: 75%;"><img src="img/01_detailed_mlp.svg" alt="linear MLP with values"></div>

Take pen and paper or a digital equivalent. Mark the path of all computations which result in $y_{1}$! Write down the weight matrix elements which will be used to compute this output.
"""

# %% [markdown] jupyter={"source_hidden": true}
# **Solution 01.1**
#
# <div style="display: block;margin-left: auto;margin-right: auto;width: 75%;"><img src="img/01_detailed_mlp.svg" alt="linear MLP with values"></div>
#
# All weight matrix elements will be used except the following 3 elements $\boldsymbol{\omega}_{00}^{II}, \boldsymbol{\omega}_{10}^{II}, \boldsymbol{\omega}_{10}^{II}$ as their are required to calculate $\hat{y}_0$. All results of the hidden layer will be used as inputs to the last layer, that is why all elements of $W^{I}$ are part of the calculation.


# %% [markdown]
# ## A simple MLP
#
# To start off this exericse, we want to construct a very simple MLP with one input unit, one hidden unit and one ouput unit. We will keep everything lightweight and one dimensional.
#
# <div style="display: block;margin-left: auto;margin-right: auto;width: 75%;"><img src="img/01_1D_mlp.svg" alt="1D MLP"></div>
#
# In the above, the network receives an input datum $x$. This is used as input to the first hidden unit $\sigma (\omega \cdot x' + \beta)$. The hidden unit consists of 3 ingredients:
# - the weight $\omega$
# - the bias $\beta$
# - the activation function $\sigma$
#
# The output of the hidden unit will be considered to be the output of the forward pass. In order to establish a learning process, this prediction $y'$ will be compared with a label $y$ using the loss the function $\mathcal{L}(y,y')$.

# %% [markdown]
r"""
**Exercise 01.2**

<div style="display: block;margin-left: auto;margin-right: auto;width: 75%;"><img src="img/01_1D_mlp_filled.svg" alt="linear MLP with values"></div>

Take pen and paper. Compute a full forward pass using the following values:

- input $x = 2$
- weight $\omega = .5$
- bias $\beta = 1$
- label $y = 1.5$

Use the [ReLU function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) for $\sigma$ and the mean squared error ([MSE](https://en.wikipedia.org/wiki/Mean_squared_error)) for the loss function.
"""

# %% [markdown] jupyter={"source_hidden": true}
# > **Solution 01.2**
# > 1. Compute $\omega \cdot x' + \beta$: We obtain `2`.
# > 2. Apply the ReLU: We obtain `y'=2` again (2 is larger than 0 and hence $f_{ReLU}(x=2)=2$).
# > 3. Compute the loss: $\mathcal{L}(y,y') = (y-y')^2 = (2 - 1.5)^2 = \frac{1}{4}$ 

# %% [markdown]
# # Supervised Learning
#
# Given a dataset $\mathcal{D} = \{\langle \vec{x}_i, y_i\rangle \dots \}$ with input data $x \in \mathbb{R}^n$ and labels $y \in \mathbb{R}^{2}$, we would like to train a model $f$ with parameters $\varphi = \{ \boldsymbol{\omega}^{I}, \vec{\beta}^{I}, \boldsymbol{\omega}^{II}, \vec{\beta}^{II} \}$ such that:
#
# $$ \vec{\hat{y}} = \hat{f}(\vec{x}|\varphi) $$
#
# To obtain a good estimate of $f$, we alter the weights of our model $\varphi$. To do this, the optimisation is performed using a loss function $\mathcal{L}$ to obtain an optimal set of weights $\varphi$ by minimizing $\mathcal{L}$:
#
# $$ \varphi \approx \text{argmin}_{\varphi} \mathcal{L}( \vec{y}, \vec{\hat{y}} = f(\vec{x} | \varphi) ) $$.
#
# The optimisation is performed using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). In this optimisation scheme, we update the parameters $\varphi$ in a step-by-step fashion. After being randomly initialized, the parameters are updated at step $s$ using the weight update rule:
#
# $$ \varphi_{t+1} = \varphi_{t} + \alpha \nabla_{\varphi} \mathcal{L}( \vec{y}, f(\vec{x}|\varphi_{s})) $$
#
# The above equation is called __weight update rule__. Here, the free parameter $\alpha$ is also known as the __learning rate__.

# %% [markdown]
# ## Backpropagation
#
# The central part of gradient descent is how to obtain the value of $\nabla_{\varphi}\mathcal{L}$. This is performed by applying the chain rule of differentiation from the back of the model to the front. For the sake of simplicity, let's say we only have part of a model: $f'(x|\varphi) = \omega \cdot x + \beta = \hat{y}$. To compute the gradient of this simple model $f'$, we have to start from the loss function $\mathcal{L}(y,\hat{y})= (y-\hat{y})^2$:
#
# $$ \nabla \mathcal{L} = \frac{\partial\mathcal{L}}{\partial\varphi} $$
#
# The gradient $\nabla$ is calculated for each weight or bias term independently (as they are also used independently during the forward pass).
#
# $$ \nabla_w \mathcal{L} = \frac{\partial\mathcal{L}}{\partial \omega} $$
#
# We can now go forward and apply the chain rule to the differential above:
#
# $$ \nabla_w \mathcal{L} = \frac{\partial\mathcal{L}}{\partial \omega} = \frac{\partial\mathcal{L}}{\partial \hat{y}}\frac{\partial \hat{y}}{\partial \omega}$$
#
# We can now move forward and evaluate each term of the chain rule expression:
#
# $$ \frac{\partial\mathcal{L}}{\partial \hat{y}} = \frac{\partial (y-\hat{y})^2}{\partial \hat{y}} = 2\cdot(y-\hat{y})\cdot(-1)$$
# $$ \frac{\partial \hat{y}}{\partial \omega} = \frac{\partial (\omega \cdot x + \beta)}{\partial \omega} = x $$
#
# If we now want to compute the value of the gradient, we would have to input concrete numbers to finally apply the weight update rule.

# %% [markdown]
r"""
**Exercise 01.3**

Take pen and paper or a digital equivalent. Compute the gradient for $\beta$, $\nabla_\beta \mathcal{L}$ of our stub model $f'$!
"""

# %% [markdown] jupyter={"source_hidden": true}
r"""
> **Solution 01.3**
> 1. Apply the chain rule to our stub network:
> $$\frac{\partial\mathcal{L}}{\partial \beta} = \frac{\partial\mathcal{L}}{\partial \hat{y}}\frac{\partial \hat{y}}{\partial \beta}$$
> 2. Evaluate each subterm:
> $$ \frac{\partial\mathcal{L}}{\partial \hat{y}} = \frac{\partial (y-\hat{y})^2}{\partial \hat{y}} = 2\cdot(y-\hat{y})\cdot(-1)$$
> $$ \frac{\partial \hat{y}}{\partial \beta} = \frac{\partial (\omega \cdot x + \beta)}{\partial \beta} = 1 $$
> 3. Put everything together:
> $$ \nabla_{\beta} \mathcal{L} = \frac{\partial\mathcal{L}}{\partial \hat{y}}\frac{\partial \hat{y}}{\partial \beta} = 2\cdot(y-\hat{y})\cdot(-1) \cdot 1 = -2\cdot(y-\hat{y})$$
"""

# %% [markdown]
# ## Putting it all together
#
# We now want to perform some stochastic gradient based optimisation by hand to conclude this pen&paper part of the exercise.
#
# <div style="display: block;margin-left: auto;margin-right: auto;width: 75%;"><img src="img/01_1D_2hidden_mlp.svg" alt="1D MLP"></div>
#
# To make this worth your time, we split the audience into groups. You will get distinct data sets. Please calculate the one iteration of gradient descent for your parameter set and check if gradient descent gets you closer to the label.
#
# **Exercise 01.4**
#
# Groups:
# 1. $x=2$, $\omega_0=4$, $\beta_0=2$, $\omega_1=.75$, $\beta_1=.5$, $y=5$
# 2. $x=2$, $\omega_0=2$, $\beta_0=.25$, $\omega_1=5$, $\beta_1=0$, $y=5$
# 3. $x=2$, $\omega_0=-4$, $\beta_0=2$, $\omega_1=.75$, $\beta_1=-.5$, $y=5$
# 4. $x=2$, $\omega_0=2$, $\beta_0=-.25$, $\omega_1=-5$, $\beta_1=0$, $y=5$
# 5. $x=2$, $\omega_0=1$, $\beta_0=1$, $\omega_1=1$, $\beta_1=1$, $y=5$

# %% [markdown]
# # Pytorch
#
# To par our pen & paper exercises with an introduction to one central library, we want to start off and compute the last exercise **01.4** in pytorch. This will also give us a chance to the check the results. On top, the central mechanisms of pytorch can be exposed which directly leads us into further topics of classification and more.
#

# %% [markdown]
# ## Our dataset
#
# We start out by defining the input data and the outputs. We will use the data of Group 1 as an example.

# %%
import torch

x = torch.Tensor([[2.]])
y = torch.Tensor([[5.]])

# %% [markdown]
"""
The central building block of `pytorch` is a `torch.Tensor` object. The API of `Tensor` is very similar to that of a `numpy.ndarray`. That makes it easier to switch between libraries.
"""

# %% [markdown]
# ## Our model
#
# We start out by defining the input data and the outputs. We will use the data of Group 1 as an example.
# To define a neural network, the mechanics of pytorch require us to define a class. This class needs to be derived from `torch.nn.Module`. Within the class, we have to define the `forward` function which is effectively the forward pass of our model and provide a constructor method `__init__`.

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

# %%
# let's instantiate the model
model = f_prime_model()

# we want to start at fixed values; as pytorch will automatically hook the model class up to it's capabilities to compute the gradient internally, we have to forbid this from happening here using the torch.no_grad context manager:

with torch.no_grad():
    model.hidden0.weight.fill_(4.)
    model.hidden0.bias.fill_(2.)
    model.hidden1.weight.fill_(0.75)
    model.hidden1.bias.fill_(0.5)

# %%
# we can convince ourselves that we achieved to set the internal values of our model
for param in model.named_parameters():
    name, value = param
    print(name, value.item())

# %%
# A first feed forward pass can be invoced like this! Note the syntax model(x) effectively calls model.forward(x).

y_hat = model(x)
print(y_hat.item(), y.item())

# %%
# Now we want to set up our loss function. As we are interested in regression, we use the mean squared error [MSE](https://en.wikipedia.org/wiki/Mean_squared_error).
loss_fn = torch.nn.MSELoss()

# %%
# Next, we want to set up the optimizer to perform one learning step. The constructor requires to provide the model parameters. This is the "bridge" to connect model with optimizer for the later optimization routine.
opt = torch.optim.SGD(model.parameters(),lr=0.01)

# %%
# finally we perform one feedforward pass and one backward pass

y_hat = model(x)
loss = loss_fn(y_hat, y) #loss function computed (computational graph is internally established)
loss.backward() #backpropagate through loss function
opt.step() #weight updated step in model.paramters()

# %%
# let's first check if the model paramters changed

for i in model.named_parameters():
    name, value = i
    print(name, value.item())

# indeed they did change!

# %%
# let's also check if the prediction changed
print(model(x).item())
# indeed it did change too!
