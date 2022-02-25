# %% [markdown]
# # Chapter 1
# ### Setup
# %% [markdown]
# Add local source files to the path.
# %%
import sys
sys.path.insert(0, "../src")
# %% [markdown]
# Import modules.
# %%
import mnist_loader
import network
import mnist_average_darkness
import mnist_svm
# %% [markdown]
# Load the data.
# %%
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# %% [markdown]
# ### Simple neural networks
# Run our first network.
# %%
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
# %% [markdown]
# Increase the number of hidden layers. This may or may not help.
# %%
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
# %% [markdown]
# Decrease the learning rate.
# %%
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 0.001, test_data=test_data)
# %% [markdown]
# Increase the learning rate.
# %%
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 100.0, test_data=test_data)
# %% [markdown]
# ### Benchmarks
# A simple heuristic: classification by darkness.
# %%
mnist_average_darkness.main()
# %% [markdown]
# A *support vector machine (SVM)* with default settings.
# %%
mnist_svm.svm_baseline()
# %% [markdown]
# ### Exercise
# A network without hidden layers - to be played around with...
# %%
net = network.Network([784, 10])
net.SGD(training_data, 50, 10, 1.0, test_data=test_data)