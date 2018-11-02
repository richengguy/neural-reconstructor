# Neural Reconstructor
This is a small [PyTorch](https://pytorch.org/) experiment that was mainly meant
to get familiar with the library.  As a side-effect, it was also a way to look
at how an objective function could be used to train a network to both recognize
a class, given an image, as well as a representation of that class.  The result
is something that is, more or less, an autoencoder but with an objective
function that minimizes both the classification error as well as the
reconstruction error.  This most likely has been done elsewhere but a thorough
literature review was out of the scope of this work.

## Environment Setup
The demo is contained within a 
[Jupyter notebook](https://jupyter-notebook.readthedocs.io/en/stable/).  The 
`network.py` file contains the PyTorch modules that implement the actual
networks.  The best way to setup the environment is to use
[conda](https://conda.io/docs/), particularly as it will pull down optimized
versions of Numpy.

Initializing the environment is done via

```
$ conda env create -f environment.yml
```

Once the command finishes, the environment is activated with

```
$ conda activate neural-reconstructor
```