import torch

class Classifier(torch.nn.Module):
    '''The classification network.

    Given an image, this will attempt to classify it into one of ten potential
    classes.  The output is represented using one-hot encoding, meaning that
    this takes a 784-vector and projects it down onto a 10-vector.
    '''
    def __init__(self):
        super().__init__()

        self.input = torch.nn.Linear(784, 256)
        self.hidden = torch.nn.Linear(256, 16)
        self.output = torch.nn.Linear(16, 10)

    def forward(self, x):
        '''Apply a forward pass of the network.'''
        x = torch.sigmoid(self.input(x))
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


class Generator(torch.nn.Module):
    '''The generator network.

    This takes in a 10-vector, which is the one-hot encoding for a set of ten
    classes, and generates a 784-vector.  The idea is that the network will
    generate an example of a particular class when given its representation.

    It is a mirror of the classifier network.
    '''
    def __init__(self):
        super().__init__()

        self.input = torch.nn.Linear(10, 16)
        self.hidden = torch.nn.Linear(16, 256)
        self.output = torch.nn.Linear(256, 784)

    def forward(self, x):
        '''Apply a forward pass of the network.'''
        x = torch.sigmoid(self.input(x))
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


class Reconstructor(torch.nn.Module):
    '''Identify a class from an image and show its representation.

    This is a composition of two separate networks, a classifier and a
    generator.  The classifier will attempt to determine what class it's
    provided while the generator will try to figure out what the class looks
    like.
    '''
    def __init__(self):
        super().__init__()
        self.classifier = Classifier()
        self.generator = Generator()

    def forward(self, x):
        '''Apply a forward pass of the network.

        Returns
        -------
        encoding : torch.Tensor
            the one-hot encoding for the particular class
        appearance : torch.Tensor
            what the network thinks the class looks like
        '''
        encoding = self.classifier(x)
        appearance = self.generator(encoding)
        return encoding, appearance
