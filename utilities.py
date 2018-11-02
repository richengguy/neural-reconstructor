import pathlib

import torch
import torchvision


__dataset = {
    'mnist': torchvision.datasets.MNIST,
    'fashion': torchvision.datasets.FashionMNIST
}


def load_dataset(name, batch_size):
    '''Loads one of the MNIST datasets.

    Parameters
    ----------
    name : str
        the data set to load; may be either ``mnist`` or ``fashion``
    batch_size : int
        the batch size used by the dataset
    
    Returns
    -------
    training_set : torch.utils.data.DataLoader
        the training dataset in a torch-friendly format
    testing_set : torch.utils.data.DataLoader
        the testing dataset in a torch-friendly format
    '''
    path = (pathlib.Path('.datasets') / name).resolve().as_posix()

    # Load the training and validation datasets
    tfrm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda tensor: tensor.squeeze())
    ])
    data_train = __dataset[name](path, download=True, train=True,
                                 transform=tfrm)
    data_test = __dataset[name](path, train=False, transform=tfrm)

    training_set = torch.utils.data.DataLoader(data_train,
                                               batch_size=batch_size,
                                               shuffle=True)
    testing_set = torch.utils.data.DataLoader(data_test, batch_size=batch_size,
                                              shuffle=False)

    return training_set, testing_set