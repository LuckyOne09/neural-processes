import glob
import numpy as np
import torch
import os
from math import pi
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from lib.datasets.dataProcessing import readFromCSV

class FaceFeatureData(Dataset):
    """
    Dataset of face feature vectors extracted from FG-NET by FaceNet dimension reduction.
    size of pair (x,y): size of x is 2048, size of y is 1

    Parameters
    ----------

    featureVectors:
        each of featureVectors is (x,y)
        x.size():
        data from extracted feature vector pre-processed by FaceNet

    num_samples: num of people in dataset

    num_points: num_of_images of each people

    """

    def __init__(self,num_of_people=82,num_of_images=18,index = None):
        self.num_samples = num_of_people
        self.num_points = num_of_images
        self.x_dim = 2048  # x and y dim are fixed for this dataset.
        self.y_dim = 1
        self.indexing = index

        self.featureVectors = []
        filePath = r'D:\PycharmProjects\ANP\neural-processes-oxford\FeatureVector'
        csvs = os.listdir(filePath)
        FeatureCSVs = map(lambda x: os.path.join(filePath, x), csvs)
        # featureVectors size():
        # (#people(82),#image of each person(18),#features in each image(2048))
        for idx, root_dir in enumerate(FeatureCSVs):
            ageVecotr, featureVector = readFromCSV(root_dir)
            ageTensor = torch.FloatTensor(ageVecotr).unsqueeze(1)
            featureTensor = torch.FloatTensor(featureVector)
            if index is not None:
                if idx == index:
                    self.featureVectors.append((featureTensor,ageTensor))
            else:
                self.featureVectors.append((featureTensor, ageTensor))

    def __getitem__(self, index):
        return self.featureVectors[index]

    def __len__(self):
        #self.numsample == len(self.featureVectors)
        if self.indexing is not None:
            return 1
        return self.num_samples

class FaceFeatureTestData(Dataset):
    """
    Dataset of face feature vectors seperated from FG-NET by FaceNet dimension reduction for testint
    size of pair (x,y): size of x is 2048, size of y is 1
    """

    def __init__(self,testFilePath = r'D:\PycharmProjects\ANP\neural-processes-oxford\TestFeatureVector'):

        self.x_dim = 2048  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        csvs = os.listdir(testFilePath)
        FeatureCSVs = map(lambda x: os.path.join(testFilePath, x), csvs)
        self.testVectors = []
        for root_dir in FeatureCSVs:
            ageVecotr, featureVector = readFromCSV(root_dir)
            ageTensor = torch.FloatTensor(ageVecotr).unsqueeze(1)
            featureTensor = torch.FloatTensor(featureVector)
            self.testVectors.append((featureTensor, ageTensor))

    def __getitem__(self, index):
        return self.testVectors[index]

    def __len__(self):
        #self.numsample == len(self.featureVectors)
        return len(self.testVectors)


class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class SineDataForMTL(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are specified
    Another verison of SineData for MTL purpose

    Parameters
    ----------

    """

    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


def mnist(batch_size=16, size=28, path_to_data='../../mnist_data'):
    """MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def celeba(batch_size=16, size=32, crop=89, path_to_data='../celeba_data',
           shuffle=True):
    """CelebA dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image.

    crop : int
        Size of center crop. This crop happens *before* the resizing.

    path_to_data : string
        Path to CelebA data files.
    """
    transform = transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    celeba_data = CelebADataset(path_to_data,
                                transform=transform)
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=shuffle)
    return celeba_loader


class CelebADataset(Dataset):
    """CelebA dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        path_to_data : string
            Path to CelebA data files.

        subsample : int
            Only load every |subsample| number of images.

        transform : torchvision.transforms
            Torchvision transforms to be applied to each image.
        """
        self.img_paths = glob.glob(path_to_data + '/*.jpg')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = Image.open(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0
