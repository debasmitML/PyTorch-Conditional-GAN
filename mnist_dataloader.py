import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


transform = transforms.Compose([
    
     
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5),std=(0.5))
            
        ])

def Dataset():

    train_dataset = MNIST(root = './mnist_data/', transform = transform,download = True, train=True)
    test_dataset = MNIST(root = './mnist_data/', transform = transform,download = False, train = False)

    
    
    return train_dataset, test_dataset