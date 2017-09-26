import torchvision 
import torch 
from torch.autograd import Variable 
from torchvision import transforms 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

def flat_trans(x):
    x.resize_(28*28)
    return x
mnist_transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(flat_trans)]
                  )
testdata  = torchvision.datasets.MNIST(root="./mnist", train=False, download=True, transform=mnist_transform)
testloader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=True, num_workers=2)

images, labels = [], []
for idx, data in enumerate(testloader):
    x, y = data 
    images.append(x.numpy())
    labels.append(y.numpy())
    if idx == 4999:
        import ipdb; ipdb.set_trace()
        break 

import ipdb; ipdb.set_trace()
with open("5k_samples.pkl", "w") as f: 
    import cPickle 
    data_dict = { "images":images, "labels": labels}
    cPickle.dump(data_dict, f)

