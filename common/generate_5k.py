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
  
if __name__ == '__main__':
  mnist_transform = transforms.Compose(
                      [transforms.ToTensor(), transforms.Lambda(flat_trans)]
                    )
  testdata  = torchvision.datasets.MNIST(root="./mnist", train=False, download=True, transform=mnist_transform)
  testloader = torch.utils.data.DataLoader(testdata, batch_size=100, shuffle=True, num_workers=1)

  images, labels = [], []
  for idx, data in enumerate(testloader):

      x_lots, y_lots = data 
      for x,y in zip(x_lots, y_lots):
          images.append(x.numpy())
          labels.append(y)

      if idx==49:
          break

  # import ipdb; ipdb.set_trace()
  with open("5k_samples.pkl", "wb") as f: 
      import pickle
      data_dict = { "images":images, "labels": labels}
      pickle.dump(data_dict, f)

