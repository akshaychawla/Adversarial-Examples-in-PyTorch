import numpy as np 
import torch
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
from torchvision import transforms
from tqdm import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # my network is composed of only affine layers 
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)   
        # self.r   = nn.Parameter(data=torch.randn(5,5), requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)
SoftmaxWithXent = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04)


# DATA LOADERS 
def flat_trans(x):
    x.resize_(28*28)
    return x
  
  
if __name__ == '__main__':
  # DEFINE NETWORK
  mnist_transform = transforms.Compose(
                      [transforms.ToTensor(), transforms.Lambda(flat_trans)]
                    )
  traindata = torchvision.datasets.MNIST(root="./mnist", train=True, download=True, transform=mnist_transform)
  trainloader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=2)
  testdata  = torchvision.datasets.MNIST(root="./mnist", train=False, download=True, transform=mnist_transform)
  testloader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=True, num_workers=2)

  # TRAIN 
  for epoch in range(100):

      print("Epoch: {}".format(epoch))
      running_loss = 0.0 
      # import ipdb; ipdb.set_trace()
      for data in tqdm(trainloader):

          # get the inputs 
          inputs, labels = data 
          # wrap them in a variable 
          inputs, labels = Variable(inputs), Variable(labels)
          # zero the gradients 
          optimizer.zero_grad() 

          # forward + loss + backward 
          outputs = net(inputs) # forward pass 
          loss = SoftmaxWithXent(outputs, labels) # compute softmax -> loss 
          loss.backward() # get gradients on params 
          optimizer.step() # SGD update 

          # print statistics 
          running_loss += loss.data[0]

      print('Epoch: {} | Loss: {}'.format(epoch, running_loss/2000.0))

  print ("Finished Training")

  # TEST 
  correct = 0.0 
  total = 0 
  for data in testloader:
      images, labels = data 
      outputs = net(Variable(images))
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0) 
      correct += (predicted == labels).sum()

  print("Accuracy: {}".format(correct/total))

  print ("Dumping weights to disk")
  weights_dict = {} 
  # import ipdb; ipdb.set_trace()
  for param in list(net.named_parameters()):
      print ("Serializing Param", param[0])
      weights_dict[param[0]] = param[1] 
  with open("weights.pkl","wb") as f:
      import pickle 
      pickle.dump(weights_dict, f)
  print ("Finished dumping to disk..")
