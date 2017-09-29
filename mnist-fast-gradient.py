import numpy as np 
import torch
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
from torchvision import transforms
from tqdm import *
import matplotlib.pyplot as plt
import pickle 
import random 

# DEFINE NETWORK
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # my network is composed of only affine layers 
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)   

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def classify(self, x):
        outputs = self.forward(x)
        outputs = outputs / torch.norm(outputs) 
        max_val, max_idx = torch.max(outputs, 1)
        return int(max_idx.data.numpy()), float(max_val.data.numpy()) 
    
net = Net()
print(net)
SoftmaxWithXent = nn.CrossEntropyLoss()

# LOAD PRE-TRAINED WEIGHTS 
weights_dict = {} 
with open("weights.pkl", "r") as f:
    weights_dict = pickle.load(f)
for param in net.named_parameters():
    if param[0] in weights_dict.keys():
        print "Copying: ", param[0]
        param[1].data = weights_dict[param[0]].data 
print "Weights Loaded!"

# Example 6 
with open("example_6.trch","r") as f: 
    x = torch.load(f) # load "6"
y = torch.LongTensor(np.array([3])) # target label : 2

# Wrap x as a variable 
x = Variable(x, requires_grad=True)
y = Variable(y, requires_grad=False)

# run forward pass 
outputs = net(x)
loss = SoftmaxWithXent(outputs, y)
loss.backward() # obtain gradients on x

# Add perturbation
epsilon = 0.25 
print "Epsilon:", epsilon
x_old = x.data
x_g   = torch.sign(x.grad.data)
# import ipdb; ipdb.set_trace()
adv_example = torch.clamp(x_old + epsilon * x_g, 0, 1) 

# Check classification
orig = net.classify(x)
new  = net.classify(Variable(adv_example))
print "Orig: {} , New: {}".format(orig, new)

    
