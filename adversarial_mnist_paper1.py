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

# DEFINE NETWORK
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # my network is composed of only affine layers 
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)   
        self.r   = nn.Parameter(data=torch.randn(1,784)/1000.0, requires_grad=True) # really small initial values

    def forward(self, x):
        x = F.relu(self.fc1(x+self.r))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
print(net)
SoftmaxWithXent = nn.CrossEntropyLoss()

# OPTIMIZE FOR "r" 
optimizer = optim.SGD(params=[net.r], lr=0.001)

# LOAD PRE-TRAINED WEIGHTS 
weights_dict = {} 
with open("weights.pkl", "r") as f:
    import pickle 
    weights_dict = pickle.load(f)
for param in net.named_parameters():
    if param[0] in weights_dict.keys():
        print "Copying: ", param[0]
        param[1].data = weights_dict[param[0]].data 
print "Weights Loaded!"


# Load a single MNIST example
with open("example_6.trch", "r") as f:
    x_6 = torch.load(f)
label = torch.LongTensor([3])
# import ipdb; ipdb.set_trace()

# Classification before Adv 
print "Before Optimization Image is classified as: "
print np.argmax(net(Variable(x_6)).data.numpy())

# Optimization Loop 
for iteration in range(10000):

    print "iteration: ", iteration
    x,y = Variable(x_6), Variable(label)
    optimizer.zero_grad() 
    outputs = net(x)
    xent_loss = SoftmaxWithXent(outputs, y) 
    adv_loss  = xent_loss + torch.mean(torch.pow(net.r,2))

    adv_loss.backward() 
    # xent_loss.backward()
    optimizer.step() 

    # print stats 
    print "xent loss: {} ".format(xent_loss.data.numpy())
    classif_op = np.argmax(net(Variable(x_6)).data.numpy())
    print "Classified as: ", classif_op
    
    # keep optimizing Until classif_op == label
    if classif_op == label.numpy():
        break 

print "After Optimization Image is classified as: "
print np.argmax(net(Variable(x_6)).data.numpy())

# Show Orig image + Noise + Adversarial Image 
orig = x_6.numpy().reshape(28,28)
nois = net.r.data.numpy().reshape(28,28)
mix = orig + nois 
plt.subplot(131)
plt.imshow(orig,'gray')
plt.subplot(132)
plt.imshow(nois,'gray')
plt.subplot(133)
plt.imshow(mix,'gray')
plt.show()
print "orig: ", orig.max(), orig.min() 
print "nois: ", nois.max(), nois.min() 
print "mix: ", mix.max(), mix.min() 

