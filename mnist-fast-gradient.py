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

# Load 5K samples 
with open("5k_samples.pkl","r") as f: 
    samples_5k = pickle.load(f) 
# import ipdb; ipdb.set_trace()
images = samples_5k["images"]
labels = samples_5k["labels"]
noise = [] 
y_preds = []
adversarial_class = [] 

for _x, _y_true in tqdm(zip(images, labels)):
    
    # import ipdb; ipdb.set_trace()
    # Wrap x as a variable 
    x = Variable(torch.FloatTensor(_x.reshape(1,784)),     requires_grad=True)
    y = Variable(torch.LongTensor(np.array([_y_true])), requires_grad=False)
    
    # Classification before Adv 
    y_pred =  np.argmax(net(x).data.numpy())
    y_preds.append(y_pred)
    
    print "Y_TRUE: {} | Y_PRED: {}".format(_y_true, y_pred)
    if _y_true != y_pred:
        print "WARNING: IMAGE WAS NOT CLASSIFIED CORRECTLY"

    # Generate Adversarial Image 
    # Forward pass
    outputs = net(x)
    loss = SoftmaxWithXent(outputs, y)
    loss.backward() # obtain gradients on x

    # Add perturbation
    epsilon = 0.1 
    x_g   = torch.sign(x.grad.data)
    adv_example = torch.clamp(x.data + epsilon * x_g, 0, 1) 

    # save adv_image  
    noise.append((adv_example - x.data).numpy())
    print "After Optimization Image is classified as: "
    print np.argmax(net(Variable(adv_example)).data.numpy())
    adversarial_class.append(np.argmax(net(Variable(adv_example)).data.numpy()))


with open("adv_results_fgsd.pkl","w") as f: 
    adv_data_dict = {
            "x" : images, 
            "y_true" : labels,
            "y_pred" : y_preds,
            "r" : noise,
            "adversarial_class" : adversarial_class
            }    
    pickle.dump(adv_data_dict, f)
