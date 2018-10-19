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

# Load pre-trained weights 
weights_dict = {} 
with open("../common/weights.pkl", "rb") as f:
    weights_dict = pickle.load(f)
for param in net.named_parameters():
    if param[0] in weights_dict.keys():
        print ("Copying: ", param[0])
        param[1].data = weights_dict[param[0]].data 
print ("Weights Loaded!")

# Load 5K samples 
with open("../common/5k_samples.pkl","rb") as f: 
    samples_5k = pickle.load(f)
    
xs = samples_5k["images"]
y_trues = samples_5k["labels"]
noises = [] 
y_preds = []
y_preds_adversarial = [] 
totalMisclassifications = 0
xs_clean = [] 
y_trues_clean = []

for x, y_true in tqdm(zip(xs, y_trues)):
    
    # Wrap x as a variable 
    x = Variable(torch.FloatTensor(x.reshape(1,784)), requires_grad=True)
    y_true = Variable(torch.LongTensor(np.array([y_true])), requires_grad=False)
    
    # Classification before Adv 
    y_pred =  np.argmax(net(x).data.numpy())
    
    # Generate Adversarial Image 

    # Forward pass
    outputs = net(x)
    loss = SoftmaxWithXent(outputs, y_true)
    loss.backward() # obtain gradients on x

    # Add perturbation
    epsilon = 0.1 
    x_grad   = torch.sign(x.grad.data)
    x_adversarial = torch.clamp(x.data + epsilon * x_grad, 0, 1) 

    # Classification after optimization  
    y_pred_adversarial = np.argmax(net(Variable(x_adversarial)).data.numpy())
    # print "Before: {} | after: {}".format(y_pred, y_pred_adversarial)
    
    # print "Y_TRUE: {} | Y_PRED: {}".format(_y_true, y_pred)
    if y_true.data.numpy() != y_pred:
        print ("WARNING: MISCLASSIFICATION ERROR")
        totalMisclassifications += 1
    else:
        y_preds.append(y_pred)
        y_preds_adversarial.append(y_pred_adversarial)
        noises.append( (x_adversarial - x.data).numpy() ) 
        xs_clean.append(x.data.numpy())
        y_trues_clean.append(y_true.data.numpy())

print ("Total totalMisclassifications : ", totalMisclassifications)
print ("out of : ", len(xs))

with open("bulk_mnist_fgsd.pkl","wb") as f: 
    adv_data_dict = {
            "xs" : xs_clean, 
            "y_trues" : y_trues_clean,
            "y_preds" : y_preds,
            "noises" : noises,
            "y_preds_adversarial" : y_preds_adversarial
            }    
    pickle.dump(adv_data_dict, f)
