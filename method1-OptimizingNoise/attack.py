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
import sys, os

# DEFINE NETWORK
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # my network is composed of only affine layers 
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)   
        self.r   = nn.Parameter(data=torch.zeros(1,784), requires_grad=True) # really small initial values

    def forward(self, x):
        x = x + self.r 
        x = torch.clamp(x, 0, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class Attack:
    def __init__(self, weights):
        self.net = Net() 
        self.softmaxwithxent = nn.CrossEntropyLoss()
        self.optimizer       = optim.SGD(params=[self.net.r], lr=0.008)
        self.load_weights(weights) 

    def load_weights(self, weights=None):
        assert os.path.isfile(weights), "Error: weight file {} is invalid".format(weights)
        # LOAD PRE-TRAINED WEIGHTS 
        weights_dict = {} 
        with open(weights, "rb") as f:
            weights_dict = pickle.load(f)
        for param in self.net.named_parameters():
            if param[0] in weights_dict.keys():
                print ("Copying: ", param[0])
                param[1].data = weights_dict[param[0]].data 
        print ("Weights Loaded!")

    def attack(self, x, y_true, y_target, regularization=None):
        """
        This method uses the method described in the paper
        "Intriguing properties of neural networks" to find a 
        noise vector 'r' that misclassifies 'x' as 'y_target'. 

        Parameters
        ----------
        x: a numpy array containing an mnist example 
        y_target: target label for attack. (int) 
        y_true: true label for x (int)

        Returns
        -------
        noise: Numpy array (1x784) of the noise to be added to x 
        y_pred: Prediction before adversarial optimization  
        y_pred_adversarial: Prediction after adversarial optimization 
        """

        _x = Variable(torch.FloatTensor(x))
        _y_target = Variable(torch.LongTensor([y_target]))

        # Reset value of r 
        self.net.r.data = torch.zeros(1,784) 

        # Classification before modification 
        y_pred =  np.argmax(self.net(_x).data.numpy())
        incorrect_classify = False
        # print "Y_TRUE: {} | Y_PRED: {}".format(_y_true, y_pred)
        if y_true != y_pred:
            incorrect_classify = True
            print ("WARNING: IMAGE WAS NOT CLASSIFIED CORRECTLY")

        # Optimization Loop 
        for iteration in range(1000):

            self.optimizer.zero_grad() 
            outputs = self.net(_x)
            xent_loss = self.softmaxwithxent(outputs, _y_target) 

            if regularization == "l1":
                adv_loss = xent_loss + torch.mean(torch.abs(self.net.r))
            elif regularization == "l2":
                adv_loss  = xent_loss + torch.mean(torch.pow(self.net.r,2))
            elif regularization == None:
                adv_loss = xent_loss
            else:
                raise Exception("regularization method {} is not implemented, please choose one of l1, l2 or None".format(regularization))

            adv_loss.backward() 
            self.optimizer.step() 

            # keep optimizing Until classif_op == _y_target
            y_pred_adversarial = np.argmax(self.net(_x).data.numpy())
            if y_pred_adversarial == y_target:
                break 

        if iteration == 999:
            print ("Warning: optimization loop ran for 1000 iterations. The result may not be correct")

        return self.net.r.data.numpy(), y_pred, y_pred_adversarial 


# net = Net()
# print(net)
# SoftmaxWithXent = nn.CrossEntropyLoss()

# # OPTIMIZE FOR "r" 
# optimizer = optim.SGD(params=[net.r], lr=0.008)


# # Load 5K samples 
# with open("5k_samples.pkl","r") as f: 
    # samples_5k = pickle.load(f) 
# # import ipdb; ipdb.set_trace()
# images = samples_5k["images"]
# labels = samples_5k["labels"]
# noise = [] 
# y_preds = []

# for _x, _y_true in zip(images, labels):
    
    # _x = torch.FloatTensor(_x)
    # # import ipdb; ipdb.set_trace() 
    # # Note: choose _y_target to be something other than _y_true
    # _y_target = random.choice( list(set([0,1,2,3,4,5,6,7,8,9]) - set([_y_true])) ) 
    # _y_target = torch.LongTensor([_y_target])

    # # Reset value of r 
    # net.r.data = torch.zeros(1,784) 
    # # import ipdb; ipdb.set_trace()

    # # Classification before Adv 
    # y_pred =  np.argmax(net(Variable(_x)).data.numpy())
    # y_preds.append(y_pred)
    
    # print "Y_TRUE: {} | Y_PRED: {}".format(_y_true, y_pred)
    # if _y_true != y_pred:
        # print "WARNING: IMAGE WAS NOT CLASSIFIED CORRECTLY"

    # # Optimization Loop 
    # tqd_loop = trange(1000)
    # for iteration in tqd_loop:

        # x,y = Variable(_x), Variable(_y_target)
        # optimizer.zero_grad() 
        # outputs = net(x)
        # xent_loss = SoftmaxWithXent(outputs, y) 
        # adv_loss  = xent_loss + torch.mean(torch.pow(net.r,2))

        # adv_loss.backward() 
        # # xent_loss.backward()
        # optimizer.step() 

        # # print stats 
        # classif_op = np.argmax(net(Variable(_x)).data.numpy())
        # tqd_loop.set_description("xent Loss: {} classif: {}".format(xent_loss.data.numpy(), classif_op))

        
        # # keep optimizing Until classif_op == _y_target
        # if classif_op == _y_target.numpy()[0]:
            # tqd_loop.close()
            # break 

    # # save adv_image and noise to list 
    # noise.append(net.r.data.numpy())
    # print "After Optimization Image is classified as: "
    # print np.argmax(net(Variable(_x)).data.numpy())

# with open("adv_results_l2.pkl","w") as f: 
    # adv_data_dict = {
            # "x" : images, 
            # "y_true" : labels,
            # "y_pred" : y_preds,
            # "r" : noise
            # }    
    # pickle.dump(adv_data_dict, f)

