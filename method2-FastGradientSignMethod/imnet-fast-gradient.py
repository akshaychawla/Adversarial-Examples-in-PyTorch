import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
from torchvision import transforms
from tqdm import *
import torchvision.models 
import pickle

with open("../common/labels.json","r") as f: 
    import json 
    ImageNet_mapping = json.loads(f.read())

def image_location_generator(_root):
    import os
    _dirs = os.listdir(_root)
    assert len(_dirs) > 0, "no directories in given root folder"
    for _dir in _dirs:
        _imfiles = os.listdir(os.path.join(_root,_dir))
        for _imfile in _imfiles:
            yield os.path.join(_root, _dir, _imfile)

imsize = (224, 224)
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def image_loader(image_name):
    """load image, returns tensor"""
    image = Image.open(image_name)
    image = image.convert("RGB") # Auto remove the "alpha" channel from png image
    image = loader(image).float()
    image = normalize(image).float()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image 

# Pretrained VGG16 model
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.eval() # disable dropout, batchnorm
SoftmaxWithXent = nn.CrossEntropyLoss()
print (".. loaded pre-trained vgg16")

xs, y_trues, y_preds, noises, y_preds_adversarial = [], [], [], [], []

for imloc in tqdm(image_location_generator("./downloads/")):
    
    x = Variable(image_loader(imloc), requires_grad=True)
    output = vgg16.forward(x)
    y = Variable(torch.LongTensor(np.array([output.data.numpy().argmax()])), requires_grad = False)
    loss = SoftmaxWithXent(output, y)
    loss.backward()

    # Add perturbation 
    epsilon = 0.02
    x_grad     = torch.sign(x.grad.data)
    adv_x   = x.data + epsilon*x_grad  # we do not know the min/max because of torch's own stuff

    # Check adversarilized output 
    y_pred_adversarial = ImageNet_mapping[ str(np.argmax(vgg16.forward(Variable(adv_x)).data.numpy())) ]
    y_true = ImageNet_mapping[ str( int( y.data.numpy() ) ) ]

    if y_pred_adversarial == y_true:
        print ("Error: Could not adversarialize image ")
    else:
        xs.append(x.data.numpy())
        y_preds.append( y_true )
        y_trues.append( y_true )
        noises.append((adv_x - x.data).numpy())
        y_preds_adversarial.append( y_pred_adversarial )

        # Display 
        # print y_preds[-1], " | ", y_preds_adversarial[-1]

import ipdb; ipdb.set_trace()
with open("bulk_imnet_fgsm.pkl", "wb") as f: 
    adv_data_dict = {
       'xs' : xs, 
       'y_trues': y_trues, 
       'y_preds': y_preds,
       'noises': noises,
       'y_preds_adversarial': y_preds_adversarial
    }
    pickle.dump(adv_data_dict, f)


