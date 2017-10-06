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

with open("./labels.json","r") as f: 
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
print ".. loaded pre-trained vgg16"

images, labels, y_preds, noise, adversarial_class = [], [], [], [], []
for _imloc in image_location_generator("./downloads/"):
    
    x = Variable(image_loader(_imloc), requires_grad=True)
    images.append(x.data.numpy())
    output = vgg16.forward(x)
    y_preds.append(ImageNet_mapping[str(output.data.numpy().argmax())])
    labels.append(ImageNet_mapping[str(output.data.numpy().argmax())])
    y = Variable(torch.LongTensor(np.array([output.data.numpy().argmax()])), requires_grad = False)
    loss = SoftmaxWithXent(output, y)
    loss.backward()

    # Add perturbation 
    epsilon = 0.02
    x_g     = torch.sign(x.grad.data)
    adv_x   = x.data + epsilon*x_g  # we do not know the min/max because of torch's own stuff

    # Check classification 
    adversarial_class.append(ImageNet_mapping[str(np.argmax(vgg16.forward(Variable(adv_x)).data.numpy()))])
    noise.append((adv_x - x.data).numpy())

    # Display 
    print y_preds[-1], " | ", adversarial_class[-1]

with open("adv_results_imagenet_fgsm.pkl", "w") as f: 
    adv_data_dict = {
       'x' : images, 
       'y_true': labels, 
       'y_pred': y_preds,
       'r': noise,
       'adversarial_class': adversarial_class 
    }
    pickle.dump(adv_data_dict, f)


