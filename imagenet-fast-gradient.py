import numpy as np
import matplotlib.pyplot as plt 
import cv2 
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
from torchvision import transforms
from tqdm import *
import torchvision.models 
from PIL import Image

with open("./labels.json","r") as f: 
    import json 
    ImageNet_mapping = json.loads(f.read())

imsize = (224, 224)
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = normalize(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image 
image = image_loader("./cat.jpeg")
vgg16 = torchvision.models.vgg16(pretrained=True)
output = vgg16.forward(image)
output = output.data.numpy() 
print ImageNet_mapping[str(output.argmax())], np.argmax(output)


