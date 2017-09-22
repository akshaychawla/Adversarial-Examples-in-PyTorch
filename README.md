# Adversarial Examples 

Note: Still under construction

This repository contains code to generate adversarial examples for the mnist, cifar and ImageNet datasets. 

How to run: 
1. train on mnist 
python pytorch_nn.py 

2. Generate an adversarial example 
python adversarial_mnist_paper1.py 

--> this will try to find an optimizer "r" that fools the network into believing that said image is a "3" instead of a "6" 
--> you can also change the target class by changing line 50  

Example: 
Input Image | Noise Added | Adversarial Image 

![example](example.png)
