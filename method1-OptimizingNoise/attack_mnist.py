import numpy as np
import os, sys, pickle
import matplotlib.pyplot as plt 
import random
from attack import Attack 
from tqdm import *


# Load 5K samples 
with open("../common/5k_samples.pkl","rb") as f: 
    samples_5k = pickle.load(f) 
images = samples_5k["images"]
labels = samples_5k["labels"]

# Aggregate
xs, y_trues, y_preds, y_preds_adversarial, noises = [], [], [], [], []

# Attack each example 
attacker = Attack(weights="../common/weights.pkl")
for x, y_true in tqdm(zip(images, labels)):

    y_target = random.choice( list(set([0,1,2,3,4,5,6,7,8,9]) - set([y_true])) ) 
    noise, y_pred, y_pred_adversarial = attacker.attack(x, y_true, y_target, regularization="l2")
    
    if y_pred == y_true:
        # store
        xs.append(x)
        y_trues.append(y_true)
        y_preds.append(y_pred)
        y_preds_adversarial.append(y_pred_adversarial)
        noises.append(noise.squeeze())
    else: 
        print ("y_pred != y_true, not storing to disk" )

with open("bulk_mnist_adversarial_examples.pkl","wb") as f: 
    save_dict = {"xs":xs,
                 "y_trues":y_trues,
                 "y_preds":y_preds,
                 "y_preds_adversarial":y_preds_adversarial,
                 "noises": noises }
    pickle.dump(save_dict, f) 
print ("..done")

    

