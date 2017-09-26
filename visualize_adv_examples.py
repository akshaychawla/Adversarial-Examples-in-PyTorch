import numpy as np
import matplotlib.pyplot as plt 
import pickle 
import sys, os 

_pkl_loc = sys.argv[1]
with open(_pkl_loc, "r") as f: 
    adv_data_dict = pickle.load(f) 

x = adv_data_dict["x"]
y_true = adv_data_dict["y_true"]
y_pred = adv_data_dict["y_pred"]
r      = adv_data_dict["r"]

# Clean up; only choose images where y_true == y_pred 
x_clean, y_true_clean, y_pred_clean, r_clean = [], [], [], []
for dat_point in zip(x, y_true, y_pred, r):
    if dat_point[1] == dat_point[2]:
        x_clean.append(dat_point[0])
        y_true_clean.append(dat_point[1])
        y_pred_clean.append(dat_point[2])
        r_clean.append(dat_point[3])

import ipdb; ipdb.set_trace()
# visualize N random images 
idxs = np.random.choice(range(50), size=(20,), replace=False)
for matidx, idx in enumerate(idxs):
    orig_im = x_clean[idx].reshape(28,28)
    adv_im  = orig_im + r_clean[idx].reshape(28,28)
    disp_im = np.concatenate((orig_im, adv_im), axis=1)
    plt.subplot(5,4,matidx+1)
    plt.imshow(disp_im, "gray")
plt.show()
    
    
# Noise statistics 
r_clean = np.array(r_clean)
print np.mean(r_clean), np.max(r_clean), np.min(r_clean)
