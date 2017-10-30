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
adversarial_class = adv_data_dict["adversarial_class"]  

# Clean up; only choose images where y_true == y_pred 
x_clean, y_true_clean, y_pred_clean, r_clean, adversarial_class_clean = [], [], [], [], []
for dat_point in zip(x, y_true, y_pred, r, adversarial_class):
    if (dat_point[1] == dat_point[2]) and (dat_point[1] != dat_point[4]):
        x_clean.append(dat_point[0])
        y_true_clean.append(dat_point[1])
        y_pred_clean.append(dat_point[2])
        r_clean.append(dat_point[3])
        adversarial_class_clean.append(dat_point[4])

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
    
# import ipdb; ipdb.set_trace()
# Noise statistics 
r_clean, x_clean, y_true_clean, y_pred_clean = np.array(r_clean), np.array(x_clean), np.array(y_true_clean), np.array(y_pred_clean)
r_clean = np.squeeze(r_clean, axis=1)
adv_exs = x_clean + r_clean 
print "Adv examples: max, min: ", adv_exs.max(), adv_exs.min()
print "Noise: Mean, Max, Min: "
print np.mean(r_clean), np.max(r_clean), np.min(r_clean)
