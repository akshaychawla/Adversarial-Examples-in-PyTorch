import numpy as np
import matplotlib.pyplot as plt 
import pickle 
import sys, os 

def tensor2im(tens):
    
    im = tens[0] 
    im[0] = im[0] * 0.229
    im[1] = im[1] * 0.224 
    im[2] = im[2] * 0.225 
    im[0] += 0.485 
    im[1] += 0.456 
    im[2] += 0.406
    im = np.moveaxis(im, 0, 2)
    return im 
    # return (im*255).astype(np.uint8) # RGB in [0,255]


_pkl_loc = sys.argv[1]
with open(_pkl_loc, "r") as f: 
    adv_data_dict = pickle.load(f) 

x = adv_data_dict["x"]
y_true = adv_data_dict["y_true"]
y_pred = adv_data_dict["y_pred"]
r      = adv_data_dict["r"]
adversarial_class = adv_data_dict["adversarial_class"]  

# Clean up; only choose images where y_true == y_pred 
# x_clean, y_true_clean, y_pred_clean, r_clean, adversarial_class_clean = [], [], [], [], []
# for dat_point in zip(x, y_true, y_pred, r, adversarial_class):
    # # if (dat_point[1] == dat_point[2]) and (dat_point[1] != dat_point[4]):
    # x_clean.append(dat_point[0])
    # y_true_clean.append(dat_point[1])
    # y_pred_clean.append(dat_point[2])
    # r_clean.append(dat_point[3])
    # adversarial_class_clean.append(dat_point[4])

# visualize N random images 
idxs = np.random.choice(range(50), size=(6,), replace=False)
for matidx, idx in enumerate(idxs):
    orig_im = x[idx]
    adv_im  = orig_im + r[idx]
    orig_im = tensor2im(orig_im)
    adv_im  = tensor2im(adv_im)
    disp_im = np.concatenate((orig_im, adv_im), axis=1)
    disp_im = np.clip(disp_im, 0, 1)
    # import ipdb; ipdb.set_trace()
    # disp_im = disp_im.astype(np.uint8)
    plt.subplot(3,2,matidx+1)
    plt.imshow(disp_im)
    plt.xticks([])
    plt.yticks([])
    plt.title("{} / {}".format(y_pred[idx][:30], adversarial_class[idx][:30]), fontsize = 9)
    
plt.show()
    
# import ipdb; ipdb.set_trace()
# Noise statistics 
# r_clean, x_clean, y_true_clean, y_pred_clean = np.array(r_clean), np.array(x_clean), np.array(y_true_clean), np.array(y_pred_clean)
# r_clean = np.squeeze(r_clean, axis=1)
# adv_exs = x_clean + r_clean 
# print "Adv examples: max, min: ", adv_exs.max(), adv_exs.min()
# print "Noise: Mean, Max, Min: "
# print np.mean(r_clean), np.max(r_clean), np.min(r_clean)
