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


pkl_loc = sys.argv[1]
with open(pkl_loc, "r") as f: 
    adv_data_dict = pickle.load(f) 

xs = adv_data_dict["xs"]
y_trues = adv_data_dict["y_trues"]
y_preds = adv_data_dict["y_preds"]
noises  = adv_data_dict["noises"]
y_preds_adversarial = adv_data_dict["y_preds_adversarial"]  

# visualize N random images 
idxs = np.random.choice(range(50), size=(9,), replace=False)
for matidx, idx in enumerate(idxs):
    orig_im = xs[idx]
    adv_im  = orig_im + noises[idx]
    orig_im = tensor2im(orig_im)
    adv_im  = tensor2im(adv_im)
    disp_im = np.concatenate((orig_im, adv_im), axis=1)
    disp_im = np.clip(disp_im, 0, 1)
    # import ipdb; ipdb.set_trace()
    # disp_im = disp_im.astype(np.uint8)
    plt.subplot(3,3,matidx+1)
    plt.imshow(disp_im)
    plt.xticks([])
    plt.yticks([])
    plt.title("{} / {}".format(y_preds[idx][:30], y_preds_adversarial[idx][:30]), fontsize = 9)
    
plt.show()
