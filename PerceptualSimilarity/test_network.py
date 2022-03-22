import torch
import lpips
from IPython import embed
import os
import sys
import cv2
import numpy as np
import torch.nn.functional as F

use_gpu = True         # Whether to use GPU
spatial = True         # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
#loss_fn = lpips.LPIPS(net='vgg', spatial=spatial)
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

if(use_gpu):
	loss_fn.cuda()

## Example usage with images
predicted_images = sorted(os.listdir('../test_raytracing_relighting_CelebAHQ_DSSIM_8x/'))
#We save out 6 files per image including the rendered image, surface normals, depth map, albedo, etc. We only want to compare against the rendered images (hence 2::6)
predicted_images = predicted_images[2::6]
print(len(predicted_images))
gt_images = sorted(os.listdir('../groundtruth_images_MP_18_lightings/'))
batch_mask_files = sorted(os.listdir('../MP_depth_masks_fill_nose/'))

lpips_metric = 0.0
all_lpips = np.zeros((862))
for i in range(len(predicted_images)):
    ex_pred = lpips.im2tensor(lpips.load_image('../test_raytracing_relighting_CelebAHQ_DSSIM_8x/'+predicted_images[i]))
    ex_ref = lpips.im2tensor(lpips.load_image('../groundtruth_images_MP_18_lightings/'+gt_images[i]))
    mask = cv2.imread('../MP_depth_masks_fill_nose/'+batch_mask_files[i])/255.0
    mask = mask[:, :, 0]

    if(use_gpu):
        ex_pred = ex_pred.cuda()
        ex_ref = ex_ref.cuda()

    ex = loss_fn.forward(ex_ref, ex_pred)

    if not spatial:
        print('Distances: (%.3f, %.3f)'%(ex_d0, ex_d1))
    else:
        print('Distances: (%.3f)'%((torch.sum(torch.from_numpy(mask).float().cuda()*ex))/torch.sum(torch.from_numpy(mask).float().cuda()*ex > 0)))            # The mean distance is approximately the same as the non-spatial distance
        lpips_metric += torch.sum(torch.from_numpy(mask).float().cuda()*ex)/torch.sum(torch.from_numpy(mask).float().cuda()*ex > 0)
        all_lpips[i] = (torch.sum(torch.from_numpy(mask).float().cuda()*ex)/torch.sum(torch.from_numpy(mask).float().cuda()*ex > 0)).cpu().detach().numpy()
        
lpips_metric = lpips_metric.cpu().detach().numpy()
print('Average lpips: '+str(lpips_metric/len(predicted_images)))
print('Std lpips: '+str(np.std(all_lpips)))
