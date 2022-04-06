import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import scipy.io

# External libs
import face_alignment
import face_alignment.detection.sfd as face_detector_module

import cv2
from PIL import Image

def load_img_2_tensors(img, fa, face_detector, img_name):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.copyMakeBorder(
        img,
        top=50,
        bottom=50,
        left=50,
        right=50,
        borderType=cv2.BORDER_DEFAULT
    )
    s = 1.5e3
    t = [0, 0, 0]
    scale = 1.2
    size = 256
    ds = face_detector.detect_from_image(img[..., ::-1].copy())

    for i in range(len(ds)):
    	d = ds[i]
    	center = [d[3] - (d[3] - d[1]) / 2.0, d[2] - (d[2] - d[0]) / 2.0]
    	center[0] += (d[3] - d[1]) * 0.06
    	center[0] = int(center[0])
    	center[1] = int(center[1])
    	l = max(d[2] - d[0], d[3] - d[1]) * scale
    	if l < 200:
        	continue
    	x_s = center[1] - int(l / 2)
    	y_s = center[0] - int(l / 2)
    	x_e = center[1] + int(l / 2)
    	y_e = center[0] + int(l / 2)
    	t = [256. - center[1] + t[0], center[0] - 256. + t[1], 0]
    	rescale = size / (x_e - x_s)
    	s *= rescale
    	t = [t[0] * rescale, t[1] * rescale, 0.]
    	img = Image.fromarray(img).crop((x_s, y_s, x_e, y_e))
    	img = cv2.resize(np.asarray(img), (size, size)).astype(np.float32)
    	break
    assert img.shape[0] == img.shape[1] == 256
    cv2.imwrite('output_image_dir/'+img_name, img[:, :, ::-1])


if __name__ == '__main__':
    img_dir = 'input_image_dir/'
    all_images = sorted(os.listdir(img_dir))
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=True)
    face_detector = face_detector_module.FaceDetector(device='cuda', verbose=False)

    for i in range(len(all_images)):
        img = cv2.imread(img_dir+all_images[i])
        load_img_2_tensors(img, fa, face_detector, all_images[i])

