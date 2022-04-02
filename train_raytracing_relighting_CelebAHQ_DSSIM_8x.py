import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import cv2
from kornia.geometry.depth import depth_to_normals
import scipy.io
import imageio
import random
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class PatchGAN(nn.Module):
    def __init__(self):
        super(PatchGAN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=(1, 1))
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=(1, 1))
        self.conv5 = nn.Conv2d(512, 1, 4, stride=1, padding=(1, 1))

        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

    def forward(self, img):
        conv1 = F.leaky_relu(self.conv1(img), 0.2)
        conv2 = F.leaky_relu(self.bn2(self.conv2(conv1)), 0.2)
        conv3 = F.leaky_relu(self.bn3(self.conv3(conv2)), 0.2)
        conv4 = F.leaky_relu(self.bn4(self.conv4(conv3)), 0.2)
        conv5 = self.conv5(conv4)

        return conv5
        

class RelightNet(nn.Module):
    def __init__(self):
        super(RelightNet, self).__init__()
        self.batch_size = 3
        self.img_height = 256
        self.img_width = 256
        self.lr = 0.0001
        self.df_dim = 64
        self.directional_intensity = 0.5
        self.light_distance = 4013.0
        self.num_sample_points = 160
        self.GD_ratio = 5

        xx, yy = np.meshgrid(range(self.img_width), range(self.img_height), indexing='xy')
        self.xx = nn.Parameter(torch.from_numpy(np.copy(xx)).unsqueeze(0).repeat([self.batch_size, 1, 1]).float(), requires_grad=False)-(self.img_width/2.0)
        self.yy = (self.img_height/2.0)-nn.Parameter(torch.from_numpy(np.copy(yy)).unsqueeze(0).repeat([self.batch_size, 1, 1]).float(), requires_grad=False)
        self.xx = self.xx.cuda()
        self.yy = self.yy.cuda()

        #encoder layers
        self.conv_c1_og = nn.Conv2d(3, 16, 5, padding=(2, 2))
        self.conv_h1_1 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.conv_h1_2 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.conv_h2_1 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.conv_h2_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.conv_shortcut_h1_out = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.conv_h3_1 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.conv_h3_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.conv_shortcut_h2_out = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.conv_h4_1 = nn.Conv2d(64, 155, 3, padding=(1, 1))
        self.conv_h4_2 = nn.Conv2d(155, 155, 3, padding=(1, 1))
        self.conv_shortcut_h3_out = nn.Conv2d(64, 155, 3, padding=(1, 1))

        self.bn_c1_og = nn.BatchNorm2d(16)
        self.bn_h1_1 = nn.BatchNorm2d(16)
        self.bn_h1_2 = nn.BatchNorm2d(16)
        self.bn_h2_1 = nn.BatchNorm2d(32)
        self.bn_h2_2 = nn.BatchNorm2d(32)
        self.bn_shortcut_h1_out = nn.BatchNorm2d(32)
        self.bn_h3_1 = nn.BatchNorm2d(64)
        self.bn_h3_2 = nn.BatchNorm2d(64) 
        self.bn_shortcut_h2_out = nn.BatchNorm2d(64)
        self.bn_h4_1 = nn.BatchNorm2d(155)
        self.bn_h4_2 = nn.BatchNorm2d(155)
        self.bn_shortcut_h3_out = nn.BatchNorm2d(155)

        #lighting branch layers
        self.AvgPool_LF = nn.AvgPool2d((16, 16), (1, 1))

        self.linear_SL1 = nn.Linear(27, 128)
        self.linear_SL2 = nn.Linear(128, 4)

        #albedo decoder layers
        self.deconv_albedo_h5_1 = nn.ConvTranspose2d(128, 64, 3, padding=(1, 1))
        self.deconv_albedo_h5_2 = nn.ConvTranspose2d(64, 64, 3, padding=(1, 1))
        self.deconv_albedo_shortcut_all_features = nn.ConvTranspose2d(128, 64, 3, padding=(1, 1))
        self.conv_albedo_skip_s1_1 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.conv_albedo_skip_s1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.deconv_albedo_h6_1 = nn.ConvTranspose2d(64, 32, 3, padding=(1, 1))
        self.deconv_albedo_h6_2 = nn.ConvTranspose2d(32, 32, 3, padding=(1, 1))
        self.deconv_albedo_shortcut_h5_out = nn.ConvTranspose2d(64, 32, 3, padding=(1, 1))
        self.conv_albedo_skip_s2_1 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.conv_albedo_skip_s2_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.deconv_albedo_h7_1 = nn.ConvTranspose2d(32, 16, 3, padding=(1, 1))
        self.deconv_albedo_h7_2 = nn.ConvTranspose2d(16, 16, 3, padding=(1, 1))
        self.deconv_albedo_shortcut_h6_out = nn.ConvTranspose2d(32, 16, 3, padding=(1, 1))
        self.conv_albedo_skip_s3_1 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.conv_albedo_skip_s3_2 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.deconv_albedo_h8_1 = nn.ConvTranspose2d(16, 16, 3, padding=(1, 1))
        self.deconv_albedo_h8_2 = nn.ConvTranspose2d(16, 16, 3, padding=(1, 1))
        self.conv_albedo_skip_s4_1 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.conv_albedo_skip_s4_2 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.conv_albedo_c2_1 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.conv_albedo_c2_2 = nn.Conv2d(16, 16, 1)
        self.conv_albedo_c2_3 = nn.Conv2d(16, 16, 1)
        self.conv_albedo_c2_o = nn.Conv2d(16, 3, 1)

        self.bn_albedo_h5_1 = nn.BatchNorm2d(64)
        self.bn_albedo_h5_2 = nn.BatchNorm2d(64)
        self.bn_albedo_shortcut_all_features = nn.BatchNorm2d(64)
        self.bn_albedo_skip_s1_1 = nn.BatchNorm2d(64)
        self.bn_albedo_skip_s1_2 = nn.BatchNorm2d(64)
        self.bn_albedo_h6_1 = nn.BatchNorm2d(32)
        self.bn_albedo_h6_2 = nn.BatchNorm2d(32)
        self.bn_albedo_shortcut_h5_out = nn.BatchNorm2d(32)
        self.bn_albedo_skip_s2_1 = nn.BatchNorm2d(32)
        self.bn_albedo_skip_s2_2 = nn.BatchNorm2d(32)
        self.bn_albedo_h7_1 = nn.BatchNorm2d(16)
        self.bn_albedo_h7_2 = nn.BatchNorm2d(16)
        self.bn_albedo_shortcut_h6_out = nn.BatchNorm2d(16)
        self.bn_albedo_skip_s3_1 = nn.BatchNorm2d(16)
        self.bn_albedo_skip_s3_2 = nn.BatchNorm2d(16)
        self.bn_albedo_h8_1 = nn.BatchNorm2d(16)
        self.bn_albedo_h8_2 = nn.BatchNorm2d(16)
        self.bn_albedo_skip_s4_1 = nn.BatchNorm2d(16)
        self.bn_albedo_skip_s4_2 = nn.BatchNorm2d(16)
        self.bn_albedo_c2_1 = nn.BatchNorm2d(16)
        self.bn_albedo_c2_2 = nn.BatchNorm2d(16)
        self.bn_albedo_c2_3 = nn.BatchNorm2d(16)

        self.upsample_albedo_h5_out = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_albedo_h6_out = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_albedo_h7_out = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_albedo_h8_out = nn.Upsample(scale_factor=2, mode='nearest')

        #depth decoder layers
        self.deconv_depth_h5_1 = nn.ConvTranspose2d(128, 64, 3, padding=(1, 1))
        self.deconv_depth_h5_2 = nn.ConvTranspose2d(64, 64, 3, padding=(1, 1))
        self.deconv_depth_shortcut_all_features = nn.ConvTranspose2d(128, 64, 3, padding=(1, 1))
        self.conv_depth_skip_s1_1 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.conv_depth_skip_s1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.deconv_depth_h6_1 = nn.ConvTranspose2d(64, 32, 3, padding=(1, 1))
        self.deconv_depth_h6_2 = nn.ConvTranspose2d(32, 32, 3, padding=(1, 1))
        self.deconv_depth_shortcut_h5_out = nn.ConvTranspose2d(64, 32, 3, padding=(1, 1))
        self.conv_depth_skip_s2_1 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.conv_depth_skip_s2_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.deconv_depth_h7_1 = nn.ConvTranspose2d(32, 16, 3, padding=(1, 1))
        self.deconv_depth_h7_2 = nn.ConvTranspose2d(16, 16, 3, padding=(1, 1))
        self.deconv_depth_shortcut_h6_out = nn.ConvTranspose2d(32, 16, 3, padding=(1, 1))
        self.conv_depth_skip_s3_1 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.conv_depth_skip_s3_2 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.deconv_depth_h8_1 = nn.ConvTranspose2d(16, 16, 3, padding=(1, 1))
        self.deconv_depth_h8_2 = nn.ConvTranspose2d(16, 16, 3, padding=(1, 1))
        self.conv_depth_skip_s4_1 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.conv_depth_skip_s4_2 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.conv_depth_c2_1 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.conv_depth_c2_2 = nn.Conv2d(16, 16, 1)
        self.conv_depth_c2_3 = nn.Conv2d(16, 16, 1)
        self.conv_depth_c2_o = nn.Conv2d(16, 1, 1)

        self.bn_depth_h5_1 = nn.BatchNorm2d(64)
        self.bn_depth_h5_2 = nn.BatchNorm2d(64)
        self.bn_depth_shortcut_all_features = nn.BatchNorm2d(64)
        self.bn_depth_skip_s1_1 = nn.BatchNorm2d(64)
        self.bn_depth_skip_s1_2 = nn.BatchNorm2d(64)
        self.bn_depth_h6_1 = nn.BatchNorm2d(32)
        self.bn_depth_h6_2 = nn.BatchNorm2d(32)
        self.bn_depth_shortcut_h5_out = nn.BatchNorm2d(32)
        self.bn_depth_skip_s2_1 = nn.BatchNorm2d(32)
        self.bn_depth_skip_s2_2 = nn.BatchNorm2d(32)
        self.bn_depth_h7_1 = nn.BatchNorm2d(16)
        self.bn_depth_h7_2 = nn.BatchNorm2d(16)
        self.bn_depth_shortcut_h6_out = nn.BatchNorm2d(16)
        self.bn_depth_skip_s3_1 = nn.BatchNorm2d(16)
        self.bn_depth_skip_s3_2 = nn.BatchNorm2d(16)
        self.bn_depth_h8_1 = nn.BatchNorm2d(16)
        self.bn_depth_h8_2 = nn.BatchNorm2d(16)
        self.bn_depth_skip_s4_1 = nn.BatchNorm2d(16)
        self.bn_depth_skip_s4_2 = nn.BatchNorm2d(16)
        self.bn_depth_c2_1 = nn.BatchNorm2d(16)
        self.bn_depth_c2_2 = nn.BatchNorm2d(16)
        self.bn_depth_c2_3 = nn.BatchNorm2d(16)

        self.upsample_depth_h5_out = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_depth_h6_out = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_depth_h7_out = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_depth_h8_out = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, img, epoch, intrinsic_matrix, masks):
        img = img.permute(0, 3, 1, 2)
  
        #encoder
        c1_og = F.leaky_relu(self.bn_c1_og(self.conv_c1_og(img)), 0.2)
        c1 = F.max_pool2d(c1_og, (2, 2))

        h1_1 = F.leaky_relu(self.bn_h1_1(self.conv_h1_1(c1)), 0.2)
        h1_2 = self.bn_h1_2(self.conv_h1_2(h1_1))
        h1_out_og = F.leaky_relu(c1+h1_2, 0.2)
 
        h1_out = F.max_pool2d(h1_out_og, (2, 2))
        h2_1 = F.leaky_relu(self.bn_h2_1(self.conv_h2_1(h1_out)), 0.2)
        h2_2 = self.bn_h2_2(self.conv_h2_2(h2_1))
        shortcut_h1_out = self.bn_shortcut_h1_out(self.conv_shortcut_h1_out(h1_out))
        h2_out_og = F.leaky_relu(shortcut_h1_out+h2_2, 0.2)

        h2_out = F.max_pool2d(h2_out_og, (2, 2))
        h3_1 = F.leaky_relu(self.bn_h3_1(self.conv_h3_1(h2_out)), 0.2)
        h3_2 = self.bn_h3_2(self.conv_h3_2(h3_1))
        shortcut_h2_out = self.bn_shortcut_h2_out(self.conv_shortcut_h2_out(h2_out))
        h3_out_og = F.leaky_relu(shortcut_h2_out+h3_2, 0.2)

        h3_out = F.max_pool2d(h3_out_og, (2, 2))
        h4_1 = F.leaky_relu(self.bn_h4_1(self.conv_h4_1(h3_out)), 0.2)
        h4_2 = self.bn_h4_2(self.conv_h4_2(h4_1))
        shortcut_h3_out = self.bn_shortcut_h3_out(self.conv_shortcut_h3_out(h3_out))
        h4_out = F.leaky_relu(shortcut_h3_out+h4_2, 0.2)

        identity_features = h4_out[:, 0:128, :, :]
        lighting_features = h4_out[:, 128:155, :, :]
        LF_shape = list(lighting_features.size())

        #lighting branch
        LF_avg_pool = self.AvgPool_LF(lighting_features)
        SL_lin1 = F.leaky_relu(self.linear_SL1(LF_avg_pool.permute(0, 2, 3, 1)), 0.2)
        SL_lin2 = self.linear_SL2(SL_lin1)

        #albedo decoder
        h5_1_albedo = F.leaky_relu(self.bn_albedo_h5_1(self.deconv_albedo_h5_1(identity_features)), 0.2)
        h5_2_albedo = self.bn_albedo_h5_2(self.deconv_albedo_h5_2(h5_1_albedo))
        shortcut_all_features_albedo = self.bn_albedo_shortcut_all_features(self.deconv_albedo_shortcut_all_features(identity_features))
        h5_out_albedo = F.leaky_relu(shortcut_all_features_albedo+h5_2_albedo, 0.2)
        h5_out_albedo = self.upsample_albedo_h5_out(h5_out_albedo)

        skip_s1_1_albedo = F.leaky_relu(self.bn_albedo_skip_s1_1(self.conv_albedo_skip_s1_1(h3_out_og)), 0.2)
        skip_s1_2_albedo = self.bn_albedo_skip_s1_2(self.conv_albedo_skip_s1_2(skip_s1_1_albedo))
        skip_s1_out_albedo = F.leaky_relu(h3_out_og+skip_s1_2_albedo, 0.2)

        if(epoch > 8):
            h5_out_albedo = h5_out_albedo+skip_s1_out_albedo

        h6_1_albedo = F.leaky_relu(self.bn_albedo_h6_1(self.deconv_albedo_h6_1(h5_out_albedo)), 0.2)
        h6_2_albedo = self.bn_albedo_h6_2(self.deconv_albedo_h6_2(h6_1_albedo))
        shortcut_h5_out_albedo = self.bn_albedo_shortcut_h5_out(self.deconv_albedo_shortcut_h5_out(h5_out_albedo))
        h6_out_albedo = F.leaky_relu(shortcut_h5_out_albedo+h6_2_albedo, 0.2)
        h6_out_albedo = self.upsample_albedo_h6_out(h6_out_albedo)

        skip_s2_1_albedo = F.leaky_relu(self.bn_albedo_skip_s2_1(self.conv_albedo_skip_s2_1(h2_out_og)), 0.2)
        skip_s2_2_albedo = self.bn_albedo_skip_s2_2(self.conv_albedo_skip_s2_2(skip_s2_1_albedo))
        skip_s2_out_albedo = F.leaky_relu(h2_out_og+skip_s2_2_albedo, 0.2)
        
        if(epoch > 10):
            h6_out_albedo = h6_out_albedo+skip_s2_out_albedo

        h7_1_albedo = F.leaky_relu(self.bn_albedo_h7_1(self.deconv_albedo_h7_1(h6_out_albedo)), 0.2)
        h7_2_albedo = self.bn_albedo_h7_2(self.deconv_albedo_h7_2(h7_1_albedo))
        shortcut_h6_out_albedo = self.bn_albedo_shortcut_h6_out(self.deconv_albedo_shortcut_h6_out(h6_out_albedo))
        h7_out_albedo = F.leaky_relu(shortcut_h6_out_albedo+h7_2_albedo, 0.2)
        h7_out_albedo = self.upsample_albedo_h7_out(h7_out_albedo)

        skip_s3_1_albedo = F.leaky_relu(self.bn_albedo_skip_s3_1(self.conv_albedo_skip_s3_1(h1_out_og)), 0.2)
        skip_s3_2_albedo = self.bn_albedo_skip_s3_2(self.conv_albedo_skip_s3_2(skip_s3_1_albedo))
        skip_s3_out_albedo = F.leaky_relu(h1_out_og+skip_s3_2_albedo, 0.2)

        if(epoch > 12):
            h7_out_albedo = h7_out_albedo+skip_s3_out_albedo

        h8_1_albedo = F.leaky_relu(self.bn_albedo_h8_1(self.deconv_albedo_h8_1(h7_out_albedo)), 0.2)
        h8_2_albedo = self.bn_albedo_h8_2(self.deconv_albedo_h8_2(h8_1_albedo))
        h8_out_albedo = F.leaky_relu(h7_out_albedo+h8_2_albedo, 0.2)
        h8_out_albedo = self.upsample_albedo_h8_out(h8_out_albedo)

        skip_s4_1_albedo = F.leaky_relu(self.bn_albedo_skip_s4_1(self.conv_albedo_skip_s4_1(c1_og)), 0.2)
        skip_s4_2_albedo = self.bn_albedo_skip_s4_2(self.conv_albedo_skip_s4_2(skip_s4_1_albedo))
        skip_s4_out_albedo = F.leaky_relu(c1_og+skip_s4_2_albedo, 0.2)

        if(epoch > 14):
            h8_out_albedo = h8_out_albedo+skip_s4_out_albedo

        c2_1_albedo = F.leaky_relu(self.bn_albedo_c2_1(self.conv_albedo_c2_1(h8_out_albedo)), 0.2)
        c2_2_albedo = F.leaky_relu(self.bn_albedo_c2_2(self.conv_albedo_c2_2(c2_1_albedo)), 0.2)
        c2_3_albedo = F.leaky_relu(self.bn_albedo_c2_3(self.conv_albedo_c2_3(c2_2_albedo)), 0.2)
        c2_o_albedo = self.conv_albedo_c2_o(c2_3_albedo)
        c2_o_albedo = torch.sigmoid(c2_o_albedo).clone()

        #depth decoder
        h5_1_depth = F.leaky_relu(self.bn_depth_h5_1(self.deconv_depth_h5_1(identity_features)), 0.2)
        h5_2_depth = self.bn_depth_h5_2(self.deconv_depth_h5_2(h5_1_depth))
        shortcut_all_features_depth = self.bn_depth_shortcut_all_features(self.deconv_depth_shortcut_all_features(identity_features))
        h5_out_depth = F.leaky_relu(shortcut_all_features_depth+h5_2_depth, 0.2)
        h5_out_depth = self.upsample_depth_h5_out(h5_out_depth)

        skip_s1_1_depth = F.leaky_relu(self.bn_depth_skip_s1_1(self.conv_depth_skip_s1_1(h3_out_og)), 0.2)
        skip_s1_2_depth = self.bn_depth_skip_s1_2(self.conv_depth_skip_s1_2(skip_s1_1_depth))
        skip_s1_out_depth = F.leaky_relu(h3_out_og+skip_s1_2_depth, 0.2)

        if(epoch > 8):
            h5_out_depth = h5_out_depth+skip_s1_out_depth

        h6_1_depth = F.leaky_relu(self.bn_depth_h6_1(self.deconv_depth_h6_1(h5_out_depth)), 0.2)
        h6_2_depth = self.bn_depth_h6_2(self.deconv_depth_h6_2(h6_1_depth))
        shortcut_h5_out_depth = self.bn_depth_shortcut_h5_out(self.deconv_depth_shortcut_h5_out(h5_out_depth))
        h6_out_depth = F.leaky_relu(shortcut_h5_out_depth+h6_2_depth, 0.2)
        h6_out_depth = self.upsample_depth_h6_out(h6_out_depth)

        skip_s2_1_depth = F.leaky_relu(self.bn_depth_skip_s2_1(self.conv_depth_skip_s2_1(h2_out_og)), 0.2)
        skip_s2_2_depth = self.bn_depth_skip_s2_2(self.conv_depth_skip_s2_2(skip_s2_1_depth))
        skip_s2_out_depth = F.leaky_relu(h2_out_og+skip_s2_2_depth, 0.2)
        
        if(epoch > 10):
            h6_out_depth = h6_out_depth+skip_s2_out_depth

        h7_1_depth = F.leaky_relu(self.bn_depth_h7_1(self.deconv_depth_h7_1(h6_out_depth)), 0.2)
        h7_2_depth = self.bn_depth_h7_2(self.deconv_depth_h7_2(h7_1_depth))
        shortcut_h6_out_depth = self.bn_depth_shortcut_h6_out(self.deconv_depth_shortcut_h6_out(h6_out_depth))
        h7_out_depth = F.leaky_relu(shortcut_h6_out_depth+h7_2_depth, 0.2)
        h7_out_depth = self.upsample_depth_h7_out(h7_out_depth)

        skip_s3_1_depth = F.leaky_relu(self.bn_depth_skip_s3_1(self.conv_depth_skip_s3_1(h1_out_og)), 0.2)
        skip_s3_2_depth = self.bn_depth_skip_s3_2(self.conv_depth_skip_s3_2(skip_s3_1_depth))
        skip_s3_out_depth = F.leaky_relu(h1_out_og+skip_s3_2_depth, 0.2)

        if(epoch > 12):
            h7_out_depth = h7_out_depth+skip_s3_out_depth

        h8_1_depth = F.leaky_relu(self.bn_depth_h8_1(self.deconv_depth_h8_1(h7_out_depth)), 0.2)
        h8_2_depth = self.bn_depth_h8_2(self.deconv_depth_h8_2(h8_1_depth))
        h8_out_depth = F.leaky_relu(h7_out_depth+h8_2_depth, 0.2)
        h8_out_depth = self.upsample_depth_h8_out(h8_out_depth)

        skip_s4_1_depth = F.leaky_relu(self.bn_depth_skip_s4_1(self.conv_depth_skip_s4_1(c1_og)), 0.2)
        skip_s4_2_depth = self.bn_depth_skip_s4_2(self.conv_depth_skip_s4_2(skip_s4_1_depth))
        skip_s4_out_depth = F.leaky_relu(c1_og+skip_s4_2_depth, 0.2)

        if(epoch > 14):
            h8_out_depth = h8_out_depth+skip_s4_out_depth

        c2_1_depth = F.leaky_relu(self.bn_depth_c2_1(self.conv_depth_c2_1(h8_out_depth)), 0.2)
        c2_2_depth = F.leaky_relu(self.bn_depth_c2_2(self.conv_depth_c2_2(c2_1_depth)), 0.2)
        c2_3_depth = F.leaky_relu(self.bn_depth_c2_3(self.conv_depth_c2_3(c2_2_depth)), 0.2)
        c2_o_depth = self.conv_depth_c2_o(c2_3_depth)

        #allow network to estimate smaller values
        c2_o_depth = 100.0*c2_o_depth

        #render image
        surface_normals = depth_to_normals(c2_o_depth+1610, intrinsic_matrix)
        surface_normals[:, 1, :, :] = -surface_normals[:, 1, :, :]

        points_3D = torch.cat((torch.reshape(self.xx, (self.batch_size, 1, self.img_height, self.img_width)), torch.reshape(self.yy, (self.batch_size, 1, self.img_height, self.img_width)), c2_o_depth), 1)
        tmp_incident_light = SL_lin2[:, :, :, 1:4].permute(0, 3, 1, 2)
        tmp_incident_light_z = torch.maximum(tmp_incident_light[:, 2], torch.tensor([[[0.0]], [[0.0]], [[0.0]]]).cuda())
        incident_light = torch.cat((tmp_incident_light[:, 0:2], torch.reshape(tmp_incident_light_z, (3, 1, 1, 1))), 1)
        incident_light = F.normalize(incident_light, p=2, dim=1)
        unit_light_direction = incident_light
        incident_light = self.light_distance*incident_light.repeat(1, 1, self.img_height, self.img_width)
        incident_light_points = incident_light
        incident_light = F.normalize(incident_light-points_3D, p=2, dim=1)
        surface_normals = F.normalize(surface_normals, p=2, dim=1)
        directional_component = self.directional_intensity*torch.maximum(torch.sum(surface_normals*incident_light, dim=1), torch.tensor([0.0]).cuda())
        ambient_values = SL_lin2[:, :, :, 0]
        ambient_light = ambient_values.repeat(1, self.img_height, self.img_width)
        full_shading = ambient_light+directional_component

        min_distance = torch.autograd.Variable(torch.empty((self.batch_size, self.img_height, self.img_width)), requires_grad=True)
        minimum_distance = min_distance.clone()

        for i in range(self.batch_size):
            points_A = points_3D[i, :, :, :]
            starting_points_xy = points_3D[i, 0:2, :, :]
            light_xy = incident_light_points[i, 0:2, :, :]
            slopes = (light_xy[1, :, :]-starting_points_xy[1, :, :])/(light_xy[0, :, :]-starting_points_xy[0, :, :]+0.0001)
            intercepts = light_xy[1, :, :]-slopes*light_xy[0, :, :]
            light_x = np.asscalar(light_xy[0, 0, 0].cpu().detach().numpy())
            light_y = np.asscalar(light_xy[1, 0, 0].cpu().detach().numpy())

            #initialization
            end_points = starting_points_xy

            if(light_x < -(self.img_width/2.0)):
                if(light_y < (1-(self.img_height/2.0))):
                    #try x=-(self.img_width/2.0)
                    x = -(self.img_width/2.0)*torch.ones((self.img_height, self.img_width))
                    y = slopes*x.cuda()+intercepts
                    end_points_x = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)
            
                    #try y=(1-(self.img_height/2.0))
                    y = (1-(self.img_height/2.0))*torch.ones((self.img_height, self.img_width))
                    x = (y.cuda()-intercepts)/(slopes+0.0001)
                    end_points_y = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)
                    intersects_y = torch.logical_and(x >= -(self.img_width/2.0), x <= (self.img_width-self.img_width/2.0-1))
                    end_points = end_points_y*intersects_y+end_points_x*torch.logical_not(intersects_y)     
                elif(light_y >= (1-(self.img_height/2.0)) and light_y <= self.img_height/2.0):
                    #x=-(self.img_width/2.0), it will intersect here
                    x = -(self.img_width/2.0)*torch.ones((self.img_height, self.img_width))
                    y = slopes*x.cuda()+intercepts
                    end_points = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)
                else:
                    #try x=-(self.img_width/2.0)
                    x = -(self.img_width/2.0)*torch.ones((self.img_height, self.img_width))
                    y = slopes*x.cuda()+intercepts
                    end_points_x = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)
            
                    #try y=(self.img_height/2.0)
                    y = (self.img_height/2.0)*torch.ones((self.img_height, self.img_width))
                    x = (y.cuda()-intercepts)/(slopes+0.0001)
                    end_points_y = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)
                    intersects_y = torch.logical_and(x >= -(self.img_width/2.0), x <= (self.img_width-self.img_width/2.0-1))
                    end_points = end_points_y*intersects_y+end_points_x*torch.logical_not(intersects_y)  
            elif(light_x >= -(self.img_width/2.0) and light_x <= (self.img_width-self.img_width/2.0-1)):
                if(light_y < (1-(self.img_height/2.0))):
                    #y=(1-(self.img_height/2.0)), it will intersect here
                    y = (1-(self.img_height/2.0))*torch.ones((self.img_height, self.img_width))
                    x = (y.cuda()-intercepts)/(slopes+0.0001)
                    end_points = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)
                elif(light_y >= (1-(self.img_height/2.0)) and light_y <= self.img_height/2.0):
                    x = light_x*torch.ones((self.img_height, self.img_width))
                    y = light_y*torch.ones((self.img_height, self.img_width))
                    end_points = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)
                else:
                    #y=(self.img_height/2.0), it will intersect here
                    y = (self.img_height/2.0)*torch.ones((self.img_height, self.img_width))
                    x = (y.cuda()-intercepts)/(slopes+0.0001)
                    end_points = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)
            else:
                if(light_y < (1-(self.img_height/2.0))):
                    #try x=(self.img_width-self.img_width/2.0-1)
                    x = (self.img_width-self.img_width/2.0-1)*torch.ones((self.img_height, self.img_width))
                    y = slopes*x.cuda()+intercepts
                    end_points_x = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)

                    #try y=(1-(self.img_height/2.0))
                    y = (1-(self.img_height/2.0))*torch.ones((self.img_height, self.img_width))
                    x = (y.cuda()-intercepts)/(slopes+0.0001)
                    end_points_y = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)
                    intersects_y = torch.logical_and(x >= -(self.img_width/2.0), x <= (self.img_width-self.img_width/2.0-1))
                    end_points = end_points_y*intersects_y+end_points_x*torch.logical_not(intersects_y)
                elif(light_y >= (1-(self.img_height/2.0)) and light_y <= self.img_height/2.0):
                    #x=(self.img_width-self.img_width/2.0-1)
                    x = (self.img_width-self.img_width/2.0-1)*torch.ones((self.img_height, self.img_width))
                    y = slopes*x.cuda()+intercepts
                    end_points = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)
                else:
                    #try x=(self.img_width-self.img_width/2.0-1)
                    x = (self.img_width-self.img_width/2.0-1)*torch.ones((self.img_height, self.img_width))
                    y = slopes*x.cuda()+intercepts
                    end_points_x = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)

                    #try y=(self.img_height/2.0)
                    y = (self.img_height/2.0)*torch.ones((self.img_height, self.img_width))
                    x = (y.cuda()-intercepts)/(slopes+0.0001)
                    end_points_y = torch.cat((torch.reshape(x.cuda(), (1, 256, 256)), torch.reshape(y.cuda(), (1, 256, 256))), 0)
                    intersects_y = torch.logical_and(x >= -(self.img_width/2.0), x <= (self.img_width-self.img_width/2.0-1))
                    end_points = end_points_y*intersects_y+end_points_x*torch.logical_not(intersects_y) 

            end_points[0, :, :][end_points[0, :, :] < -128] = -128.0
            end_points[0, :, :][end_points[0, :, :] > 127] = 127.0
            end_points[1, :, :][end_points[1, :, :] < -127] = -127.0
            end_points[1, :, :][end_points[1, :, :] > 128] = 128.0
            
            difference = end_points-starting_points_xy
            sample_increments = torch.reshape(torch.tensor(np.arange(0.025, 0.825, 0.005)), (self.num_sample_points, 1, 1, 1))
            sample_increments = sample_increments.repeat(1, 2, self.img_height, self.img_width)
            starting_points_xy = starting_points_xy.repeat(self.num_sample_points, 1, 1, 1)
            difference = difference.repeat(self.num_sample_points, 1, 1, 1)
            sampled_points = torch.round(starting_points_xy+sample_increments.cuda()*difference)
            sampled_points[:, 0, :, :] += (self.img_width/2.0)
            sampled_points[:, 1, :, :] = (self.img_height/2.0)-sampled_points[:, 1, :, :]
            sampled_indices = torch.round(sampled_points).int()
            sampled_indices = sampled_indices.permute(0, 2, 3, 1)
            sampled_indices = torch.reshape(sampled_indices, (self.num_sample_points*self.img_height*self.img_width, 2))
            sampled_3D_points = torch.reshape(points_A[:, sampled_indices[:, 1].long(), sampled_indices[:, 0].long()], (3, self.num_sample_points, self.img_height, self.img_width))

            sampled_points_unrounded = starting_points_xy+sample_increments.cuda()*difference
            sampled_points_unrounded[:, 0, :, :] += (self.img_width/2.0)
            sampled_points_unrounded[:, 1, :, :] = (self.img_height/2.0)-sampled_points_unrounded[:, 1, :, :]
            sampled_indices_unrounded = sampled_points_unrounded-0.0001
            sampled_indices_unrounded = sampled_indices_unrounded.permute(0, 2, 3, 1)
            sampled_indices_unrounded = torch.reshape(sampled_indices_unrounded, (self.num_sample_points*self.img_height*self.img_width, 2))
            sampled_indices_ceiled = torch.ceil(sampled_indices_unrounded).int() 
            sampled_indices_floored = torch.floor(sampled_indices_unrounded).int()
            sampled_depths_upper_left = points_A[2, sampled_indices_floored[:, 1].long(), sampled_indices_floored[:, 0].long()]
            sampled_depths_upper_right = points_A[2, sampled_indices_floored[:, 1].long(), sampled_indices_ceiled[:, 0].long()]
            sampled_depths_lower_left = points_A[2, sampled_indices_ceiled[:, 1].long(), sampled_indices_floored[:, 0].long()]
            sampled_depths_lower_right = points_A[2, sampled_indices_ceiled[:, 1].long(), sampled_indices_ceiled[:, 0].long()]
            sampled_depths_interpolated_x_upper = sampled_depths_upper_left*(sampled_indices_ceiled[:, 0]-sampled_indices_unrounded[:, 0])+sampled_depths_upper_right*(sampled_indices_unrounded[:, 0]-sampled_indices_floored[:, 0])
            sampled_depths_interpolated_x_lower = sampled_depths_lower_left*(sampled_indices_ceiled[:, 0]-sampled_indices_unrounded[:, 0])+sampled_depths_lower_right*(sampled_indices_unrounded[:, 0]-sampled_indices_floored[:, 0])
            sampled_depths_interpolated = sampled_depths_interpolated_x_upper*(sampled_indices_ceiled[:, 1]-sampled_indices_unrounded[:, 1])+sampled_depths_interpolated_x_lower*(sampled_indices_unrounded[:, 1]-sampled_indices_floored[:, 1])

            #modified code
            sampled_indices_unrounded = sampled_indices_unrounded.permute(1, 0)
            sampled_indices_unrounded[0, :] -= (self.img_width/2.0)
            sampled_indices_unrounded[1, :] = (self.img_height/2.0)-sampled_indices_unrounded[1, :]
            sampled_3D_points_interpolated = torch.reshape(torch.cat((sampled_indices_unrounded, torch.reshape(sampled_depths_interpolated, (1, self.num_sample_points*self.img_height*self.img_width))), 0), (3, self.num_sample_points, self.img_height, self.img_width))

            points_A = sampled_3D_points_interpolated.float()
            points_B = torch.reshape(points_3D[i, :, :, :], (3, 1, self.img_height, self.img_width)).repeat(1, self.num_sample_points, 1, 1)
            BA = points_A-points_B
            points_C = torch.reshape(incident_light_points[i, :, :, :], (3, 1, self.img_height, self.img_width))
            points_C = points_C.repeat(1, self.num_sample_points, 1, 1)
            BC = points_C-points_B
            cross_product = torch.cross(BA, BC, dim=0)
            point_to_line_distances = (torch.sqrt(torch.sum(cross_product*cross_product, dim=0)+0.0001))/(torch.sqrt(torch.sum(BC*BC, dim=0)+0.0001))
            outside_of_face = (masks[i, sampled_indices[:, 1].long(), sampled_indices[:, 0].long()] == 0)
            point_to_line_distances = torch.reshape(point_to_line_distances, (self.num_sample_points*self.img_height*self.img_width, 1))
            point_to_line_distances = torch.logical_not(outside_of_face)*point_to_line_distances+outside_of_face*1000000.0
            point_to_line_distances = torch.reshape(point_to_line_distances, (self.num_sample_points, self.img_height, self.img_width))
            (values, idx) = torch.min(point_to_line_distances, dim=0)
            minimum_distance[i, :, :] = values

        shadow_mask_weights = -4*torch.exp(-minimum_distance)/torch.pow((1+torch.exp(-minimum_distance)), 2)+1
        final_shading = shadow_mask_weights.cuda()*full_shading+(1-shadow_mask_weights.cuda())*ambient_light
        rendered_images = c2_o_albedo.clone()
        rendered_images[:, 0, :, :] = rendered_images[:, 0, :, :].clone()*final_shading
        rendered_images[:, 1, :, :] = rendered_images[:, 1, :, :].clone()*final_shading
        rendered_images[:, 2, :, :] = rendered_images[:, 2, :, :].clone()*final_shading

        return c2_o_albedo, c2_o_depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values
            

def load_data():
    training_images = np.zeros((29890, 256, 256, 3))
    training_lightings = np.zeros((29890, 4))
    training_depths = np.zeros((29890, 256, 256, 1))
    training_masks = np.zeros((29890, 256, 256, 1))
    training_albedo = np.zeros((29890, 256, 256))     
    training_masks_fill_nose_and_mouth = np.zeros((29890, 256, 256, 1)) 

    images = sorted(os.listdir('MP_data/CelebA-HQ_DFNRMVS_cropped/'))
    lightings = sorted(os.listdir('MP_data/lighting_directions_CelebAHQ_DFNRMVS/'))
    depths = sorted(os.listdir('MP_data/depth_maps_CelebA-HQ/'))
    masks = sorted(os.listdir('MP_data/depth_masks_CelebA-HQ_DFNRMVS/'))
    albedo = sorted(os.listdir('MP_data/CelebA-HQ_albedo_grayscale/'))

    training_lightings[:, 0] = 0.5

    for i in range(len(depths)):
        print(i)
        training_depths[i, :, :, :] = np.reshape(scipy.io.loadmat('MP_data/depth_maps_CelebA-HQ/'+depths[i])['depth_img'], (256, 256, 1))
        training_masks[i, :, :, :] = np.reshape(imageio.imread('MP_data/depth_masks_CelebA-HQ_DFNRMVS/'+masks[i]), (256, 256, 1))

        name_parts = depths[i].split('_')
        training_lightings[i, 1:4] = scipy.io.loadmat('MP_data/lighting_directions_CelebAHQ_DFNRMVS/'+name_parts[0]+'.jpg.mat')['lighting_direction']
        training_images[i, :, :, :] = imageio.imread('MP_data/CelebA-HQ_DFNRMVS_cropped/'+name_parts[0]+'.jpg')/255.0
        training_albedo[i, :, :] = imageio.imread('MP_data/CelebA-HQ_albedo_grayscale/'+name_parts[0]+'.jpg')
        tmp = np.reshape(imageio.imread('MP_data/CelebAHQ_face_masks/'+name_parts[0]+'.jpg'), (256, 256, 1))
        tmp = np.maximum(tmp, training_masks[i, :, :, :])
        tmp[tmp > 128] = 255.0
        tmp[tmp <= 128] = 0.0
        training_masks_fill_nose_and_mouth[i, :, :, :] = tmp
        
    return training_images, training_lightings, training_depths, training_masks, training_albedo, training_masks_fill_nose_and_mouth

def main():
    model = RelightNet()
    model = model.float()
    model = model.cuda()
    print(model)
    patchgan = PatchGAN()
    patchgan = patchgan.float()
    patchgan = patchgan.cuda()
    print(patchgan)
    (training_images, training_lightings, training_depths, training_masks, training_albedo, training_masks_fill_nose_and_mouth) = load_data()
    epoch = 200
    intrinsic_matrix = np.zeros((1, 3, 3))
    intrinsic_matrix[:, 0, 0] = 1570.0
    intrinsic_matrix[:, 1, 1] = 1570.0
    intrinsic_matrix[:, 2, 2] = 1.0
    intrinsic_matrix[:, 0, 2] = model.img_width/2.0
    intrinsic_matrix[:, 1, 2] = model.img_height/2.0
    intrinsic_matrix = torch.from_numpy(intrinsic_matrix)
    batch_list = np.arange(int(29890/model.batch_size))
    max_epoch = 1000
    num_batches = 700

    L1_loss = nn.L1Loss()
    L1_loss_sum = nn.L1Loss(reduction='sum')
    L2_loss_sum = nn.MSELoss(reduction='sum')
    BCE_loss = torch.nn.BCEWithLogitsLoss()
    fake_labels = torch.zeros([model.batch_size, 1, 15, 15], dtype=torch.float32)
    real_labels = torch.ones([model.batch_size, 1, 15, 15], dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr = model.lr)
    optimizer_patchgan = torch.optim.Adam(patchgan.parameters(), lr=model.lr)

    for i in range(max_epoch):
        np.random.shuffle(batch_list)
        total_loss_epoch = 0.0
        reconstruction_loss_epoch = 0.0
        depth_loss_epoch = 0.0
        ambient_loss_epoch = 0.0
        dir_lighting_loss_epoch = 0.0
        albedo_loss_epoch = 0.0
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        d_loss_real_epoch = 0.0
        d_loss_fake_epoch = 0.0
        DSSIM_loss_epoch = 0.0
        
        for j in range(700):
            curr_input_images = torch.from_numpy(training_images[(batch_list[j]*model.batch_size):((batch_list[j]+1)*model.batch_size)]) 
            curr_training_lightings = torch.from_numpy(training_lightings[(batch_list[j]*model.batch_size):((batch_list[j]+1)*model.batch_size)])
            curr_depth_maps = torch.from_numpy(training_depths[(batch_list[j]*model.batch_size):((batch_list[j]+1)*model.batch_size)])
            curr_masks = torch.from_numpy(training_masks[(batch_list[j]*model.batch_size):((batch_list[j]+1)*model.batch_size)])/255.0

            curr_masks_fill_nose_and_mouth = torch.from_numpy(training_masks_fill_nose_and_mouth[(batch_list[j]*model.batch_size):((batch_list[j]+1)*model.batch_size)])/255.0
            curr_masks_3_channels_fill_nose_and_mouth = curr_masks_fill_nose_and_mouth.permute(0, 3, 1, 2).repeat(1, 3, 1, 1)

            curr_albedo = torch.from_numpy(training_albedo[(batch_list[j]*model.batch_size):((batch_list[j]+1)*model.batch_size)])/255.0

            optimizer_patchgan.zero_grad()
            albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values = model(curr_input_images.float().cuda(), i, intrinsic_matrix.cuda(), curr_masks_fill_nose_and_mouth.cuda())
            logits_fake = patchgan(rendered_images*curr_masks_3_channels_fill_nose_and_mouth.float().cuda()+(1.0-curr_masks_3_channels_fill_nose_and_mouth.float().cuda())*curr_input_images.permute(0, 3, 1, 2).float().cuda())
            logits_real = patchgan(curr_input_images.permute(0, 3, 1, 2).float().cuda())
            d_loss_fake = 0.01*BCE_loss(logits_fake, fake_labels.cuda())
            d_loss_real = 0.01*BCE_loss(logits_real, real_labels.cuda())
            d_loss = d_loss_fake+d_loss_real
            if(j % model.GD_ratio == 0): 
                d_loss.backward(retain_graph=True)
                optimizer_patchgan.step()
            d_loss_epoch += d_loss.item()
            d_loss_real_epoch += d_loss_real.item()
            d_loss_fake_epoch += d_loss_fake.item()

            optimizer.zero_grad()
            
            reconstruction_loss = 20.0*L2_loss_sum(rendered_images*curr_masks_3_channels_fill_nose_and_mouth.cuda(), curr_input_images.permute(0, 3, 1, 2).float().cuda()*curr_masks_3_channels_fill_nose_and_mouth.cuda())/torch.sum(curr_masks_3_channels_fill_nose_and_mouth.cuda())
            depth_loss = L1_loss_sum(depth.permute(0, 2, 3, 1)*curr_masks.cuda(), curr_depth_maps.cuda()*curr_masks.cuda())/torch.sum(curr_masks.cuda())
            ambient_loss = 2.5*L1_loss(ambient_values, torch.reshape(curr_training_lightings[:, 0].cuda(), (model.batch_size, 1, 1)))
            dir_lighting_loss = torch.sum(1-torch.sum(unit_light_direction*torch.reshape(curr_training_lightings[:, 1:4].cuda(), (model.batch_size, 3, 1, 1)), dim=1))/model.batch_size
            grayscale_albedo = torch.mean(albedo, 1)
            
            albedo_loss = 5.0*L1_loss_sum(torch.reshape(grayscale_albedo, (model.batch_size, model.img_height, model.img_width, 1))*curr_masks_fill_nose_and_mouth.cuda(), torch.reshape(curr_albedo, (model.batch_size, model.img_height, model.img_width, 1)).cuda()*curr_masks_fill_nose_and_mouth.cuda())/torch.sum(curr_masks_fill_nose_and_mouth.cuda())

            logits_fake = patchgan(rendered_images*curr_masks_3_channels_fill_nose_and_mouth.float().cuda()+(1.0-curr_masks_3_channels_fill_nose_and_mouth.float().cuda())*curr_input_images.permute(0, 3, 1, 2).float().cuda())
            g_loss = 0.01*BCE_loss(logits_fake, real_labels.cuda())
            DSSIM_loss = 8.0*(1 - ssim(rendered_images*curr_masks_3_channels_fill_nose_and_mouth.float().cuda()+(1.0-curr_masks_3_channels_fill_nose_and_mouth.float().cuda())*curr_input_images.permute(0, 3, 1, 2).float().cuda(), curr_input_images.permute(0, 3, 1, 2).float().cuda(), data_range=1.0, size_average=True, nonnegative_ssim=True))/2.0

            total_loss = reconstruction_loss+depth_loss+ambient_loss+dir_lighting_loss+albedo_loss+g_loss+DSSIM_loss
     
            total_loss_epoch += total_loss.item()
            reconstruction_loss_epoch += reconstruction_loss.item()
            depth_loss_epoch += depth_loss.item()
            ambient_loss_epoch += ambient_loss.item()
            dir_lighting_loss_epoch += dir_lighting_loss.item()
            albedo_loss_epoch += albedo_loss.item()
            g_loss_epoch += g_loss.item()
            DSSIM_loss_epoch += DSSIM_loss.item()
            total_loss.backward()
            optimizer.step()
            print("Epoch: "+str(i)+", Batch: "+str(j))
            print("Total loss: "+str(total_loss.item()))
            print("Reconstruction loss: "+str(reconstruction_loss.item()))
            print("Depth loss: "+str(depth_loss.item()))
            print("Ambient loss: "+str(ambient_loss.item()))
            print("Lighting loss: "+str(dir_lighting_loss.item()))
            print("Albedo loss: "+str(albedo_loss.item()))
            print("Generator loss: "+str(g_loss.item()))
            print("Discriminator loss: "+str(d_loss.item()))
            print("Discriminator Real loss: "+str(d_loss_real.item()))
            print("Discriminator Fake loss: "+str(d_loss_fake.item()))
            print("DSSIM loss: "+str(DSSIM_loss.item()))
            print("\n")

        losses = {} 
        losses['total'] = total_loss_epoch/num_batches
        losses['recon'] = reconstruction_loss_epoch/num_batches
        losses['depth'] = depth_loss_epoch/num_batches
        losses['ambient'] = ambient_loss_epoch/num_batches
        losses['lighting'] = dir_lighting_loss_epoch/num_batches
        losses['albedo'] = albedo_loss_epoch/num_batches
        losses['generator'] = g_loss_epoch/num_batches
        losses['discriminator'] = d_loss_epoch/num_batches
        losses['discriminator_real'] = d_loss_real_epoch/num_batches
        losses['discriminator_fake'] = d_loss_fake_epoch/num_batches
        losses['DSSIM'] = DSSIM_loss_epoch/num_batches
        scipy.io.savemat('losses_raytracing_relighting_CelebAHQ_DSSIM_8x/losses_epoch'+str(i)+'.mat', losses)
        torch.save(model.state_dict(), 'saved_epochs_raytracing_relighting_CelebAHQ_DSSIM_8x/model_epoch'+str(i)+'.pth')
        torch.save(patchgan.state_dict(), 'saved_epochs_raytracing_relighting_CelebAHQ_DSSIM_8x/patchgan_epoch'+str(i)+'.pth')
            
if __name__ == '__main__':
    main()

    

