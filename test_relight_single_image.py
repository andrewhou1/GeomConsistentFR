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

class RelightNet(nn.Module):
    def __init__(self):
        super(RelightNet, self).__init__()
        self.batch_size = 1
        self.img_height = 256
        self.img_width = 256
        self.lr = 0.0001
        self.df_dim = 64
        self.directional_intensity = 0.5
        self.light_distance = 4013.0
        self.num_sample_points = 160

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

    def forward(self, img, epoch, intrinsic_matrix, mask, target_lighting, target_ambient_values, batch_mask):
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

        #(batch_size, 3, 1, 1)
        incident_light = target_lighting
        incident_light = F.normalize(incident_light, p=2, dim=1)
        unit_light_direction = incident_light
        incident_light = self.light_distance*incident_light.repeat(1, 1, self.img_height, self.img_width)
        incident_light_points = incident_light
        incident_light = F.normalize(incident_light-points_3D, p=2, dim=1)
        surface_normals = F.normalize(surface_normals, p=2, dim=1)
        directional_component = self.directional_intensity*torch.maximum(torch.sum(surface_normals*incident_light, dim=1), torch.tensor([0.0]).cuda())
      
        #For qualitative results feel free to tune the ambient intensity to produce the best results. This will mostly affect the shadow intensity and is flexible since the user can specify their desired target lighting.
        ambient_light = SL_lin2[:, :, :, 0]-0.1
   
        ambient_values = ambient_light
        ambient_light = ambient_light.repeat(1, self.img_height, self.img_width)
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
            numerator = torch.sqrt(torch.sum(cross_product*cross_product, dim=0)+0.0001)
            denominator = torch.sqrt(torch.sum(BC*BC, dim=0)+0.0001)
            point_to_line_distances = numerator/denominator
            outside_of_face = (mask[sampled_indices[:, 1].long(), sampled_indices[:, 0].long()] == 0)
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

        return c2_o_albedo, c2_o_depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals

def main():
    model = RelightNet()
    model.load_state_dict(torch.load('model/model_epoch99.pth'))
    model = model.float()
    model = model.cuda()
    model.eval()
    training_images = np.zeros((1, 256, 256, 3))
    img_name = '00295.png'
    training_images[0, :, :, :] = cv2.resize(imageio.imread('sample_test_images_FFHQ/'+img_name)/255.0, (256, 256))
    training_lightings = np.zeros((1, 4))
    training_lightings[0, 0] = 0.5
   
    #Multi-PIE lighting 4, used to generate 00110.png, 00300.png, 00525.png 
    '''training_lightings[0, 1] = 0.7518
    training_lightings[0, 2] = 0.0
    training_lightings[0, 3] = 0.6594'''
    #Multi-PIE lighting 14, used to generate 00104.png
    '''training_lightings[0, 1] = 0.6893
    training_lightings[0, 2] = 0.3991
    training_lightings[0, 3] = 0.6047'''
    #Multi-PIE lighting 5
    '''training_lightings[0, 1] = 0.5145
    training_lightings[0, 2] = 0.0
    training_lightings[0, 3] = 0.8575'''
    #Multi-PIE lighting 9, used to generate 00290.png
    '''training_lightings[0, 1] = -0.5843
    training_lightings[0, 2] = 0.0
    training_lightings[0, 3] = 0.8115'''
    #Multi-PIE lighting 10, used to generate 00322.png, 00572.png
    '''training_lightings[0, 1] = -0.7574
    training_lightings[0, 2] = 0
    training_lightings[0, 3] = 0.6529'''
    #Multi-PIE lighting 18
    '''training_lightings[0, 1] = -0.7076
    training_lightings[0, 2] = 0.3892
    training_lightings[0, 3] = 0.5897'''
    #Multi-PIE lighting 17, used to generate 00695.png
    '''training_lightings[0, 1] = -0.5151
    training_lightings[0, 2] = 0.4722
    training_lightings[0, 3] = 0.7154'''
    #Multi-PIE lighting 15
    '''training_lightings[0, 1] = 0.4478
    training_lightings[0, 2] = 0.4925
    training_lightings[0, 3] = 0.7463'''
    #A00E45, top lighting, used to generate 00295.png
    training_lightings[0, 1] = 0
    training_lightings[0, 2] = 0.7071
    training_lightings[0, 3] = 0.7071
    #A60E-20, bottom left
    '''training_lightings[0, 1] = -0.8138
    training_lightings[0, 2] = -0.3420
    training_lightings[0, 3] = 0.4698'''
    #A-60E-20, bottom right, used to generate 00508.png
    '''training_lightings[0, 1] = 0.8138
    training_lightings[0, 2] = -0.3420
    training_lightings[0, 3] = 0.4698'''

    training_masks = np.zeros((1, 256, 256, 1))
    training_masks[0, :, :, :] = np.reshape(imageio.imread('FFHQ_skin_masks/'+img_name), (256, 256, 1))
    training_masks_fill_nose = np.zeros((1, 256, 256, 1))
    training_masks_fill_nose[0, :, :, :] = np.reshape(imageio.imread('FFHQ_skin_masks/'+img_name), (256, 256, 1))
    epoch = 200
    intrinsic_matrix = np.zeros((1, 3, 3))
    intrinsic_matrix[:, 0, 0] = 1570.0
    intrinsic_matrix[:, 1, 1] = 1570.0
    intrinsic_matrix[:, 2, 2] = 1.0
    intrinsic_matrix[:, 0, 2] = model.img_width/2.0
    intrinsic_matrix[:, 1, 2] = model.img_height/2.0
    intrinsic_matrix = torch.from_numpy(intrinsic_matrix)
    batch_list = np.arange(1)
    num_batches = 1

    L1_loss = nn.L1Loss()
    L1_loss_sum = nn.L1Loss(reduction='sum')

    with torch.no_grad():
        for j in range(num_batches):
            curr_input_images = torch.from_numpy(training_images[(batch_list[j]*model.batch_size):((batch_list[j]+1)*model.batch_size)]) 
            curr_training_lightings = torch.from_numpy(training_lightings[(batch_list[j]*model.batch_size):((batch_list[j]+1)*model.batch_size)])
            curr_mask = torch.from_numpy(training_masks[batch_list[j]])/255.0
            batch_mask = curr_mask.repeat(model.batch_size, 1, 1, 1)
            albedo, depth, shadow_mask_weights, ambient_light, full_shading, rendered_images, unit_light_direction, ambient_values, final_shading, surface_normals = model(curr_input_images.float().cuda(), epoch, intrinsic_matrix.cuda(), curr_mask.cuda(), torch.reshape(curr_training_lightings[:, 1:4].float().cuda(), (model.batch_size, 3, 1, 1)), torch.reshape(curr_training_lightings[:, 0].float().cuda(), (model.batch_size, 1, 1)), batch_mask.cuda())
            
            rendered_images = rendered_images.permute(0, 2, 3, 1)
            rendered_images = rendered_images.cpu().numpy()
            albedo = albedo.permute(0, 2, 3, 1)
            albedo = albedo.cpu().numpy()
            depth = depth.permute(0, 2, 3, 1)
            depth = depth.cpu().numpy()
            depth = -depth
            depth = (depth-np.amin(depth))/(np.amax(depth)-np.amin(depth))
            final_shading = final_shading.cpu().numpy()
            surface_normals = surface_normals.permute(0, 2, 3, 1)
            surface_normals = surface_normals.cpu().numpy()
            surface_normals = 255.0*(surface_normals+1.0)/2.0

            curr_mask_3_channels = np.zeros((model.img_height, model.img_width, 3))
            curr_mask_3_channels[:, :, 0] = np.reshape(curr_mask.numpy(), (model.img_height, model.img_width))
            curr_mask_3_channels[:, :, 1] = np.reshape(curr_mask.numpy(), (model.img_height, model.img_width))
            curr_mask_3_channels[:, :, 2] = np.reshape(curr_mask.numpy(), (model.img_height, model.img_width))

            curr_mask_fill_nose = torch.from_numpy(training_masks_fill_nose[batch_list[j]])/255.0
            curr_mask_fill_nose_3_channels = np.zeros((model.img_height, model.img_width, 3))
            curr_mask_fill_nose_3_channels[:, :, 0] = np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width))
            curr_mask_fill_nose_3_channels[:, :, 1] = np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width))
            curr_mask_fill_nose_3_channels[:, :, 2] = np.reshape(curr_mask_fill_nose.numpy(), (model.img_height, model.img_width))

            for k in range(model.batch_size):
                name_parts = img_name.split('.')
                input_image = training_images[batch_list[j]*model.batch_size+k]*255.0
                input_image = input_image[:, :, ::-1]
                rendered_image = 255.0*rendered_images[k, :, :, ::-1]*curr_mask_fill_nose_3_channels
                input_image[curr_mask_fill_nose_3_channels > 0] = rendered_image[curr_mask_fill_nose_3_channels > 0]
                cv2.imwrite('FFHQ_relighting_results/'+name_parts[0]+'_rendered_image.png', input_image)
            
if __name__ == '__main__':
    main()

    

