all_img_files = dir('MP_data/test_raytracing_relighting_CelebAHQ_DSSIM_8x/');
all_img_files = all_img_files(3:end);
%We save out 6 files per image including the rendered image, surface normals, depth map, albedo, etc. We only want to compare against the rendered images (hence 3:6:end)
all_img_files = all_img_files(3:6:end);

batch_mask_files = dir('MP_data/MP_depth_masks_fill_nose/');
batch_mask_files = batch_mask_files(3:end);

groundtruth_img_files = dir('MP_data/groundtruth_images_MP_18_lightings/');
groundtruth_img_files = groundtruth_img_files(3:end);

MSE = zeros(1, 862);

for i = 1:862
       curr_recon_img = double(imread(sprintf('MP_data/test_raytracing_relighting_CelebAHQ_DSSIM_8x/%s', all_img_files(i).name)));
       curr_gt_img = double(imread(sprintf('MP_data/groundtruth_images_MP_18_lightings/%s', groundtruth_img_files(i).name)));
       curr_batch_mask = double(imread(sprintf('MP_data/MP_depth_masks_fill_nose/%s', batch_mask_files(i).name)))/255.0;
       curr_batch_mask_3_channels = zeros(256, 256, 3);
       curr_batch_mask_3_channels(:, :, 1) = curr_batch_mask;
       curr_batch_mask_3_channels(:, :, 2) = curr_batch_mask;
       curr_batch_mask_3_channels(:, :, 3) = curr_batch_mask;
       curr_recon_img = curr_recon_img/255.0;
       curr_gt_img = curr_gt_img/255.0;
       MSE(i) = sum(sum(sum(abs(curr_recon_img.*curr_batch_mask-curr_gt_img.*curr_batch_mask).^2)))/(3*sum(sum(curr_batch_mask)));
end
