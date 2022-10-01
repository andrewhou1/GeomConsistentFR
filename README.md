# Face Relighting with Geometrically Consistent Shadows
[Andrew Hou](https://andrewhou1.github.io/), Michel Sarkis, Ning Bi, Yiying Tong, Xiaoming Liu. In CVPR, 2022. 

![alt text](https://github.com/andrewhou1/GeomConsistentFR/blob/main/Overview_Figure1_CVPR2022.png)

![Alt text](https://github.com/andrewhou1/GeomConsistentFR/blob/main/CVPR2022_relighting_video_final.gif)

The code for this project was developed using Python 3.8.8 and PyTorch 1.7.1. 

## Training Data 
The training data can be downloaded from: https://drive.google.com/file/d/1Jh4a5zvx92NRjC5E_MyaKMbFymx2Iw9E/view?usp=sharing 

The original CelebAHQ dataset can be downloaded from: https://github.com/switchablenorms/CelebAMask-HQ

These images will need to be cropped and will go into their own folder **MP_data/CelebA-HQ_DFNRMVS_cropped/**

For the cropping code specifically, we will need to create a separate conda environment to ensure that the images are cropped consistently to match the rest of the training data (albedo, depth maps, etc.) Note that this conda environment is different from the conda environment we will use for training and testing. 

To set up the conda environment for cropping the CelebA-HQ images, run the following:
```
conda create --name CelebAHQCrop --file cropping_dependencies.txt
conda activate CelebAHQCrop
pip install scipy
```
Once this environment is set up, place all of the CelebAHQ images that you downloaded in a directory named **input_image_dir/** and make a second directory called **output_image_dir/** which will store the cropped images. To crop the images, run the following command: **CUDA_VISIBLE_DEVICES=0 python recrop_CelebA-HQ_images.py**

## Create the conda environment for training and testing
```
conda create --name GeomShadows --file training_dependencies.txt
conda activate GeomShadows
pip3 install opencv-python
pip3 install kornia==0.4.1
pip install scipy
pip install imageio
```
## Training 
Once you've downloaded all of the training data from our google drive link (albedo, depth, etc.) and finished cropping the CelebAHQ images as described above, move the cropped CelebAHQ images from **output_image_dir/** to **MP_data/CelebA-HQ_DFNRMVS_cropped/**

To train the target lighting model, where the user can specify the desired lighting direction, make two additional directories to store the loss values and saved model weights: **losses_raytracing_relighting_CelebAHQ_DSSIM_8x/** and **saved_epochs_raytracing_relighting_CelebAHQ_DSSIM_8x/**

Finally, you can train the target lighting model using the following command: **CUDA_VISIBLE_DEVICES=0 python train_raytracing_relighting_CelebAHQ_DSSIM_8x.py**

If you instead would like to train the lighting transfer model, make the following two directories instead: **losses_lighting_transfer/** and **saved_epochs_lighting_transfer/**

To train the lighting transfer model, use the following command: **CUDA_VISIBLE_DEVICES=0 python train_lighting_transfer.py**

Be sure to use the correct conda environment (GeomShadows) for training. 

## Testing 
To run our testing code where the user specifies the target lighting, use the following command: **CUDA_VISIBLE_DEVICES=0 python test_relight_single_image.py**

You can specify both an input image and a target lighting direction within the code (we provide some sample lightings to generate the results in **FFHQ_relighting_results/**). You will also need a face mask for the input image (some examples are shown in **FFHQ_skin_masks/**). To generate a face mask, one option is to use the following repo: https://github.com/zllrunning/face-parsing.PyTorch 

Once a relit image is generated, use our **fix_border_artifacts_CVPR2022.m** postprocessing code to improve the face boundary if there are any dark or black pixels that appear along the boundary. This can sometimes happen as our method relights the face region only, and the boundary between the relit face and the original image may need to be smoothed. 

To run our testing code for lighting transfer, first make a directory called **lighting_transfer_results/** and then use the following command: **CUDA_VISIBLE_DEVICES=0 python test_relight_single_image_lighting_transfer.py input_image reference_image face_mask**

Be sure to use the correct conda environment (GeomShadows) for testing. 

## Citation 
If you utilize our code in your work, please cite our CVPR 2022 paper. 
```
@inproceedings{ face-relighting-with-geometrically-consistent-shadows,
  author = { Andrew Hou and Michel Sarkis and Ning Bi and Yiying Tong and Xiaoming Liu },
  title = { Face Relighting with Geometrically Consistent Shadows },
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = { 2022 }
}
```

## Contact 
If there are any questions, please feel free to post here or contact the first author at **houandr1@msu.edu** 
