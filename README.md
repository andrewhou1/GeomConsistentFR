# Face Relighting with Geometrically Consistent Shadows
Andrew Hou, Michel Sarkis, Ning Bi, Yiying Tong, Xiaoming Liu. In CVPR, 2022. 

![alt text](https://github.com/andrewhou1/GeomConsistentFR/blob/main/Overview_Figure1_CVPR2022.png)

![](https://github.com/andrewhou1/GeomConsistentFR/blob/main/CVPR2022_relighting_video_final.gif)

The code for this project was developed using Python 3 and PyTorch 1.7.1. 

## Training Data 
The training data can be downloaded from: https://drive.google.com/file/d/1Jh4a5zvx92NRjC5E_MyaKMbFymx2Iw9E/view?usp=sharing 

The original CelebAHQ dataset can be downloaded from: https://github.com/switchablenorms/CelebAMask-HQ

These images will need to be cropped and will go into their own folder MP_data/CelebA-HQ_DFNRMVS_cropped/

For the cropping code specifically, we will need to create a separate conda environment to ensure that the images are cropped consistently to match the rest of the training data (albedo, depth maps, etc.) Note that this conda environment is different from the conda environment we will use for training and testing. 

## Testing 
To run our testing code, use the following command: "CUDA_VISIBLE_DEVICES=0 python test_relight_single_image.py"

You can specify both an input image and a target lighting direction within the code (we provide some sample lightings to generate the results in FFHQ_relighting_results/). You will also need a face mask for the input image (some examples are shown in FFHQ_skin_masks/). To generate a face mask, one option is to use the following repo: https://github.com/zllrunning/face-parsing.PyTorch 

Once a relit image is generated, use our fix_border_artifacts_CVPR2022.m postprocessing code to improve the face boundary if there are any dark or black pixels that appear along the boundary. This can sometimes happen as our method relights the face region only, and the boundary between the relit face and the original image may need to be smoothed. 

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
If there are any questions, please feel free to post here or contact the authors at {houandr1, ytong, liuxm}@msu.edu, {msarkis, nbi}@qti.qualcomm.com
