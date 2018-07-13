step 0_1: Save the given videos dataset in ./Data/videos/train folder 

step0_2: Break each video into 125 frames and save all the frames of training data in './Data/frames/train' folder in the respective './Data/frames/train/X' and './Data/frames/train/Y' folder and all the val datas in './Data/val' folder in './Data/val/X' folder.

step1: Generate ground truth masks for each frame of the training videos using below file-
python genMask.py

STAGE1:

step2: Train stage1 network for the decaptioning task which involves mask generation and inpainting of frames using the below mentioned python file, it will save the model files in './model' folder
python train.py 

step3: Pass the trainig data through the above trained network and obtain the mask and inpainted output of the first stage. Run below code for the same.
python test_val.py
update DATASET_PATH= ../Data/videos, savepath='../Output/stage1' and part ='train'

During testing change the DATASET_PATH ='../Data/videos', savepath='../Output/stage1/val' and part = 'val'


STAGE2:

step4: Train stage2 network by executing below file,
python train_perceptual.py

step3: Test the above trained/pretrained network on the validation dataset, output will get saved in './Output/stage2' folder.
python test_val_stage2.py

