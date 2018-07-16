# Video Decaptioning for Chalearn Track 2

## Setup Models and Data

### Download VGG model weights
We use tensorflow implemention of VGG 16 based on [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg) for perceptual loss. Download the weights of the vgg model from [VGG16 NPY](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) and keep them in the code directory as './vgg16.npy'.

### Setting up Data
Save all the training images in './Data/train' folder in the respective './Data/train/X' and './Data/train/Y' folder and all the test images in './Data/test' folder in './Data/test/X' folder

### Download models
The pretrained models are kept at [modelfile](https://drive.google.com/drive/folders/1fAsd0eJVCoOXNi_xmJ947ze_hh7IikV-?usp=sharing). Download all the files and keep in their respective './model' folder.

## Runnung the code

### STAGE1:

step2: Train stage1 network for the decaptioning task which involves mask generation and inpainting of frames using the below mentioned python file, it will save the model files in './model' folder
```
python train.py 
```
step3: Pass the trainig data through the above trained network and obtain the mask and inpainted output of the first stage. Run below code for the same.
```
python test_val.py
```
update DATASET_PATH= ../Data/videos, savepath='../Output/stage1' and part ='train'

During testing change the DATASET_PATH ='../Data/videos', savepath='../Output/stage1/val' and part = 'val'


### STAGE2:

step4: Train stage2 network by executing below file,
```
python train_perceptual.py
```
step3: Test the above trained/pretrained network on the validation dataset, output will get saved in './Output/stage2' folder.
```
python test_val_stage2.py
```
