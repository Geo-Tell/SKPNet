# SKPNet
Official implementation of the paper "Learning Semantic Keypoints for Diverse Point Cloud Completion".

## Installation
 ### dependencies
```
pytorch
trimesh
plyfile
scipy
h5py
ninja
easydict
tensorboardX
einops
transforms3d
pytorch3d
```
 ###  cuda extensions
```
cd extensions/pointnet2_ops_lib
python setup.py install

cd extensions/chamfer_dist
python setup.py install
```

## Dataset
You can download the datasets from here [PCN](https://gateway.infinitescript.com/?fileName=ShapeNetCompletion),[MVP](https://www.dropbox.com/sh/la0kwlqx4n2s5e3/AACjoTzt-_vlX6OF9mfSpFMra?dl=0&lst=), and [ScanNet](https://drive.google.com/drive/folders/1GY7tlnLC5PMDZoae3JC7NxplMADjoQ76?usp=drive_link).
Please modify the dataset paths in the config_files under the [folder](cfgs/dataset_configs).

## Usage
### Training
For MVP dataset:
```
CUDA_VISIBLE_DEVICES=0 main.py --config ./cfgs/MVP_models/SKPNet.yaml --exp_name default (--start_val_epoch 100 --val_freq 10 --val_interval 2000) 
```
For ScanNet dataset, first pretrain on the MVP dataset and then finetune on the ScanNet dataset:
```
CUDA_VISIBLE_DEVICES=0 main.py --config ./cfgs/ScanNet_models/SKPNet.yaml --exp_name pretrain (--start_val_epoch 50 --val_freq 10 --val_interval 20) 
CUDA_VISIBLE_DEVICES=0 main.py --config ./cfgs/ScanNet_models/SKPNet_finetune.yaml --exp_name default --finetune --start_ckpts pth/to/pretrained/model/best.pth (--start_val_epoch 10 --val_freq 5 --val_interval 20) 
```


### Testing
```
CUDA_VISIBLE_DEVICES=0 python main.py --ckpts ckpts/MVP-best.pth --config ./cfgs/MVP_models/SKPNet.yaml --test --exp_name test_mvp
CUDA_VISIBLE_DEVICES=0 python main.py --ckpts ckpts/ScanNet-best.pth --config ./cfgs/ScanNet_models/SKPNet_finetune.yaml --test --exp_name test_scannet
```
 We provide our pretrained models [here]([https://drive.google.com/file/d/1SNudAqsxsrxlzVFlqJoxckJQV1mlpSRJ/view?usp=drive_link](https://drive.google.com/drive/folders/1u-yHgTP45rfqCor2IQhbQbPUwlgp9Mzg?usp=drive_link)).
