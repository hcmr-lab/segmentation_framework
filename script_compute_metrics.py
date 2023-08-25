import torch

import ricardo_setup
import argparse

import thop

parser = argparse.ArgumentParser(description='Train Segnet with pytorch')
parser.add_argument('--root',  type=str, default="/home/deep/Dropbox/SHARE/ricardoPereira/SUN_Segmentation")
parser.add_argument('--batch_size',  '-B',  type=int, default=32)
parser.add_argument('--weights',  '-EF', type=str, default="weights/BothDataset_deeplabv3plus_mobilenet_sunrgbd_RGB/iou_0.77.pth")

args = parser.parse_args()
    
dataset_root = args.root 

run_base = "BothDataset"
fusion_types  = ['RGB']
dataset_name = 'sunrgbd'
model_name   = 'deeplabv3plus_mobilenet'

dataloader,_,labels = ricardo_setup.inference_load_dataset_setup(dataset_root,
                                dataset_name,set_name='train',batch_size=32)

model = ricardo_setup.load_model_setup(model_name,len(labels))

for imgs,masks in dataloader:
    macs, params = thop.profile(model, inputs=(imgs,), verbose=False)
    macs_str, params_str = thop.clever_format([macs, params], "%.3f")
    print(macs_str, params_str)
        
    #macs, params 	= profile(model, inputs=(input ,))
    flops = 2*macs
    gflops = flops / 1e9

    print(f"GFLOPs: {gflops}")
    print(f"Params (M): {params/1e6}")