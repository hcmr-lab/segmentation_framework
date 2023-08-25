
import numpy as np
import torch
from torch.utils import data
import cv2

import os
from . import transforms as T
torch

PREPROCESSING = T.Compose([T.ToTensor()])


def expand_mask(mask,num_class,height,width):
    
    mask_expand = np.zeros((height,width,num_class+1),np.uint8)
    labels = np.unique(mask)
    for label in labels:
        mask_expand[:,:,label] = 1

    return mask_expand
     
def convert_mask(masks,num_class,height,width):
    import tqdm
    expand_masks = []
    for mask in tqdm.tqdm(masks,"Convert masks"):
        exp_mask = expand_mask(mask,num_class,height,width)
        expand_masks.append(exp_mask)

    return expand_masks
# --------------------------------------------------------- #
# 						  Dataset  							#
# --------------------------------------------------------- #
class sunrgbd(data.Dataset):
    def __init__(self, data_path, splitset, dataset_dir = 'SUN_Segmentation',img_size=224,num_class=37):
        if splitset in ['train','training']:
            raw_rgb_path = os.path.join(data_path,dataset_dir,'x_rgb_224_training.npy')
            
            labels_path = os.path.join(data_path,dataset_dir,'Y_Seg_37Labels_training_224.npy')
        if splitset in ['test','testing']:
            raw_rgb_path = os.path.join(data_path,dataset_dir,'x_rgb_224_testing.npy')
            labels_path = os.path.join(data_path,dataset_dir,'Y_Seg_37Labels_testing_224.npy')
        
        self.RGB_imgs = np.load(raw_rgb_path)
        gt_masks = np.load(labels_path)
        
        self.classes = np.unique(gt_masks)
        self.num_classes = len(self.classes)
        self.gt_masks = gt_masks #convert_mask(gt_masks,self.num_classes,img_size,img_size)
        self.RGB_imgs = self.resizeData(self.RGB_imgs, img_size)
        

    def resizeData(self, data, size):
        if data.shape[1] == size: return data

        print('Resizing...')

        resize_data = []
        for idx, img in enumerate(data):
            img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
            resize_data.append(img)

        resized_data = np.array(resized_data)

        return resized_data

    def __len__(self):
        return len(self.RGB_imgs)

    def __getitem__(self, index):
        img = self.RGB_imgs[index,:,:,:]
        mask = self.gt_masks[index]
        img,gt_mask = PREPROCESSING(img,mask)


        return img,gt_mask


class mit67(data.Dataset):
    def __init__(self, data_path, splitset, dataset_dir = 'MIT67_Generate_SegMasks',img_size=224):
        if splitset in ['train','training']:
            raw_rgb_path = os.path.join(data_path,dataset_dir,'x_rgb_224_training.npy')
        if splitset in ['test','testing']:
            raw_rgb_path 	= os.path.join(data_path,dataset_dir,'x_rgb_224_testing.npy')
        
        self.RGB_imgs = np.load(raw_rgb_path)
        self.RGB_imgs = self.resizeData(self.RGB_imgs, img_size)
        self.num_classes = 37

    def resizeData(self, data, size):
        if data.shape[1] == size: return data

        print('Resizing...')

        resize_data = []
        for idx, img in enumerate(data):
            img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
            resize_data.append(img)

        resized_data = np.array(resized_data)

        return resized_data

    def __len__(self):
        return len(self.RGB_imgs)

    def __getitem__(self, index):
        img = self.RGB_imgs[index,:,:,:]
        img = torch.from_numpy(img).float()
        img = img.permute(2,0,1)

        return img,np.array([])




