import os 
import cv2
import sys
import glob
import random
import pickle
import shutil
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from . import transforms as T
import torchvision.transforms as Tr
from tqdm import tqdm


labels = [
	[0,		'Ignore',			[0	,	0,	0]],	
	[1,		'wall',				[98,   59, 14]],
 	[2,		'floor',			[12,   57, 226]],
 	[3,		'cabinet',			[ 83, 144,  31]],
 	[4,		'bed',				[161, 141,  54]],
 	[5,		'chair',			[165, 189, 218]],
 	[6,		'sofa',				[241,  51,  10]],
 	[7,		'table',			[ 11,  97,  48]],
 	[8,		'door',				[159, 177, 237]],
 	[9,		'window',			[33,  107,  43]],
 	[10,	'bookshelf',		[135, 248, 213]],
 	[11,	'picture',			[225,  79, 175]],
 	[12,	'counter',			[ 14, 232,   9]],
 	[13,	'blinds',			[ 91, 96, 164]],
	[14,	'desk',				[101, 42,  34]],
 	[15,	'shelves',			[ 69, 78, 232]],
 	[16,	'curtain',			[19, 180, 97]],
 	[17,	'dresser',			[44, 182, 106]],
 	[18,	'pillow',			[205, 142, 95]],
 	[19,	'mirror',			[ 42,  56,216]],
 	[20,	'floor_mat',		[186, 194,193]],
 	[21,	'clothes',			[ 34,  21,  6]],
 	[22,	'ceiling',			[119, 151, 94]],
 	[23,	'books',			[ 71, 237,  2]],
 	[24,	'fridge',			[227, 246, 31]],
 	[25,	'tv',				[170,  82, 51]],
 	[26,	'paper',			[ 48, 167, 191]],
 	[27,	'towel',			[25, 7, 159]],
 	[28,	'shower_curtain',	[135, 99, 144]],
 	[29,	'box',				[35, 63, 245]],
 	[30,	'whiteboard',		[58, 51, 121]],
 	[31,	'person',			[208, 213, 105]],
 	[32,	'night_stand',		[113, 195, 221]],
 	[33,	'toilet',			[52, 10, 95]],
 	[34,	'sink',				[136, 125, 156]],
 	[35,	'lamp',				[23, 252, 112]],
 	[36,	'bathtub',			[40, 124, 164]],
 	[37,	'bag',				[12, 65, 184]]
]

IGNORE_LABEL = 255
NUM_LABELS   = len(labels)

PREPROCESSING = T.Compose([T.ToTensor(),T.Resize([256,256])])
VIZ_TRANSFORM = Tr.ToTensor()
RESTORE       = Tr.ToPILImage()

#LABEL_DISTRO  = label_disto

LABELS = [label[1] for label in labels]
labelid    = [label[0] for label in labels]
id2color   = {label[0]:label[2] for label in labels}
label2name = {label[0]:label[1] for label in labels}
label2id   = {label[1]:label[0] for label in labels}
id2label   = {label[0]:label[0] for label in labels}
label2color= {label[0]:label[2] for label in labels}

def img_to_tensorboard(rgb,dsm,mask,pred):
    
	assert len(rgb.shape) == 3  # (C,H,W)
	assert rgb.shape[0] == 3    # C = [R G B] 
	
	if mask.shape[0] > 1 :
		mask = torch.argmax(mask,dim=0,keepdim=True)
	
	if pred.shape[0] > 1:
		pred = torch.argmax(pred,dim=0,keepdim=True)
	
	mask = conv_mask_to_img_torch(mask)
	pred = conv_mask_to_img_torch(pred)
	
	dsm = dsm.numpy().astype(np.uint8)
	dsm = torch.tensor(dsm,dtype = torch.uint8)
	
	img = (rgb,dsm,mask,pred)
	img = [RESTORE(x)       for x in img] 
	img = [x.convert('RGB') for x in img]
	img = [VIZ_TRANSFORM(x) for x in img]
	
	img_cat= torch.cat(img,dim=2)
	return({'tb':img_cat,'img':img})

def conv_mask_to_img_torch(mask):
    mask  =conv_mask_to_img_np(mask)
    mask = torch.tensor(mask,dtype = torch.uint8)
    mask = torch.permute(mask,(-1,0,1))
    return(mask)

def conv_mask_to_img_np(mask):
	if not isinstance(mask,(np.ndarray, np.generic)):
		mask = np.array(mask)
	if mask.shape[0]<mask.shape[-1]:
		mask = np.transpose(mask,(1,2,0))
	mask = mask.squeeze()
	mask_png   = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.float32)
	unq_labels = np.unique(mask)
	
	for i,label in enumerate(unq_labels):
		name = label2name[label]
		color_vector = list(label2color[label])
		mask_png[mask == label,:]=color_vector
	return(mask_png)

def conv_img_to_mask_np(mask):
	if not isinstance(mask,(np.ndarray, np.generic)):
		mask = np.array(mask)
	if mask.shape[0]<mask.shape[0-1]:
		mask = np.transpose(mask,(-1,0,1))
	if len(mask.shape)<3:
		mask = np.expand_dims(mask,axis=0)
	
	mask  = mask.squeeze()
	shape =  mask.shape
	unq_id= np.unique(mask)
	new_mask = np.zeros((shape[0],shape[1],NUM_LABELS),dtype=np.uint8)

	for id in unq_id:
		#if (id == 0):
		#	continue
		mask_bin = mask == id
		new_mask[:,:,id] = mask_bin
	return(new_mask)

# --------------------------------------------------------- #
# 					Pytorch Data Class  					#
# --------------------------------------------------------- #

class SUNRGBDDataset(data.Dataset):
	# http://rgbd.cs.princeton.edu/challenge.html
	def __init__(self, root,set_folder,dataset_mode='DISK'):
		
		self.num_classes = NUM_LABELS
		self.labels = LABELS
		self.data_path = root
		self.dataset_mode = dataset_mode
		if not set_folder in ['train','test']:
			raise NameError

		self.preprocessing = PREPROCESSING
		self.data_path = os.path.join(root,'SUNRGBD',set_folder)

		self.rgb_raw_path 		= sorted(glob.glob(os.path.join(self.data_path,'RGB','*.jpg')),key=os.path.getmtime)
		rgb_name = [int(file.split('.')[0].split(os.sep)[-1].split('_')[0].split('ID')[-1]) for file in self.rgb_raw_path ]
		indices   = np.argsort(rgb_name)
		self.rgb_raw_path = np.array(self.rgb_raw_path)[indices]
		#[print(f) for f in self.rgb_raw_path[:10]]
		self.depth_raw_path 	= sorted(glob.glob(os.path.join(self.data_path,'Depth','*.png')),key=os.path.getmtime)
		rgb_name = [int(file.split('.')[0].split(os.sep)[-1].split('_')[0].split('ID')[-1])  for file in self.depth_raw_path]
		indices   = np.argsort(rgb_name)
		self.depth_raw_path = np.array(self.depth_raw_path)[indices]
		#[print(f) for f in self.depth_raw_path[:10]]
		self.masks_raw_path 	= sorted(glob.glob(os.path.join(self.data_path,'Seg_Masks_37','*.pkl')),key=os.path.getmtime)
		rgb_name = [int(file.split('.')[0].split(os.sep)[-1].split('_')[0].split('ID')[-1])  for file in self.masks_raw_path]
		indices   = np.argsort(rgb_name)
		self.masks_raw_path = np.array(self.masks_raw_path)[indices]
		#[print(f) for f in self.masks_raw_path[:10]]
		self.seg_local_classes 	= sorted(glob.glob(os.path.join(self.data_path,'Seg_Masks_37','*.txt')),key=os.path.getmtime)
		rgb_name = [int(file.split('.')[0].split(os.sep)[-1].split('_')[0].split('ID')[-1]) for file in self.seg_local_classes]
		indices   = np.argsort(rgb_name)
		self.seg_local_classes = np.array(self.seg_local_classes)[indices]
		#[print(f) for f in self.seg_local_classes[:10]]

		if self.dataset_mode == 'RAM':
			self.load_dataset_to_RAM()

	def load_dataset_to_RAM(self):
		self.rgb_bag = []
		self.dsm_bag = []
		self.mask_bag =[]
		self.names_bag=[]
		self.scene_label_bag = []
		self.seg_local_classes_bag =[]
		
		self.dataset_len = len(self.rgb_raw_path)
		
		for index in tqdm(range(self.dataset_len),"Loading to RAM: "):
			rgb, depth, mask, scene_label = self.load_data_37(index)
			
			mask[mask>0]=1 # Needed because resize causes label distorchen
			self.rgb_bag.append(rgb)
			self.dsm_bag.append(depth)
			self.mask_bag.append(mask)
			self.names_bag.append(index)
			self.scene_label_bag.append(scene_label)
			# self.seg_local_classes_bag.append(seg_local_classes)

	def load_data_37(self, index):
		# Get RGB Image
		rgb_img = cv2.imread(self.rgb_raw_path[index], cv2.IMREAD_UNCHANGED)
		# Get Depth Image
		depth_img = cv2.imread(self.depth_raw_path[index], cv2.IMREAD_UNCHANGED)
		depth_img = np.array(depth_img, dtype= np.int32)
		# Tenho ideia que li que se quiseres os valores em metros é dividir por 10000... MAS não tou a conseguir encontrar essa info
		# Get Segmentation Masks
		with open(self.masks_raw_path[index], 'rb') as fin:
			seg_mask = pickle.load(fin)
		seg_mask = np.array(seg_mask,dtype=np.uint8)
		seg_mask = np.expand_dims(seg_mask,axis=-1)

		mask = conv_img_to_mask_np(seg_mask)
		rgb,depth,mask = self.preprocessing(rgb_img,depth_img,mask)

		# Get Scene Label
		scene_label = int(self.rgb_raw_path[index].split('\\')[-1].split('.')[0].split('_')[1])

		return rgb, depth, mask, scene_label

	def get_data(self,indx):
		
		if self.dataset_mode == 'RAM':
			rgb = self.rgb_bag[indx]
			dsm = self.dsm_bag[indx]
			mask = self.mask_bag[indx]
			file_name= self.names_bag[indx]
			scene_label= self.scene_label_bag[indx]
			#seg_local_classes = self.seg_local_classes_bag[indx]
		else: 
			# Read from DISK
			rgb, depth, mask, scene_label = self.load_data_37(indx)
			mask[mask>0]=1 # Needed because resize causes label distorchen
			file_name = indx

		rgb = rgb.type(torch.float32)
		depth = depth.type(torch.float32)
		mask = mask.type(torch.float32)

		return rgb, depth, mask,scene_label

	def __len__(self):
		return len(self.rgb_raw_path)


	def __getitem__(self, index):
		rgb_img, depth_img, mask, scene_class = self.get_data(index)

		return(rgb_img, depth_img, mask, scene_class)


class SUNRGBD():
    def __init__(self,**kwargs):
        
        root =  kwargs['root']
        #root = kwargs['root']
        
        train_config = kwargs['train_loader']
        val_config   = kwargs['val_loader']
        dataset_mode= kwargs['dataset_mode']

        #self.label_disto = label_disto
        self.train_loader = SUNRGBDDataset(root = root, 
                                    set_folder = 'train',
                                    dataset_mode = dataset_mode,
                                    #**train_config
									)

        self.val_loader = SUNRGBDDataset(root = root, 
                                    set_folder = 'test',
                                    dataset_mode = dataset_mode,
                                    #**val_config
									)
        
        self.trainloader   = DataLoader(self.train_loader,
                                    batch_size = train_config['batch_size'],
                                    shuffle    = train_config['shuffle'],
                                    num_workers= train_config['num_workers']
                                    )
        
        self.valloader   = DataLoader(self.val_loader,
                                    batch_size = val_config['batch_size'],
                                    shuffle    = val_config['shuffle'],
                                    num_workers= val_config['num_workers']
                                    )

    def get_train_loader(self):
        return self.trainloader 
    

    def get_test_loader(self):
        raise NotImplementedError
    
    def get_val_loader(self):
        return  self.valloader
    
    def get_label_distro(self):
        return(np.ones(NUM_LABELS))
		#return  1-np.array(self.label_disto)



	



