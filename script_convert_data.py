

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')  # or try 'Qt5Agg', 'Agg', etc.

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

label2color= {label[0]:label[2] for label in labels}

if __name__=='__main__':
    root = "weights/BothDataset_deeplabv3plus_mobilenet_sunrgbd_train"
    target_dir = "pred_masks_iou_0.77"
    path_to_target_dir = os.path.join(root,target_dir)
    
    files =  np.sort([os.path.join(path_to_target_dir,file) for file in os.listdir(path_to_target_dir)])
    
    print(files[0])
    #fig, axes = plt.subplots(1, 2)
    img_path = os.path.join(root,"img_"+target_dir)
    os.makedirs(img_path,exist_ok=True)
    
    print(len(files))
    
    for file in files:
        print(file)
        img_file = file.split("/")[-1].split('.')[0]
        fig, axes = plt.subplots(1,3)
        full_img_path = os.path.join(img_path,img_file+ '.png')
        with open(file, 'rb') as handle:
            
            data = pickle.load(handle)
        
            # Plot the RGB image
            img = data['input'].cpu().detach().numpy()
            img = img*255
            img = np.swapaxes(img, -1, 0)
            img = np.swapaxes(img, 1, 0)
            
            img = img.astype(np.uint8)
            axes[0].imshow(img)
            axes[0].axis('off')
            axes[0].set_title('Image')
            
            # Plot the corresponding segmentation ground truth mask
            
            mask = data['mask'].cpu().detach().numpy()
            if mask.shape[0]:
                unique_labes = np.unique(mask).astype(np.uint8)
                w,h = mask.shape
                mask_img = np.zeros((w,h,3),dtype=np.uint8)
                for label in unique_labes:
                    mask_img[mask == label] = label2color[label]
            else:
                mask_img = np.zeros_like(img,dtype=np.uint8)
                
            axes[1].imshow(mask_img)
            axes[1].axis('off')
            axes[1].set_title('Ground Truth Mask')
                
            # Plot the corresponding segmentation prediction mask
            mask_pred = data['pred']
            unique_labes = np.unique(mask_pred).astype(np.uint8)
            #w,h = mask.shape
            mask_pred_img =np.zeros_like(img,dtype=np.uint8)
            for label in unique_labes:
                mask_pred_img[mask_pred == label] = label2color[label]
            
            axes[2].imshow(mask_pred_img)
            axes[2].axis('off')
            axes[2].set_title('Pred Mask')
            

            plt.savefig(full_img_path) # Non-blocking show
            #input("Press Enter to continue to the next image...")
            plt.close() # Close the current figure

    
    