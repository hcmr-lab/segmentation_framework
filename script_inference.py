import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import time
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import network
import pickle
from utils import metrics as segmetrics
from base import base_framework
import logging
import ricardo_setup

batch_size = 32

#lr_start = 0.0001
#fusion_type='rgb' #rgb | ndvi | early | late

overal_perfm = "global_perfm.txt"

model_dir = 'weights/'


def main(model,test_dataloader,labels,parameters):

    torch.manual_seed(0)
    np.random.seed(0)
    best_score = 0.0
    
    # Create a logger
    log_file  = os.path.join(parameters['save_name'],'log.log')
    
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
    
    #logging.basicConfig(level = logging.INFO, filename = log_file)
        
    # Load pre-trained weights
    if os.path.isfile(parameters['best_weigths']):
        try:
            stored_data = torch.load(parameters['best_weigths'])
            model_weigths = stored_data['weigths']
            model.load_state_dict(model_weigths)
            best_score = stored_data['iou'] # update best score with pre-trained 
            logging.info("Loading pre-trained model from ... " + parameters['best_weigths'] )
            logging.info(f"Pre-trained model with the score ... iou: {best_score}")
        except:
            logging.warn("Not able to Load the weights")
            
    model.to(parameters['device'])
      
    training = base_framework(model,labels,batch_size=parameters["batch_size"],device=parameters["device"])  
    
    # CREATE DIR TO SAVE PRED.
    loaded_score = round(best_score,2)
    save_pred_mask_dir_target_score = os.path.join(parameters['save_name'],f'pred_masks_iou_{loaded_score}')
    
    os.makedirs(save_pred_mask_dir_target_score,exist_ok=True)
    logging.info("Loading Pre-trained weights with %f"%(loaded_score))
    logging.info("Saving to ... " + save_pred_mask_dir_target_score)
    
    # INDIVIDUAL MASK FOR THE BEST MODEL
    global_metric = training.inference_epoch(test_dataloader, 
                save_batch=True,save_pred_mask_dir=save_pred_mask_dir_target_score)
    
    logging.info("| FINAL | f1 %f iou %f D %fs"%(global_metric['f1'],global_metric['iou'],global_metric['mdurantion']))


    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segnet with pytorch')
    parser.add_argument('--root',  type=str, default="/home/deep/Dropbox/SHARE/ricardoPereira/")
    parser.add_argument('--batch_size',  '-B',  type=int, default=32)
    parser.add_argument('--weights',  '-EF', type=str, default="weights/BothDataset_deeplabv3plus_mobilenet_sunrgbd_RGB/iou_0.77.pth")


    args = parser.parse_args()

    dataset_root = args.root 
    print("Getting data from: .... " + dataset_root)
    
    run_base = "BothDataset"
    fusion_types  = ['RGB']
    dataset_names = [
                    #'sunrgbd',
                    'mit67'
                     ]
    
    model_names   = [
                     'deeplabv3plus_mobilenet',
                     #'deeplabv3plus_resnet50'
                     #'deeplabv3plus_resnet',
                     #'segnet'
                     ] #segnet or deeplabv3

    for set_type in ['train','test']:
        for dataset_name in dataset_names:
            for model_name in model_names:
                save_name = os.path.join('weights',f'{run_base}_{model_name}_{dataset_name}_{set_type}')
                weights_file = args.weights
                
                assert os.path.isfile(weights_file) 
                os.makedirs(save_name, exist_ok=True)
      
                
                parameters = {'root':dataset_root,
                            'dataset':dataset_name,
                            'model_name':model_name,
                            'save_name':save_name,
                            'best_weigths':weights_file,
                            'target_scores':[0.5],
                            'batch_size':args.batch_size,
                            'optimizer': 'Adam',
                            'optimizer_args':{'lr':0.0001},
                            'scheduler':'StepLR',
                            'scheduler_args':{'step_size':50, 'gamma':0.1},
                            'device':'cuda'
                            }

                
                # LOAD DATASET
                dataloader,_,labels = ricardo_setup.inference_load_dataset_setup(parameters['root'],
                                parameters['dataset'],set_name=set_type,batch_size=parameters['batch_size'])
                
                # LOAD MODEL
                model = ricardo_setup.load_model_setup(parameters['model_name'],len(labels))
    
                main(model,
                     dataloader,
                     labels,
                     parameters)