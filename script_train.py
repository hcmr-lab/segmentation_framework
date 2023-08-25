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
import shutil

batch_size = 32
epochs = 100

#lr_start = 0.0001
#fusion_type='rgb' #rgb | ndvi | early | late

overal_perfm = "global_perfm.txt"

model_dir = 'weights/'


def main(epochs,model,train_dataloader,test_dataloader,labels,parameters):

    torch.manual_seed(0)
    np.random.seed(0)
    best_score = 0.0
    
    
    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    # Create a handler that writes log messages to a file
    fh = logging.FileHandler(os.path.join(parameters['save_name'],'log.log'))
    # Add the file handler to the logger
    logger.addHandler(fh)
    
    writer = SummaryWriter() # Tensorbaord
    
    
    # Load pre-trained weights
    if os.path.isfile(parameters['best_weigths']):
        try:
            stored_data = torch.load(parameters['best_weigths'])
            model_weigths = stored_data['weigths']
            model.load_state_dict(model_weigths)
            best_score = stored_data['iou'] # update best score with pre-trained 
            logger.info("Loading pre-trained model from ... " + parameters['best_weigths'] )
            logger.info(f"Pre-trained model with the score ... iou: {best_score}")
        except:
            logger.warn("Not able to Load the weights")
            
    model.to(parameters['device'])
  
    # SET AND CONFIG OPTIMIZER AND SCHEDULER
    optimizer = torch.optim.__dict__[parameters['optimizer']](model.parameters(),**parameters['optimizer_args'])
    scheduler = lr_scheduler.__dict__[parameters['scheduler']](optimizer, **parameters['scheduler_args'])
    logger.info(f".....")
    logger.info(f"Optimizer: {parameters['optimizer']} \nLearning Rate: {str(optimizer.param_groups[0]['lr'])}")
    logger.info(f"Scheduler: {parameters['scheduler']}")
    logger.info(f".....")
    
    
    training = base_framework(model,labels,optimizer,device=parameters["device"])
    
    for epoch in range(0, epochs):
        # TRAINING
        loss,results = training.train_epoch(train_dataloader)
        logger.info("| (%d): Training | loss :%f iou %f"%(epoch,round(loss,3),results['iou']))
        scheduler.step()
        
        # SHOW TRAINING SCORES ON TENSORBOARD
        # Plot Loss
        writer.add_scalar('train/loss',loss , epoch)
        # Plot learning rate 
        for i, opt_group in enumerate(optimizer.param_groups):
            writer.add_scalar("train/lr+%s"%(parameters['scheduler']), opt_group['lr'], epoch)
        # Plot performance scores
        for key,values in results.items():
            writer.add_scalar(f'{key}/train',values , epoch)
        
  
        # TESTING
        test_metrics = training.inference_epoch(test_dataloader,save_batch=False)  #TEST DATASET
        test_score = test_metrics['iou'] # Use IoU to track the best model
        logger.info("| (%d) TESTING | iou %f D %fs"%(epoch,test_metrics['iou'],test_metrics['mdurantion']))
        
        # SHOW TEST SCORES ON TENSORBOARD
        for key,values in test_metrics.items():
            writer.add_scalar(f'{key}/test',values , epoch)
            

        # SAVE WEIGTHS AND PRED. MASK 
        # Save weights and prediction mask at target performance scores
        test_cond = np.array([1 for value in parameters['target_scores']  if abs(test_score - value)<0.03  ])
        if test_cond.sum()>0:
            file = parameters['save_name'] + f'_iou_{round(test_score,2)}.pth'
            save_pred_mask_dir_target_score = save_pred_mask_dir + f'_iou_{round(test_score,2)}'
            os.makedirs(save_pred_mask_dir_target_score,exist_ok=True)
            logger.info("Saving weights at ... " + save_pred_mask_dir_target_score)
            # SAVE INDIVIDUAL DATA: RGB;GT and PRED. MASKS
            training.inference_epoch(test_dataloader,save_batch=True,save_pred_mask_dir=save_pred_mask_dir_target_score)
            # SAVE MODEL
            torch.save(model.state_dict(), file)
        
        # Keep track of the best weights;
        if test_score > best_score:
            best_score = test_score
            to_save = {'weigths':model.state_dict(),'iou':test_score}
            torch.save(to_save, parameters['best_weigths'])
            logger.info("Saving weiths at ... " + parameters['best_weigths'])
            logger.info("Saving weights with score %f"%(test_score))
            

    # LOAD BEST MODEL AND SAVE PRED. MASKS
    loaded_stuff = torch.load(parameters['best_weigths'])
    model.load_state_dict(loaded_stuff['weigths'])
    model.to(parameters["device"])
    
    # CREATE DIR TO SAVE PRED.
    loaded_score = round(loaded_stuff['iou'],2)
    save_pred_mask_dir_target_score = save_pred_mask_dir + f'_iou_{loaded_score}'
    os.makedirs(save_pred_mask_dir_target_score,exist_ok=True)
    logger.info("Loading Pre-trained weights with %f"%(loaded_score))
    logger.info("Loading from ... " + save_pred_mask_dir_target_score)
    
    # INDIVIDUAL MASK FOR THE BEST MODEL
    global_metric = training.inference_epoch(test_dataloader, 
                save_batch=True,save_pred_mask_dir=save_pred_mask_dir_target_score)
    
    logger.info("| (%d) FINAL | f1 %f iou %f D %fs"%(epoch,global_metric['f1'],global_metric['iou'],global_metric['mdurantion']))

    # Rename the checkpoint model file to the final model file
    final_model_file = parameters['save_name'] + f'best_model_iou_{round(test_score,2)}.pth'
    #os.cp(parameters['best_weigths'], final_model_file)
    
    shutil.copyfile(parameters['best_weigths'], final_model_file)
    
    

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segnet with pytorch')
    parser.add_argument('--root',  type=str, default="/home/deep/Dropbox/SHARE/ricardoPereira/SUN_Segmentation")
    parser.add_argument('--batch_size',  '-B',  type=int, default=32)
    parser.add_argument('--epochs' ,  '-E',  type=int, default=100)
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()

    dataset_root = args.root 
    print("Getting data from: .... " + dataset_root)
    
    run_base = "BothDataset"
    fusion_types  = ['RGB']
    dataset_names = ['sunrgbd']
    model_names   = [
                     'deeplabv3plus_mobilenet',
                     #'deeplabv3plus_resnet50'
                     #'deeplabv3plus_resnet',
                     #'segnet'
                     ] #segnet or deeplabv3

    for fusion_type in fusion_types:
        for dataset_name in dataset_names:
            for model_name in model_names:
                save_name = os.path.join('weights',f'{run_base}_{model_name}_{dataset_name}_{fusion_type}')

                os.makedirs(save_name, exist_ok=True)
                checkpoint_model_file = os.path.join(save_name, 'tmp.pth')
                checkpoint_optim_file = os.path.join(save_name, 'tmp.optim')
                best_model_file       = os.path.join(save_name, 'best.pth')  
                final_model_file      = os.path.join(save_name, 'final.pth')
                log_file              = os.path.join(save_name, 'log.txt')
                save_pred_mask_dir    = os.path.join(save_name, 'pred_masks')

                
                print('| training %s on GPU #%d with pytorch' % (save_name, args.gpu))
                print('| model will be saved in: %s' % save_name)
                
                parameters = {'root':dataset_root,
                            'dataset':dataset_name,
                            'model_name':model_name,
                            'save_name':save_name,
                            'best_weigths':best_model_file,
                            'target_scores':[],
                            'batch_size':args.batch_size,
                            'optimizer': 'Adam',
                            'optimizer_args':{'lr':0.0001},
                            'scheduler':'StepLR',
                            'scheduler_args':{'step_size':50, 'gamma':0.1},
                            'device':'cuda'
                            }

                
                # LOAD DATASET
                train_dataloader,test_dataloader,labels = ricardo_setup.load_dataset_setup(parameters['root'],
                                parameters['dataset'],batch_size=parameters['batch_size'])
                
                # LOAD MODEL
                model = ricardo_setup.load_model_setup(parameters['model_name'], len(labels))
    
                main(args.epochs,
                     model,
                     train_dataloader,
                     test_dataloader,
                     labels,
                     parameters)