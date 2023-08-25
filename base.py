
from utils import metrics as segmetrics
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
import time

import pickle


class base_framework():
    def __init__(self,model, classes, optimizer = None, batch_size = 32, device="cuda"):
        self.classes = classes
        self.device = device
        self.model = model
        self.optimizer = optimizer 
        self.batch_size = batch_size
   
        
    def train_epoch(self,train_loader):
    
        from torch import nn
        criterion = nn.CrossEntropyLoss()

        metrics = segmetrics.multi_class_metrics(self.classes)

        loss_buf = []
        self.model.train()
        import tqdm
        for data in tqdm.tqdm(train_loader,"training"):
            rgb, gt_mask = data

            rgb = rgb.to(self.device)
            gt_mask = gt_mask.to(self.device).long() #[32,1,240,240]

            self.optimizer.zero_grad()
            
            outputs,_ = self.model(rgb)

            loss = criterion(F.softmax(outputs,dim=1), gt_mask)
            
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels = gt_mask.cpu().numpy()

            metrics.compute_multi_class_metrics(true_labels,predictions)   
            
            loss.backward()
            self.optimizer.step()
            loss_buf.append(float(loss))

        results = metrics.get_global_results()
        return np.mean(loss_buf),results



    def inference_epoch(self,test_dataloader,save_batch=False,save_pred_mask_dir=''):

        
        metrics = segmetrics.multi_class_metrics(classes=self.classes)
        
        batch_duration = []
        
        self.model.eval()
        with torch.no_grad():
        
            for id,data in enumerate(tqdm(test_dataloader,"Testing")):
                rgb, mask = data   
   
                rgb = rgb.to(self.device)
                mask = mask # there is no need to send to GPU
                
                t0 = time.time()
                # Run the MODEL
                output,_ = self.model(rgb)
                
                batch_duration.append(time.time() - t0) # save duration
                
                predictions = torch.argmax(output, dim=1).cpu().numpy()
                true_labels = mask.cpu().numpy()
                
                # COMPUTE SEGMENTATION METRICS
                batch_pred = {'iou':None}
                if mask.shape[1] > 0:
                    batch_pred = metrics.compute_multi_class_metrics(true_labels,predictions)
                
                if save_batch == True:
                    # SAVE DATA FOR FUTURE
                    for i, (input,mask,pred) in enumerate(zip(rgb,mask,predictions)):
                        file = os.path.join(save_pred_mask_dir,"%d"%(id*self.batch_size + i) + '.pickle')
                        with open(file, 'wb') as handle:
                            pickle.dump({'iou':batch_pred['iou'],'input':input,'mask':mask,'pred':pred}, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
            
        global_metrics = metrics.get_global_results()
        global_metrics['mdurantion'] = np.mean(batch_duration)
                    
        return global_metrics


        