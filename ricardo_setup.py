import torch
import network

from dataloaders import sunrgbd_labels



def inference_load_dataset_setup(root,dataset,set_name,batch_size=2):
    dataset = dataset.lower()
    from dataloaders import INDOORDATASETNP
    
    assert dataset in ['sunrgbd','mit67'],'Dataset not recognized'
    from torch.utils import data
        
    dataset = INDOORDATASETNP.__dict__[dataset](root,set_name)
    #test_dataset  = INDOORDATASETNP.__dict__[dataset](root,"test")
    
    labels = sunrgbd_labels.labelid
    #train_loader = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    
    train_dataloader = data.DataLoader(dataset,batch_size,shuffle=False)
   
    
    return train_dataloader,[],labels
 

def train_load_dataset_setup(root,dataset,batch_size=2):
    dataset = dataset.lower()
    from dataloaders import INDOORDATASETNP
    
    assert dataset in ['sunrgbd','mit67'],'Dataset not recognized'
    from torch.utils import data
        
    train_dataset = INDOORDATASETNP.__dict__[dataset](root,"train")
    test_dataset  = INDOORDATASETNP.__dict__[dataset](root,"test")
    
    labels = sunrgbd_labels.labelid
    train_loader = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    
    train_dataloader = data.DataLoader(train_loader,batch_size,shuffle=True)
    test_dataloader = data.DataLoader(train_loader,batch_size,shuffle=False)
    
    return train_dataloader,test_dataloader,labels


def load_model_setup(name,num_classes):
   
    model = None  
    # Selection of the model
    if 'deeplab' in name:
        model = network.modeling.__dict__[name](in_channels=3,num_classes=num_classes, output_stride=8)

    return model