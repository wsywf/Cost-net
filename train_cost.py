import torch
import cv2
import numpy as np
from cost_Net_new_dila import cost_NET
from torch.utils.data import DataLoader
from data_crop import KittiDataset_crop
from data_load_prepared import KittiDataset_prepared

import torch.optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from get_cost_volume import test_cost_NET

batch_size = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_prepared_data = True
if use_prepared_data:
    train_loader =  DataLoader(KittiDataset_prepared('imgname_lists.txt'),batch_size, shuffle=True, pin_memory=False,num_workers=6)
else:
    train_loader =  DataLoader(KittiDataset_crop('imgname_lists.txt'),batch_size, shuffle=True, pin_memory=False,num_workers=6)
writer = SummaryWriter("/home/wsy/workspace/COST_Net_dianji/modle")
load_model_flag = False
model = cost_NET()
if load_model_flag:
    model.load_state_dict(torch.load('/modle/COSTNet_dila.pkl'))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

model.train()
literation_number = 0
for epoch in range(1000):

    exp_lr_scheduler.step()
    train_file_number = 0
    loss_list = []
    for (left_original_patch,right_original_patch,left_downsample_patch,right_downsample_patch,lable) in train_loader:
        left_original_patch = left_original_patch.to(device)
        right_original_patch = right_original_patch.to(device)
        left_downsample_patch = left_downsample_patch.to(device)    
        right_downsample_patch = right_downsample_patch.to(device)       
        lable = lable.to(device)
        score= model.forward(left_original_patch,right_original_patch,left_downsample_patch,right_downsample_patch)
        optimizer.zero_grad()       
        loss = (score-lable)*(score-lable)
        print(loss.size())  
        loss = torch.sum(loss)
        
        loss_list.append(loss.item())

        writer.add_scalar('loss',loss,literation_number)
        train_file_number = train_file_number+1
        literation_number = literation_number+1

        loss.backward()
        optimizer.step()
        if train_file_number>=10 and train_file_number % 20 ==0:
            print("epoch ",epoch," image ",train_file_number,"  loss = ",loss,loss.dtype)

      
        if train_file_number>=10 and train_file_number % 1000 ==0:
             torch.save(model.state_dict(), '/modle/COSTNet_dila.pkl')  


    accuracy_rate = test_cost_NET(train_file_number,epoch)
    writer.add_scalar('accuracy_rate',accuracy_rate,epoch)
    model.train()
    torch.save(model.state_dict(), '/modle/COSTNet_dila.pkl')   
    loss_list = np.array(loss_list)
    loss_mean = np.mean(loss_list)
    print(loss_mean) 
    writer.add_scalar('epoch_mean_loss',loss_mean,epoch)
    print("epoch  ",epoch," done")

    
        
        
        
