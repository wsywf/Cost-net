import torch
import cv2
import numpy as np
from cost_Net_new_dila import cost_NET
from torch.utils.data import DataLoader
from data_crop_batchsize import KittiDataset
import torch.optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from error_ratio import accurate_ratio
import time


def test_cost_NET(train_file_number,epoch_num):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cost_NET()
    model.load_state_dict(torch.load('/modle/COSTNet_dila.pkl'))
    model = model.to(device)
    model.eval()

    save_root = '/test_file/'

    accuracy_all = []
    for image_num in range (20,21):
        data_root = 'datasets/training'
        left_path = data_root + '/image_2/' + '0000'+str(image_num) + '_10.png'
        left_image = cv2.imread(left_path,0)

        
        right_path = data_root + '/image_3/' + '0000' + str(image_num) + '_10.png'
        right_image = cv2.imread(right_path,0)

        
        disp_path = data_root + '/disp_occ_0/' + '0000' + str(image_num) + '_10.png'
        grand_truth_image = cv2.imread(disp_path,0)

        
        
        left_image = np.pad(left_image,((13,13),(13,13)),'constant')
        right_image = np.pad(right_image,((13,13),(13,13)),'constant')
        
        

        img_h , img_w = left_image.shape
        img_h1 = img_h-26
        img_w1 = img_w-26
        
        left_originla_feature = torch.zeros(img_h1,img_w1,64)
        right_originla_feature = torch.zeros(img_h1,img_w1,64)
        left_downsample_feature = torch.zeros(img_h1,img_w1,64)
        right_downsample_feature = torch.zeros(img_h1,img_w1,64)
        
        
        net_start = time.clock()
        for y in range (13,img_h1+13):
            left_origin_row_patches = []
            right_origin_row_patches = []
            left_downsample_row_patches = []
            right_downsample_row_patches = []
            for x in range (13,img_w1+13):
                    x0 = x-6
                    x1 = x+7
                    y0 = y-6
                    y1 = y+7
                    left_original_patch = left_image[y0:y1, x-6:x+7]
                    right_original_patch = right_image[y0:y1, x0:x1]
                    x0 = x-13
                    x1 = x+14
                    y0 = y-13
                    y1 = y+14
                    left_downsample_patch = left_image[y0:y1, x-13:x+14]
                    right_downsample_patch = right_image[y0:y1, x0:x1]

                    left_original_patch = left_original_patch.astype(np.float32)
                    right_original_patch = right_original_patch.astype(np.float32)
                    left_downsample_patch = left_downsample_patch.astype(np.float32)
                    right_downsample_patch = right_downsample_patch.astype(np.float32)
                    
                    left_origin_row_patches.append(left_original_patch)
                    right_origin_row_patches.append(right_original_patch)
                    left_downsample_row_patches.append(left_downsample_patch)
                    right_downsample_row_patches.append(right_downsample_patch)
                        
            left_origin_row_patches = np.array(left_origin_row_patches)
            left_origin_row_patches = torch.tensor(left_origin_row_patches)
            left_origin_row_patches = left_origin_row_patches.type(torch.FloatTensor)
            right_origin_row_patches = np.array(right_origin_row_patches)
            right_origin_row_patches = torch.tensor(right_origin_row_patches)
            right_origin_row_patches = right_origin_row_patches.type(torch.FloatTensor)  
            right_downsample_row_patches = np.array(right_downsample_row_patches)
            right_downsample_row_patches = torch.tensor(right_downsample_row_patches)
            right_downsample_row_patches = right_downsample_row_patches.type(torch.FloatTensor) 
            left_downsample_row_patches = np.array(left_downsample_row_patches)
            left_downsample_row_patches = torch.tensor(left_downsample_row_patches)
            left_downsample_row_patches = left_downsample_row_patches.type(torch.FloatTensor)   
            
            left_origin_row_patches = left_origin_row_patches.unsqueeze(1)
            right_origin_row_patches = right_origin_row_patches.unsqueeze(1)
            right_downsample_row_patches = right_downsample_row_patches.unsqueeze(1)
            left_downsample_row_patches = left_downsample_row_patches.unsqueeze(1)
#            print(left_origin_row_patches.size())
            left_origin_row_patches = left_origin_row_patches.to(device)
            right_origin_row_patches = right_origin_row_patches.to(device)
            left_downsample_row_patches = left_downsample_row_patches.to(device)
            right_downsample_row_patches = right_downsample_row_patches.to(device)
                    
            original_left_feature , original_right_feature , downsample_left_feature , downsample_right_feature = model.Feature_Extract       (left_origin_row_patches,right_origin_row_patches,left_downsample_row_patches,right_downsample_row_patches)

            
            original_left_feature = original_left_feature.detach()
            original_right_feature = original_right_feature.detach()
            downsample_left_feature = downsample_left_feature.detach()
            downsample_right_feature = downsample_right_feature.detach()
            
            
                
            left_originla_feature[y-13,:,:] =  original_left_feature
            right_originla_feature[y-13,:,:] =  original_right_feature
            left_downsample_feature[y-13,:,:] =  downsample_left_feature
            right_downsample_feature[y-13,:,:] =  downsample_right_feature
            
            
            
        net_end = time.clock()
        print('img_' + str() + 'net_time      ' ,net_end - net_start)
        original_cost_volume = torch.zeros(((128,img_h1,img_w1)))
        downsample_cost_volume = torch.zeros(((128,img_h1,img_w1)))
            
        for x in range(0,img_w1):

            for d in range (0,128):
                
                if x - d >=0:
                    original_cost_volume[d,:,x] = torch.sum(left_originla_feature[:,x,:] * right_originla_feature[:,x-d,:],1)
                    downsample_cost_volume[d,:,x] = torch.sum(left_downsample_feature[:,x,:] * right_downsample_feature[:,x-d,:],1)
                else:
                    original_cost_volume[d,:,x] = 0
                    downsample_cost_volume[d,:,x] = 0
        start2 = time.clock()
        print('img_' + str() + 'cost_volume time      ' ,start2 - net_end)

        
        weight1,weight2,bias = model.out_fc()
        
        weight1 = weight1.detach()
        weight2 = weight2.detach()
        bias = bias.detach()
        print(weight1)
        print(weight2)
        print(bias)
        cost_volume = original_cost_volume * weight1.item() + downsample_cost_volume * weight2.item() + bias.item()
        
        cost_volume = cost_volume.numpy()
        disp_image = np.argmax(cost_volume,0)
        
        
        accuracy = accurate_ratio(disp_image,grand_truth_image)
        accuracy_all.append(accuracy)
        
        print(disp_image.shape)
        cost_volume_path =save_root + 'img_'+ str(image_num) + 'cost_volume_epoch' + str(epoch_num) + 'train_file_number' + str(train_file_number) + '.npy'
        disp_gray_path =save_root +'img_'+ str(image_num) + 'disp_gray' + str(epoch_num) + 'train_file_number' + str(train_file_number) + '.png'
        np.save(cost_volume_path,cost_volume)
        cv2.imwrite(disp_gray_path,disp_image)
        print('done')
        

    acc=0
    for i in range(len(accuracy_all)):
            acc += accuracy_all[i]
    accuracy_rate =  acc/len(accuracy_all)       
    return accuracy_rate

    
    
    
##运行    
if __name__ == '__main__':
    accuracy = test_cost_NET(0,0)
    
    
                    
        
        
    
                         
            
