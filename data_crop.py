import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

class KittiDataset_crop(Dataset):
    def __init__(self, train_file, mode='train', n_samples=None):
        self.mode = mode
        self.train_file = open(train_file,'r')
        self.image_lists = []
        for line in self.train_file:
            line = line.strip('\n')
            self.image_lists.append(line)
        self.data_root = '/home/wsy/datasets/training'

    def __len__(self):
        return len(self.image_lists)

    def __getitem__(self, i):
        image_name = self.image_lists[i]
#        image_name = '000002_10'
        
        left_path = self.data_root + '/image_2/' + image_name + '.png'
        left_image = cv2.imread(left_path,0)
        
        right_path = self.data_root + '/image_3/' + image_name + '.png'
        right_image = cv2.imread(right_path,0)
        
        disp_path = self.data_root + '/disp_occ_0/' + image_name + '.png'
        disp_image = cv2.imread(disp_path,0)
        

        
        max_pixel_l = np.max(left_image)
        max_pixel_r = np.max(right_image)
        disp_val,x,y =0,0,0 
        img_h , img_w = left_image.shape

        left_image = left_image.astype(np.float32)
        right_image = right_image.astype(np.float32)
##随机取有真实视差的点        
        while(disp_val==0 or disp_val>80):
            x = random.randint(200,img_w-145)
            y = random.randint(20, img_h-20)
            disp_val = disp_image[y,x]

##随机选取样本label类型
#        sample_mode  = 1
        sample_mode  = random.randint(0,1)
##样本为正样本        
        if sample_mode == 1:
            pos =[n for n in range(0,1)] + [n for n in range(0)]
            opos = 0
            xpos = x-disp_val +opos
            x0 = xpos-6
            x1 = xpos+7
            y0 = y-6
            y1 = y+7
            left_original_patch = left_image[y0:y1, x-6:x+7]
            right_original_patch = right_image[y0:y1, x0:x1]
            x0 = xpos-13
            x1 = xpos+14
            y0 = y-13
            y1 = y+14
            left_downsample_patch = left_image[y0:y1, x-13:x+14]
            right_downsample_patch = right_image[y0:y1, x0:x1]

##样本为负样本           
        if sample_mode == 0:
            neg =[n for n in range(1,70)] + [n for n in range(-70,0)]
            oneg = random.sample(neg,1)[0]
            xneg = x-disp_val + oneg

            x0 = xneg-6
            x1 = xneg+7
            y0 = y-6
            y1 = y+7
            left_original_patch = left_image[y0:y1, x-6:x+7]                    
            right_original_patch = right_image[y0:y1, x0:x1]
            x0 = xneg-13
            x1 = xneg+14
            y0 = y-13
            y1 = y+14
            left_downsample_patch = left_image[y0:y1, x-13:x+14]
            right_downsample_patch = right_image[y0:y1, x0:x1]

        lable = []
        if left_original_patch is None or right_original_patch is None or left_downsample_patch is None or right_downsample_patch is None:
            left_original_patch = np.zeros((13,13))
            right_original_patch = np.zeros((13,13))
            left_downsample_patch = np.zeros((27,27))
            right_downsample_patch = np.zeros((27,27))
            lable.append(1)   
        else:
            lable.append(sample_mode)        
#        print("left_original_patch",left_original_patch)
#        print("left_downsample_patch",left_downsample_patch)
##图像块归一化            
        left_original_patch = (left_original_patch - np.mean(left_original_patch))/max_pixel_l 
        right_original_patch = (right_original_patch - np.mean(right_original_patch))/max_pixel_r 
        left_downsample_patch = (left_downsample_patch - np.mean(left_downsample_patch))/max_pixel_l 
        right_downsample_patch = (right_downsample_patch - np.mean(right_downsample_patch))/max_pixel_r 

##图像块类型转换       
        left_original_patch = np.array(left_original_patch)
        left_original_patch = torch.tensor(left_original_patch)
        left_original_patch = left_original_patch.unsqueeze(0)  ## [n,1,5,5]
        left_original_patch = left_original_patch.type(torch.FloatTensor) 
        
        right_original_patch = np.array(right_original_patch)
        right_original_patch = torch.tensor(right_original_patch)
        right_original_patch = right_original_patch.unsqueeze(0)  ## [n,1,5,5]
        right_original_patch = right_original_patch.type(torch.FloatTensor) 
        
        left_downsample_patch = np.array(left_downsample_patch)
        left_downsample_patch = torch.tensor(left_downsample_patch)
        left_downsample_patch = left_downsample_patch.unsqueeze(0)  ## [n,1,5,5]
        left_downsample_patch = left_downsample_patch.type(torch.FloatTensor) 
        
        right_downsample_patch = np.array(right_downsample_patch)
        right_downsample_patch = torch.tensor(right_downsample_patch)
        right_downsample_patch = right_downsample_patch.unsqueeze(0)  ## [n,1,5,5]
        right_downsample_patch = right_downsample_patch.type(torch.FloatTensor) 
        
        
        lable = np.array(lable)
        lable = lable.astype(np.int64)
        lable = torch.tensor(lable)
        lable = lable.type(torch.FloatTensor)

        return left_original_patch,right_original_patch,left_downsample_patch,right_downsample_patch,lable
        

