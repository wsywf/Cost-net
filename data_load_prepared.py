import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

class KittiDataset_prepared(Dataset):
    def __init__(self, train_file, mode='train', n_samples=None):
        self.mode = mode
        self.train_file = open(train_file,'r')
        self.image_lists = []
        for line in self.train_file:
            line = line.strip('\n')
            self.image_lists.append(line)
        self.data_root = '/data_prepare'

    def __len__(self):
        return len(self.image_lists)

    def __getitem__(self, i):
        image_name = self.image_lists[i]
#        print(image_name)
#        image_name = '000002_10'
        
        left_original_patch_path = self.data_root + '/left_origin/' + image_name + 'left_ori.png'
        left_original_patch = cv2.imread(left_original_patch_path,0)
        
        right_original_patch_path = self.data_root + '/right_origin/' + image_name + 'right_origin.png'
        right_original_patch = cv2.imread(right_original_patch_path,0)

        left_downsample_patch_path = self.data_root + '/left_downsample/' + image_name + 'left_downsample.png'
        left_downsample_patch = cv2.imread(left_downsample_patch_path,0)

        right_downsample_patch_path = self.data_root + '/right_downsample/' + image_name + 'right_downsample.png'
        right_downsample_patch = cv2.imread(right_downsample_patch_path,0)  
        
        if left_original_patch is None or right_original_patch is None or left_downsample_patch is None or right_downsample_patch is None:
            left_original_patch = np.zeros((13,13))
            right_original_patch = np.zeros((13,13))
            left_downsample_patch = np.zeros((27,27))
            right_downsample_patch = np.zeros((27,27))

            lable = []
            lable.append(1)
            
        
        else:
            lable_idex = int(image_name)
            lable_path = '/home/wsy/work/NEW_WORK/data_load_4-80/lable/lable.npy'
            lable_volume = np.load(lable_path)
            lable = []
            lable.append(lable_volume[lable_idex])


        
        
        
        left_original_patch = left_original_patch.astype(np.float32)
        right_original_patch = right_original_patch.astype(np.float32)
        left_downsample_patch = left_downsample_patch.astype(np.float32)
        right_downsample_patch = right_downsample_patch.astype(np.float32)
        

##图像块归一化            
        left_original_patch = left_original_patch.astype(np.float32)
        right_original_patch = right_original_patch.astype(np.float32)
        left_downsample_patch = left_downsample_patch.astype(np.float32)
        right_downsample_patch = right_downsample_patch.astype(np.float32)


##图像块类型转换       
        left_original_patch = np.array(left_original_patch)
        left_original_patch = torch.tensor(left_original_patch)
#        print(left_original_patch.size())
        left_original_patch = left_original_patch.unsqueeze(0)  ## [n,1,5,5]
        left_original_patch = left_original_patch.type(torch.FloatTensor) 
#        print(left_original_patch.size())        
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
#        print(lable.size())


        return left_original_patch,right_original_patch,left_downsample_patch,right_downsample_patch,lable
        

