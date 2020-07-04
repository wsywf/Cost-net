import torch
from torch.nn import Conv2d, Module, ReLU, MaxPool2d, AvgPool2d ,init, BatchNorm2d
import torch.nn.functional as F
import numpy as np
from torch import nn




class cost_NET(Module):
    def __init__(self):
        super(cost_NET, self).__init__()
        
        
        self.BN_original = nn.BatchNorm2d(16)
        self.IN_original = nn.InstanceNorm2d(16,affine=True)        
        
        self.original_Feature_extract_L1_L2 = torch.nn.Sequential(
        Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 0),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 0))
                
        self.original_Feature_extract_L3_L4 = torch.nn.Sequential(
        Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 0),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 1, padding = 0),
        nn.BatchNorm2d(64))
        
        
        
        self.BN_downsample = nn.BatchNorm2d(16)
        self.IN_downsample = nn.InstanceNorm2d(16,affine=True)
        
        self.downsample_Feature_extract_L1_L2 = torch.nn.Sequential(
        Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 0),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 0))
        
        self.downsample_Feature_extract_L3_L4 = torch.nn.Sequential(
        Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 0),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 1, padding = 0),
        nn.BatchNorm2d(64))

        self.ReLU = nn.ReLU()
        
        self.fc1 = nn.Linear(in_features = 2 , out_features = 1)
        
        
    def Original_Feature_Extract(self, patches):
        feature = self.original_Feature_extract_L1_L2(patches)
        
        
        splited_feature = torch.split(feature, 16, 1)
        out1 = self.IN_original(splited_feature[0].contiguous())
        out2 = self.BN_original(splited_feature[1].contiguous())
        out = torch.cat((out1, out2), 1)
        feature = self.ReLU(out)
        
        feature = self.original_Feature_extract_L3_L4(feature)

        return feature
        
        
        
    def Downsample_Feature_Extract(self, patches):
        feature = self.downsample_Feature_extract_L1_L2(patches)
        
        
        splited_feature = torch.split(feature, 16, 1)
        out1 = self.IN_downsample(splited_feature[0].contiguous())
        out2 = self.BN_downsample(splited_feature[1].contiguous())
        out = torch.cat((out1, out2), 1)
        feature = self.ReLU(out)
        
        feature = self.downsample_Feature_extract_L3_L4(feature)

        return feature

        
    def forward(self, left_original_patch,right_original_patch,left_downsample_patch,right_downsample_patch):
        
        left_downsample_patch = torch.nn.functional.interpolate(left_downsample_patch,(13,13),mode='nearest')
        right_downsample_patch = torch.nn.functional.interpolate(right_downsample_patch,(13,13),mode='nearest')
        
        original_left_feature = self.Original_Feature_Extract (left_original_patch)
        original_right_feature = self.Original_Feature_Extract (right_original_patch)  
        #print( original_left_feature.size())
        #print( original_right_feature.size())
        original_left_feature = torch.squeeze(original_left_feature)
        original_right_feature = torch.squeeze(original_right_feature)
        #print( original_left_feature.size())
        #print( original_right_feature.size())
        s1 = torch.sum(original_left_feature * original_right_feature,1)
        #print(s1.size())
        
        
        
        downsample_left_feature = self.Downsample_Feature_Extract (left_downsample_patch)        
        downsample_right_feature = self.Downsample_Feature_Extract (right_downsample_patch)   
        downsample_left_feature = torch.squeeze(downsample_left_feature)
        downsample_right_feature = torch.squeeze(downsample_right_feature)        
        

        s2 = torch.sum(downsample_left_feature * downsample_right_feature,1)
        
       
        s1 = s1.unsqueeze(1)
        s2 = s2.unsqueeze(1)

        S = torch.cat((s1,s2),1) 
 
        S = self.fc1(S)

        return S
        
        

        
        
        
        
        
    def Feature_Extract(self, left_origin_row_patches , right_origin_row_patches , left_downsample_row_patches , right_downsample_row_patches):
    
        left_downsample_row_patches = torch.nn.functional.interpolate(left_downsample_row_patches,(13,13),mode='nearest')
        right_downsample_row_patches = torch.nn.functional.interpolate(right_downsample_row_patches,(13,13),mode='nearest')
            
        original_left_feature = self.Original_Feature_Extract (left_origin_row_patches)
        original_right_feature = self.Original_Feature_Extract (right_origin_row_patches) 
        downsample_left_feature = self.Downsample_Feature_Extract (left_downsample_row_patches)        
        downsample_right_feature = self.Downsample_Feature_Extract (right_downsample_row_patches)  
                
        original_left_feature = torch.squeeze(original_left_feature)
        original_right_feature = torch.squeeze(original_right_feature)
        downsample_left_feature = torch.squeeze(downsample_left_feature)
        downsample_right_feature = torch.squeeze(downsample_right_feature)
        #print(original_left_feature.size())
    
        return original_left_feature, original_right_feature, downsample_left_feature, downsample_right_feature
        
    def out_fc(self,):
        w = self.fc1.weight.data
        b =self.fc1.bias.data
        weight1 = w[0,0]
        bias = b[0] 
        weight2 = w[0,1]
        
        return weight1,weight2,bias

        
        
        
        
        
        
        
