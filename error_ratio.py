import cv2
import random
import numpy as np
import torch


def accurate_ratio(disp_gray,gt_image):
    img_h,img_w = disp_gray.shape
    valid_num = 0
    right_num_0 = 0
    right_num_1 = 0
    right_num_2 = 0
    right_num_3 = 0
    for y in range(0,img_h):
        for x in range(0,img_w):
            disp = disp_gray[y,x]
            gt = gt_image[y,x]
            if gt>0:
                valid_num = valid_num+1
                if abs(gt-disp)<=3:
                    right_num_3 = right_num_3+1                
                if abs(gt-disp)<=2:
                    right_num_2 = right_num_2+1
                if abs(gt-disp)<=1:   
                    right_num_1 = right_num_1+1
                if abs(gt-disp)==0:   
                    right_num_0 = right_num_0+1      

    right_ratio_0 = right_num_0/valid_num
    right_ratio_1 = right_num_1/valid_num
    right_ratio_2 = right_num_2/valid_num
    right_ratio_3 = right_num_3/valid_num
    print('0_piexl',right_ratio_0)
    print('1_piexl',right_ratio_1)
    print('2_piexl',right_ratio_2)
    print('3_piexl',right_ratio_3)
    acc_rate = np.zeros(4)
    acc_rate[0] = right_ratio_0
    acc_rate[1] = right_ratio_1
    acc_rate[2] = right_ratio_2
    acc_rate[3] = right_ratio_3
                
    return right_ratio_3

    
#for image_num in range (20,50):
#    print('img',image_num)
#    
#    left_path = '/home/wsy/work/NEW_WORK/embedding_stereo/test_file/' + 'img_'+ str(image_num) + 'disp_gray' + str(9) + 'train_file_number' + str(3) + '.png'
#    gray_image = cv2.imread(left_path,0)
#    
#    
#    disp_path = '/home/wsy/datasets/training' + '/disp_noc_0/' + '0000' + str(image_num) + '_10.png'
#    gt_image = cv2.imread(disp_path,0)
#    
#    
#    accurate_ratio(gray_image,gt_image)
