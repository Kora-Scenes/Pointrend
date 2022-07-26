# Ensemble of Mask RCNN and Pointrend
# Predicted Masks of 10 images KITTI semantic stored in .npy files

import warnings as wr
import glob
import os
import cv2
import numpy as np

wr.filterwarnings("ignore")

# Save predicted masks (im_predmasks) in mrcnn/ptrend.npy files
# with open('ptrend_masks.npy', 'wb') as f:
#     np.save(f,im_predmasks)
    
# with open('mrcnn_masks.npy', 'wb') as f:
#     np.save(f,im_predmasks)
    
with open('mrcnn_masks.npy','rb') as f: 
    mrcnn_masks = np.load(f)   

with open('ptrend_masks.npy','rb') as f:
    ptrend_masks = np.load(f)

path = '/home/yln1kor/Downloads/kitti_official_semantic/training'
path_images = path + '/image_2'
path_instance = path + '/instance'
path_semantic = path + '/semantic_rgb'

gt_masks = []
c = 1
for imageName in sorted(glob.glob(os.path.join(path_semantic, '*.png'))):
    im = cv2.imread(imageName)
    mask = (im == [142,0,0]).all(-1)
    gt_masks.append(mask)
    c += 1
    if c == 10:
        break

im_predmasks = []
for i in range(len(gt_masks)):
    pred_mask = np.full(gt_masks[0].shape,False, dtype =bool)
    pred_mask = np.logical_or(pred_mask,ptrend_masks[i])
    pred_mask = np.logical_and(pred_mask,mrcnn_masks[i])
    im_predmasks.append(pred_mask)
    cv2.imshow('ensemble_masks',pred_mask.astype(np.uint8)*255)
    cv2.waitKey(0)

sum_IOU = 0
sum_DSC = 0
for i in range(len(gt_masks)):
    gt = gt_masks[i]
    pred = im_predmasks[i]
    intersection = np.logical_and(gt,pred)
    union = np.logical_or(gt,pred)
    IOU = np.sum(intersection) / np.sum(union)
    sum_IOU += IOU
    Dice_coeff = 2 * np.sum(intersection) / (np.sum(gt) + np.sum(pred))
    sum_DSC += Dice_coeff
    #print('IOU:',IOU, 'Dice Coeff:',Dice_coeff)  '''Individual Images'''

print('IOU',sum_IOU/len(gt_masks))
print('DSC',sum_DSC/len(gt_masks))

def ensemble():
    pass
