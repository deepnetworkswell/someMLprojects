
# coding: utf-8

# In[1]:


from __future__ import print_function
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Dense, UpSampling2D,Dropout,BatchNormalization,Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from iou import mean_iou,iou_loss,iou_metric_batch
from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model

from util import *

# In[ ]:


from numpy.random import seed
seed(345)
from tensorflow import set_random_seed
set_random_seed(456)


# In[ ]:


print('loading data...')
data_imgs = np.load('imgs_train.npy')
data_masks = np.load('imgs_mask_train.npy')


data_imgs = data_imgs[:,:,:,np.newaxis]
data_masks = data_masks[:,:,:,np.newaxis]

data_imgs = data_imgs.astype('float32')
data_imgs /=255.0

data_masks = data_masks.astype('float32')
data_masks /=255.0


print(data_imgs.shape,data_masks.shape)

'''
#remove the mean image-wise
for i in (range(data_imgs.shape[0])):
    print(i,np.mean(data_imgs[i]),data_imgs[i].shape)
    data_imgs[i]=data_imgs[i]-np.mean(data_imgs[i])
    print('after op  ',np.mean(data_imgs[i]))
'''

#data_masks=to_categorical(data_masks, num_classes=2)


train_imgs, valid_imgs, train_masks, valid_masks = train_test_split(data_imgs, data_masks, test_size=0.2, random_state=234)


model = load_model('checkpoint/keras.model', custom_objects={'mean_iou': mean_iou, 'dice_coef_loss': dice_coef_loss })


#lets check iou on validation again
preds_valid = model.predict(valid_imgs)



print(valid_masks.shape, preds_valid.shape)


for thresh in range(1, 10, 1):
    thresh=0.1*thresh
    print(thresh)
    preds_valid_rounded = np.round(preds_valid>thresh)
    #mm=iou_metric_batch(valid_masks, preds_valid_rounded)
    #print(mm)
    #mm=iou_metric_batch(valid_masks[:,:,:,1], preds_valid_rounded[:,:,:,1])
    #print(mm)
    #mm=iou_metric_batch(valid_masks[:,:,:,0], preds_valid_rounded[:,:,:,0])
    #print(mm)
    
    mm=iou_metric_batch(valid_masks, preds_valid_rounded[:,:,:,1])
    print(mm+"\n")


nnn=132
image_show(np.squeeze(valid_imgs[nnn]), './out/'+str(nnn)+'img.bmp')
image_show(np.squeeze(valid_masks[nnn]), './out/'+str(nnn)+'mask.bmp')
image_show(np.round(np.squeeze(preds_valid[nnn,:,:,1])>0.5), './out/'+str(nnn)+'pred.bmp')

nnn=332
image_show(np.squeeze(valid_imgs[nnn]), './out/'+str(nnn)+'img.bmp')
image_show(np.squeeze(valid_masks[nnn]), './out/'+str(nnn)+'mask.bmp')
image_show(np.round(np.squeeze(preds_valid[nnn,:,:,1])>0.5), './out/'+str(nnn)+'pred.bmp')


nnn=402
image_show(np.squeeze(valid_imgs[nnn]), './out/'+str(nnn)+'img.bmp')
image_show(np.squeeze(valid_masks[nnn]), './out/'+str(nnn)+'mask.bmp')
image_show(np.round(np.squeeze(preds_valid[nnn,:,:,1])>0.5), './out/'+str(nnn)+'pred.bmp')






'''
original things   categorical_crossentropy
0.9029336734693877
0.7488520408163266
0.8910714285714284
 after 170 epochs




dice_coef_loss
0.8959183673469389
0.743112244897959
0.8802295918367345
after 291 epoch
'''
