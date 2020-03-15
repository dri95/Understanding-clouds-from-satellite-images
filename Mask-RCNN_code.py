#%%
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import cv2
import seaborn as sns
#%%
#loading and organinzing the data
os.chdir('D:/WORK/DA/understanding_cloud_organization')
path = os.getcwd()
path
traincsv = pd.read_csv("train.csv")
traincsv = traincsv.dropna()
cloud_types = ["Fish","Flower","Gravel","Sugar"]
train_dict = {}
train_labelsid_dict = {}
train_labels_dict = {}
for idx, row in traincsv.iterrows():
    image = row.Image_Label.split("_")[0]
    label = row.Image_Label.split("_")[1]
    label_id = cloud_types.index(label)
    if train_dict.get(image):
        train_dict[image].append(row.EncodedPixels)
        train_labelsid_dict[image].append(label_id)
        train_labels_dict[image].append(label)
    else:
        train_dict[image] = [row.EncodedPixels]
        train_labelsid_dict[image] = [label_id]
        train_labels_dict[image] = [label]
df = pd.DataFrame(columns=["Image","EncodedPixels","Labels","LabelsId", "Height", "Width"])

for key, value in train_dict.items():
    #img = Image.open(path + "/train_images/" + key)
    df = df.append({"Image": key, 
                        "EncodedPixels": value, 
                        "Labels": train_labels_dict[key], 
                        "LabelsId": train_labelsid_dict[key], 
                        "Height" : int(1400), "Width" : int(2100)},
                    ignore_index=True)
df.head()
for i, row in df.head().iterrows():
    print(row.LabelsId)
#%%
    
ROW_SIZE = 512;
COLUMN_SIZE = 512;

class CloudDataset(utils.Dataset):

    def __init__(self, df):
        super().__init__(self)
        
        
        self.add_class("cloud", 1, "Fish")
        self.add_class("cloud", 2, "Flower")
        self.add_class("cloud", 3, "Gravel")
        self.add_class("cloud", 4, "Sugar")
        
        # Add images 
        for i, row in df.iterrows():
            self.add_image("cloud", 
                           image_id=row.name, 
                           path=path + '/train_images/' + str(row.Image), 
                           labels=row.LabelsId,
                           encoded_pixels_array=row.EncodedPixels,
                           height=row.Height, width=row.Width)

    def load_image(self, image_id):
        return resize_image(self.image_info[image_id]['path'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [cloud_types[int(x)] for x in info['labels']]
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
                
        mask = np.zeros((ROW_SIZE, COLUMN_SIZE, len(info['encoded_pixels_array'])), dtype=np.uint8)
        labels = []
        
        for m, (encoded_pixels, label) in enumerate(zip(info['encoded_pixels_array'], info['labels'])):
            temp_mask = rle_decode(encoded_pixels)
            temp_mask = cv2.resize(temp_mask, (COLUMN_SIZE, ROW_SIZE), interpolation=cv2.INTER_NEAREST)
            mask[:, :, m] = temp_mask
            labels.append(int(label)+1)
      
        return mask, np.array(labels)
#%%     
def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

#%% 
def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (COLUMN_SIZE, ROW_SIZE), interpolation=cv2.INTER_AREA)  
    return img

#%% 
class CloudConfig(Config):
    NAME = "clouds"
    NUM_CLASSES = 1 + 4 
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    BACKBONE = 'resnet101'
    IMAGE_MIN_DIM = COLUMN_SIZE
    IMAGE_MAX_DIM = ROW_SIZE
    #TRAIN_ROIS_PER_IMAGE = 200
    IMAGE_RESIZE_MODE = 'none'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    STEPS_PER_EPOCH = 2500
    VALIDATION_STEPS = 400
#%%    
config = CloudConfig()
config.display()
training_percentage = 0.9
#%%
training_set_size = int(training_percentage*len(df))
validation_set_size = int((1-training_percentage)*len(df))

train_dataset = CloudDataset(df[:training_set_size])
train_dataset.prepare()

valid_dataset = CloudDataset(df[training_set_size:training_set_size+validation_set_size])
valid_dataset.prepare()

#%%

#VISUALIZE image with masks ###################################################
for i in range(5):
    image_id = random.choice(train_dataset.image_ids)
    print(train_dataset.image_reference(image_id))
    
    image = train_dataset.load_image(image_id)
    mask, class_ids = train_dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, train_dataset.class_names, limit=5)

    
#%%
    
#TRAIN##########################################################################
    
LR = 1e-4
import warnings 
from imgaug import augmenters as iaa
warnings.filterwarnings("ignore")

augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),iaa.OneOf([ 
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ])
], random_order=True)
   
ROOT_DIR = os.path.abspath("../../DA/understanding_cloud_organization")

CW = 'mask_rcnn_coco.h5'

model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

model.load_weights(CW, by_name=True, exclude= ['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

#%%
######### visualize agumentations

image_id = random.choice(train_dataset.image_ids)
print(train_dataset.image_reference(image_id))
image = train_dataset.load_image(image_id)
cv2.imshow('image',image)

imggrid = augmentation.draw_grid(image[:, :, 0], cols=5, rows=2)
plt.figure(figsize=(30, 12))
_ = plt.imshow(imggrid[:, :, 0],cmap='gray')
del imggrid; del image

#%%
# train heads (top layers)
model.train(train_dataset, valid_dataset, 
            learning_rate=1e-4*2, 
            epochs=1, 
            layers='heads',
            augmentation=None)


history = model.keras_model.history.history

#%%
# train all layers
model.train(train_dataset, valid_dataset, 
            learning_rate=1e-4*2, 
            epochs=15, 
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]
#%%
#all layers
model_path = (r"D:\WORK\DA\understanding_cloud_organization\clouds20191121T0320\mask_rcnn_clouds_0015.h5")
model.load_weights(model_path, by_name=True, exclude= ['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

model.train(train_dataset, valid_dataset, 
            learning_rate=1e-4*2, 
            epochs=25, 
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

model_path = os.path.join(ROOT_DIR, "mask_rcnn_clouds_0025.h5")
model.keras_model.save_weights(model_path)
#%%
#loss plots
epochs = range(1, len(history['loss'])+1)
pd.DataFrame(history, index=epochs)

plt.figure(figsize=(21,11))

plt.subplot(231)
plt.plot(epochs, history["loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(236)
plt.plot(epochs, history["mrcnn_mask_loss"], label="Train Mask loss")
plt.plot(epochs, history["val_mrcnn_mask_loss"], label="Valid Mask loss")
plt.legend()

plt.show()

sns.set(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 12))

ax1.plot(history['loss'], label='Train loss')
ax1.plot(history['val_loss'], label='Validation loss')
ax1.legend(loc='best')
ax1.set_title('Loss')

ax2.plot(history['mrcnn_mask_loss'], label='Train Dice loss')
ax2.plot(history['val_mrcnn_mask_loss'], label='Validation Dice loss')
ax2.legend(loc='best')
ax2.set_title('Dice loss (1 - Dice Coefficient')

plt.xlabel('Epochs')
sns.despine()
plt.show()



#%%
#to find best weights
best_epoch = np.argmin(history["val_loss"])
score = history["val_loss"][best_epoch]
print(f'Best Epoch:{best_epoch+1} val_loss:{score}')

#%% Infernece mode
class InferenceConfig(CloudConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=ROOT_DIR)

#%%
#loading best weights
model_path = (r"D:\WORK\DA\understanding_cloud_organization\clouds20191121T0320\mask_rcnn_clouds_0024.h5")

# Load best trained weights (ours for 24 epochs)
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

#%%
# predict and visualize on a random image
image_id = random.choice(valid_dataset.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(valid_dataset, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            train_dataset.class_names, figsize=(8, 8))


results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            valid_dataset.class_names, r['scores'])



#%% mAP eval on validation
image_ids = valid_dataset.image_ids
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(valid_dataset, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))