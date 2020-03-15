import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import keras
import albumentations as A
import segmentation_models as sm
import keras.backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras.losses import binary_crossentropy
import tensorflow as tf

#%%
#loading and organinzing the data

path='D:/WORK/DA/understanding_cloud_organization'
train = pd.read_csv(f'{path}/train.csv')

n_train = len(os.listdir(f'{path}/train_images'))


train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

image_id_list = train['im_id'].unique()

train.head()

#%%
# helper functions to visual, decode and build segmentation mask
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def rle_decode(mask_rle: str = '', shape: tuple =(1400, 2100)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def bulid_mask(train_df,image_id,image_shape):
    masks=np.zeros((*image_shape,4))
    for i, (idx, row) in enumerate(train.loc[train['im_id'] == image_id].iterrows()):
        
        mask_rle = row['EncodedPixels']
        
        try: # label might not be there!
            mask = rle_decode(mask_rle)
        except:
            mask = np.zeros(image_shape)
        masks[:,:,i]=mask
   
    return masks 

def show_image_mask(image_id):
    image = Image.open(f"{path}/train_images/{image_id}")
    print("actual mask")
    mask=bulid_mask(train,image_id ,(1400, 2100))
    
    visualize(
       image=image, 
       Fish_mask=mask[..., 0].squeeze(),
       Flower_mask=mask[..., 1].squeeze(),
       Gravel_mask=mask[..., 2].squeeze(),
       Suger_mask=mask[..., 3].squeeze(),    
    )

#%%
#VISUALIZE image with masks ###################################################
fig = plt.figure(figsize=(25, 16))
for j, im_id in enumerate(np.random.choice(train['im_id'].unique(), 4)):
    for i, (idx, row) in enumerate(train.loc[train['im_id'] == im_id].iterrows()):
        ax = fig.add_subplot(5, 4, j * 4 + i + 1, xticks=[], yticks=[])
        im = Image.open(f"{path}/train_images/{row['Image_Label'].split('_')[0]}")
        plt.imshow(im)
        mask_rle = row['EncodedPixels']
        try: 
            mask = rle_decode(mask_rle)
        except:
            mask = np.zeros((1400, 2100))
        plt.imshow(mask, alpha=0.5, cmap='gray')
        ax.set_title(f"Image: {row['Image_Label'].split('_')[0]}. Label: {row['label']}")


#%%
#VISUALIZE image with mask cropped ###################################################
np.random.seed(0)
for i in np.random.randint(0,len(image_id_list),size=5):
    image_id=image_id_list[i]
    show_image_mask(image_id)
    

#%%
#Dataset class which  Read images, apply augmentation and preprocessing transformations.
   
class Dataset:
    
    CLASSES = ['Fish', 'Flower','Gravel','Suger']
    
    def __init__(
            self,
            tain_df,
            images_dir, 
            image_id_list, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = image_id_list
        self.images_fps = [f"{images_dir}/train_images/{image_id}" for image_id in self.ids]

        self.train_df= tain_df 
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        # apply augmentations
        mask=bulid_mask(self.train_df,self.ids[i] ,(1400, 2100))
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    
    
#%%
# keras image datagenerator class
class Dataloder(keras.utils.Sequence):

    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)



def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    #Add paddings to make image shape divisible by 32
    test_transform = [
        A.Resize(320, 320, interpolation=1, always_apply=False, p=1)
        
    ]
    
    
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)    

   
#%% 
# modified adam for adaptive optimization
class AdamAccumulate(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lrr = K.variable(lr,name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lrr

        completed_updates = K.cast(tf.math.floordiv(self.iterations, self.accum_iters), K.floatx())

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * completed_updates))

        t = completed_updates + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))
 
        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lrr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#%% 
# loss functions

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def mAP(y_true,y_pred):
	num_classes = 4
	average_precisions = []
	relevant = K.sum(K.round(K.clip(y_true, 0, 1)))
	tp_whole = K.round(K.clip(y_true * y_pred, 0, 1))
	for index in range(num_classes):
		temp = K.sum(tp_whole[:,:index+1],axis=1)
		average_precisions.append(temp * (1/(index + 1)))
	AP = sum(average_precisions) / relevant
	mAP = K.mean(AP,axis=0)
	return mAP

#%% 
#Unet model and complilation   
preprocess_input = sm.get_preprocessing('efficientnetb3')    
EPOCHS = 15
BATCH_SIZE = 8
CLASSES = ['Fish', 'Flower','Gravel','Suger']
n_classes = len(CLASSES)
opt = AdamAccumulate(lr=0.002, accum_iters=8)
model = sm.Unet(
    'efficientnetb3',#resnet101,inceptionresnetv2,efficientnetb7
    encoder_weights='imagenet',
    classes=4,
    input_shape=(320, 320, 3),
    activation='sigmoid'
)
model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[dice_coef,mAP])
   
#%% 
#train/validation splits  
from sklearn.model_selection import train_test_split
train_id_list, val_id_list = train_test_split(
    image_id_list, random_state=2019, test_size=0.1
)
# Dataset for training images
train_dataset = Dataset(
    train,
    path, 
    train_id_list, 
    classes=['Fish', 'Flower','Gravel','Suger'],
    augmentation=get_validation_augmentation(),#get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    train,
    path, 
    val_id_list, 
    classes=['Fish', 'Flower','Gravel','Suger'], 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

#%%
train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)   

#%%  
# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
]
# train model
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)   
   
#%%
#loss plots   
 
history_df = pd.DataFrame(history.history)
history_df.to_csv('history.csv', index=False)

history_df[['loss', 'val_loss']].plot()
history_df[['dice_coef', 'val_dice_coef']].plot()
history_df[['lr']].plot()  
   
#%%
#initilizing the model with best weights and defining a function for predictions
   
model.load_weights(r"C:\Users\cool_\best_model.h5")   



def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
def predict_mask(image_file,threshold):

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    width = 320
    height = 320
    dim = (width, height)
    if image.shape[0]!=width and image.shape[0]!=width:
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    preprocss=get_preprocessing(preprocess_input)
    image=preprocss(image=image)
    image = np.expand_dims(image['image'], axis=0)
    p_mask = model.predict(image)
    p_mask=(p_mask > threshold).astype(np.float_)
    image = cv2.resize(image.squeeze(axis=0),(2100,1400), interpolation = cv2.INTER_AREA)
    p_mask =cv2.resize(p_mask.squeeze(axis=0), (2100,1400), interpolation = cv2.INTER_AREA)
    
    visualize(
        image=denormalize(image),
        Fish_mask=p_mask[..., 0],
        Flower_mask=p_mask[..., 1],
        Gravel_mask=p_mask[..., 2],
        Suger_mask=p_mask[..., 3],  
    )
    
    return image,p_mask



#%% 
# predict and visualize on random images

np.random.seed(8)
for i in np.random.randint(0,len(image_id_list),size=5):
    image_id=image_id_list[i]
    show_image_mask(image_id)
    print("predicted mask")
    image,p_mask=predict_mask(f"{path}/train_images/{image_id}",0.5)   


#%% 
# mAP eval on validation set  

scores = model.evaluate_generator(valid_dataloader)

print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(model.metrics, scores[1:]):
    print("meanAP {}: {:.5}".format(metric.__name__, value))   

   
   
   
   
   
   
   
   
   
   
   

