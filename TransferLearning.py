import tensorflow
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from tensorflow.keras.optimizers import Adam
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from livelossplot.inputs.keras import PlotLossesCallback
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from tensorflow.keras import layers, models
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, recall_score, accuracy_score, precision_score, f1_score

from tqdm.notebook import tqdm

#%%
""" Gpu memory allocation control """
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
#%%   
import numpy as np

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """ 
    Example code snippet 

        batch_size = 16
        train_gen = DataGenerator(traindata, traindata_label, batch_size)
        test_gen = DataGenerator(x_test, y_test, batch_size)

    """
    def __init__(self, x_set:np.ndarray, y_set:np.ndarray, batch_size:int):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
    
#%%
#D:\Spyder Verisetleri
images_path = r"D:\Spyder Verisetleri\fruits_25\Train" 
fruits= os.listdir(images_path) 

data=[]

for fruit in fruits: 
    for i, image_file in tqdm(enumerate(os.listdir(os.path.join(images_path, fruit)))):
        img = cv2.imread(os.path.join(images_path, fruit, image_file))
        image = img / 255.0
        data.append([image, fruits.index(fruit)])
        


#%%
x = []
y = []

for images, label in data:
    x.append(images)
    y.append(label)

# Converting to Numpy Array    
x = np.array(x)
y = np.array(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 1)

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)



#%%
plot_loss_1 = PlotLossesCallback()

tl_checkpoint_1 = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)

early_stop = EarlyStopping(monitor='val_loss',
                           patience=5,
                           restore_best_weights=True,
                           mode='min')   

#%%
batch_size = 16
train_gen = DataGenerator(x_train, y_train, batch_size)
val_gen = DataGenerator(x_val, y_val, batch_size)

#%%
# VGG16 is a pre-trained CNN model. 
conv_base = tensorflow.keras.applications.VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(100, 100, 3)
                  )

# Showing the convolutional layers.
conv_base.summary()

# Deciding which layers are trained and frozen.
# Until 'block5_conv1' are frozen.
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# An empyty model is created.
model = tensorflow.keras.models.Sequential()

# VGG16 is added as convolutional layer.
model.add(conv_base)

# Layers are converted from matrices to a vector.
model.add(tensorflow.keras.layers.Flatten())

# Our neural layer is added.
model.add(tensorflow.keras.layers.Dense(256, activation='relu'))
model.add(tensorflow.keras.layers.Dense(25, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.00001), 
           loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = False), 
           metrics = ['accuracy'])

# Showing the created model.
model.summary()

#%%
%%time

plot_loss_2 = PlotLossesCallback()

conv_history = model.fit(train_gen, 
                        epochs = 5, batch_size = 16,  
                        validation_data = val_gen,
                        callbacks = [tl_checkpoint_1, early_stop, plot_loss_2],
                       verbose=1)

#%%
# Saving the trained model to working directory.
model.save('fruits1.h5')

yp_train = model.predict(x_train)
yp_train = np.argmax(yp_train, axis = 1)

yp_val = model.predict(x_val)
yp_val = np.argmax(yp_val, axis = 1)

#%%
#Load Model

load_model = models.Sequential()
load_model=tf.keras.models.load_model('fruits1.h5')

#%%
##Firstly deep learning algorithms applied to the unenhanced images then enhanced images.

images_test_path = r"D:\Spyder Verisetleri\fruits_25\Test"
fruits_test= os.listdir(images_test_path) 
#['Apple', 'Apricot', 'Banana', 'Fig', 'Kiwi', 'Mandarine', 'Orange', 'Peach', 'Pear', 'Pomegranate']

data_test=[]

for f_test in fruits_test: 
    for i, image_file in tqdm(enumerate(os.listdir(os.path.join(images_test_path, f_test)))):
        img = cv2.imread(os.path.join(images_test_path, f_test, image_file))
        image = img / 255.0
        data_test.append([image, fruits_test.index(f_test)])
        
#%%
x_test= []
y_test = []

for images, label in data_test:
    x_test.append(images)
    y_test.append(label)

# Converting to Numpy Array    
x_test = np.array(x_test)
y_test = np.array(y_test)

#%%
yp_test = model.predict(x_test)
yp_test = np.argmax(yp_test, axis = 1)

#%%
yp_test = load_model.predict(x_test)
yp_test = np.argmax(yp_test, axis = 1)
#%%

print("Train")
print(classification_report(y_train, yp_train)) 
print("Validation")
print(classification_report(y_val, yp_val))
print("Test")
print(classification_report(y_test, yp_test)) 

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_accuracy*100:.2f}%')
#%%
test_loss, test_accuracy = load_model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_accuracy*100:.2f}%')
        