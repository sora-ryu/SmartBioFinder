import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

import importlib
import numpy as np 
import cv2


def prediction(image_filename, kerasModel):
    test_image = load_img(image_filename, target_size = (50, 50))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = preprocess_input(test_image)
    prediction_probability = kerasModel.predict(test_image)

    return prediction_probability

def build_dataset(train_folder, val_folder, params):

    # Basic setting
    epochs = params['epochs']
    batch_size = params['batch_size']
    input_shape = params['input_shape']
    pretrained_model_name = params['pretrained_model']

    # 'preprocess_input' function varies based on the backbone model type
    if pretrained_model_name.startswith('VGG16'):
        backbone = getattr(importlib.import_module("tensorflow.keras.applications.vgg16"), pretrained_model_name)
        preprocess_input = getattr(importlib.import_module("tensorflow.keras.applications.vgg16"), 'preprocess_input')
    
    elif pretrained_model_name.startswith('ResNet50'):
        backbone = getattr(importlib.import_module("tensorflow.keras.applications.resnet50"), pretrained_model_name)
        preprocess_input = getattr(importlib.import_module("tensorflow.keras.applications.resnet50"), 'preprocess_input')
   
    elif pretrained_model_name.startswith('InceptionV3'):
        backbone = getattr(importlib.import_module("tensorflow.keras.applications.inception_v3"), pretrained_model_name)
        preprocess_input = getattr(importlib.import_module("tensorflow.keras.applications.inception_v3"), 'preprocess_input')
    
    elif pretrained_model_name.startswith('Xception'):
        backbone = getattr(importlib.import_module("tensorflow.keras.applications.xception"), pretrained_model_name)
        preprocess_input = getattr(importlib.import_module("tensorflow.keras.applications.xception"), 'preprocess_input')

    else:
        print("Please check the pretrained model name...")
        raise
    
    # Create dataset from image directories
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       width_shift_range=0.1,   # data augmentation
                                       height_shift_range=0.1,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       rotation_range=10,
                                       fill_mode='nearest')

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory( 
        train_folder, 
        target_size = input_shape[:2],
        batch_size = batch_size, 
        class_mode = 'categorical', 
        shuffle=True,
        seed=42) 

    val_generator = val_datagen.flow_from_directory( 
        val_folder, 
        target_size = input_shape[:2],
        batch_size = batch_size, 
        class_mode = 'categorical',
        shuffle=True,
        seed=42)
    
    return train_generator, val_generator, backbone


def create_models(hp, backbone, params, compile_params):
    
    # Load pretrained model - classification layers are excluded
    base_model = backbone(include_top = False, weights = 'imagenet', input_shape = params['input_shape'])

    # Define the freezing layers during training
    finetune = hp.Int('finetune', 3, 5, step=1)
    if finetune > 0:
        for layer in base_model.layers[:-finetune]:
            layer.trainable = False
    
    else:
        for layer in base_model.layers[:]:
            layer.trainable = False

    # Add the last layers for our specific domain (Bootstrapping a new top model onto the pretrained layers)
    init_weights = hp.Choice('kernel_initializer', values = ['uniform', 'normal', 'he_uniform'])
    
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(hp.Choice('units_1', [30, 50, 100]), activation='relu', kernel_initializer=init_weights)(x)
    x = layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1))(x)
    x = layers.Dense(hp.Choice('units_2', [10, 30]), activation='relu', kernel_initializer=init_weights)(x)
    output = layers.Dense(params['n_classes'], activation = 'softmax', kernel_initializer = init_weights)(x)

    model = keras.models.Model(inputs = [base_model.input], outputs = [output])
    model.summary()

    # Tune the learning rate
    initial_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
    decay_rate = hp.Float('decay_rate', 0.5, 1.0, step = 0.01)

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps = compile_params['step_size_train'] * 2,
        decay_rate = decay_rate,
        staircase = True
    )

    # Tune the optimizer
    opt_name = hp.Choice('optimizer', values = ['Adam', 'SGD'])
    if opt_name == 'Adam':
        optimizer = Adam(learning_rate = lr_schedule)
    elif opt_name == 'SGD':
        optimizer = SGD(learning_rate = lr_schedule)
    else:
        raise
    
    # Compile the model
    model.compile(
        optimizer = optimizer,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model


def exp_decay_scehdule(lr, decay_rate, epoch):
    return lr * np.exp(-decay_rate * epoch)

def create_model_opt(hp_params, backbone, params, compile_params):
    # Load pretrained model - classification layers are excluded
    base_model = backbone(include_top = False, weights = 'imagenet', input_shape = params['input_shape'])

    # Define the freezing layers during training
    finetune = hp_params['finetune']
    if finetune > 0:
        for layer in base_model.layers[:-finetune]:
            layer.trainable = False
    
    else:
        for layer in base_model.layers[:]:
            layer.trainable = False

    # Add the last layers for our specific domain (Bootstrapping a new top model onto the pretrained layers)
    init_weights = hp_params['kernel_initializer']
    
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(hp_params['units_1'], activation='relu', kernel_initializer=init_weights)(x)
    x = layers.Dropout(hp_params['dropout'])(x)
    x = layers.Dense(hp_params['units_2'], activation='relu', kernel_initializer=init_weights)(x)
    output = layers.Dense(params['n_classes'], activation = 'softmax', kernel_initializer = init_weights)(x)

    model = keras.models.Model(inputs = [base_model.input], outputs = [output])
    model.summary()

    # Tune the learning rate
    initial_learning_rate = hp_params['learning_rate']
    decay_rate = hp_params['decay_rate']

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps = compile_params['step_size_train'] * 2,
        decay_rate = decay_rate,
        staircase = True
    )

    # Tune the optimizer
    opt_name = hp_params['optimizer']
    if opt_name == 'Adam':
        optimizer = Adam(learning_rate = lr_schedule)
    elif opt_name == 'SGD':
        optimizer = SGD(learning_rate = lr_schedule)
    else:
        raise
    
    # Compile the model
    model.compile(
        optimizer = optimizer,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model