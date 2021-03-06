import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from PIL import Image
import os

def turn_image_to_array():  
    path_of_the_directory = '../Hand_drawn_images/'
    list_of_files = os.listdir(path_of_the_directory)
    list_of_files.pop(0)
    
    y_test_spec = [] 
    x_test_spec = [] 
    
    for filename in list_of_files:
        image = tf.keras.preprocessing.image.load_img(path_of_the_directory + filename, 
                                                      color_mode="grayscale", 
                                                      target_size=(28,28))
       
        
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        x_test_spec.append(input_arr)
        
        # Get labels
        stringlabel = filename.split(".jpg")[0]
        y_test_spec.append(int(stringlabel))
        
    x_test_spec = np.asarray(x_test_spec)
    y_test_spec = np.asarray(y_test_spec)

    return x_test_spec, y_test_spec

def plot_digits(digits):
    for digit in digits:
        image, label, predict = digit["image"], digit["target"], digit["predict"]
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title(f"Tallet skrevet av eleven: {label}, tallet gjettet av maskin {predict}.", fontsize=16)
        plt.show()
        
        
def create_model(input_shape=(28,28,1), num_classes=10):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), padding='same',
                               activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['acc'])
    return model


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True
