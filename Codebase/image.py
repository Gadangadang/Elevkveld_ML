import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from PIL import Image
import os


def create_model():
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
        if(logs.get('acc')>0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True

def turn_image_to_array():  
    path_of_the_directory = '../Hand_drawn_images/'
    list_of_files = os.listdir(path_of_the_directory)
    y_test_spec = [] 
    x_test_spec = [] 
    
    for filename in list_of_files:
        image = tf.keras.preprocessing.image.load_img(path_of_the_directory + filename, 
                                                      color_mode="grayscale", 
                                                      target_size=(28,28))
       
        plt.imshow(image)
        plt.show()
        
        input_arr = tf.keras.preprocessing.image.img_to_array(image)

        print(np.shape(input_arr))
        
        x_test_spec.append(input_arr)
        
        # Get labels
        stringlabel = filename.split(".jpeg")[0]
        y_test_spec.append(int(stringlabel))
        
    x_test_spec = np.asarray(x_test_spec)
    y_test_spec = np.asarray(y_test_spec)
    
    print(y_test_spec, np.shape(x_test_spec))
    
    return x_test_spec, y_test_spec

    
    
  

if __name__ == "__main__":
    
    x_test_spec, y_test_spec = turn_image_to_array()
    
    
    mnist = tf.keras.datasets.mnist
    callbacks = myCallback()
    
    tf.config.set_visible_devices([], 'GPU')
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    

    input_shape = (28, 28, 1)

    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train = x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_test = x_test/255.0

    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

    batch_size = 64
    num_classes = 10
    epochs = 5

    model = create_model()
    
    #with tf.device("/physical_device: CPU: 0"):  # Use CPU
    #history = model.fit(x_train, y_train,
    #                    batch_size=batch_size,
    #                    epochs=epochs,
    #                    validation_split=0.2,
    #                    callbacks=[callbacks])
    
    #model.save('saved_model/trained_model')
    model = tf.keras.models.load_model('saved_model/trained_model')
    prediction = model.predict(x_test)
        
    prediction = np.argmax(prediction, axis=1)
    print(prediction)
    
    
    #Prediction for the handwritten images
    predict_hand = model.predict(x_test_spec)
    predict_hand = np.argmax(predict_hand, axis=1)
    print(predict_hand)
    
    print(y_test_spec)

