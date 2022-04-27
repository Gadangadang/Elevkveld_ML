import numpy as np 
import tensorflow as tf 

from utilities import *





if __name__ == "__main__":
    """
    This file is created to train the model using the MNIST dataset.
    """
    mnist = tf.keras.datasets.mnist
    callbacks = myCallback()
    
    tf.config.set_visible_devices([], 'GPU')
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    """
    digits = []
    for i in range(4):
        digits.append({"image": x_train[i], "target": y_train[i], "predict": 2})
    plot_digits(digits)
    """

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

    model = create_model(input_shape, num_classes)
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=[callbacks])
    
    model.save('saved_model/trained_model')
    
    
