from utilities import *


if __name__ == "__main__":
    """
    In this file we predict all the handrawn images using the trained model.
    plot_digits plots all the digits allong with the prediction. 
    """
    
    x_test_spec, y_test_spec = turn_image_to_array()
    x_test_spec =  x_test_spec/255.0

    x_test_spec = 1 - x_test_spec 
    model = tf.keras.models.load_model('saved_model/trained_model')
    predict_hand = model.predict(x_test_spec)

    predict_hand = np.argmax(predict_hand, axis=1)

    digits = []
    for i in range(len(x_test_spec)):
        digits.append({"image": x_test_spec[i], "target": y_test_spec[i], "predict": predict_hand[i]})
    plot_digits(digits)
