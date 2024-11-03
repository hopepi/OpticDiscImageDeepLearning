import tensorflow as tf
from keras.src.saving import load_model
from keras.src.utils.image_utils import img_to_array
import numpy as np
from PIL import Image

class ImageModelPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.target_size = (224, 224)

    def preprocess_image(self, cv_image):
        image = Image.fromarray(cv_image)
        image = image.resize(self.target_size)
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def get_prediction(self, cv_image):
        processed_image = self.preprocess_image(cv_image)
        prediction = self.model.predict(processed_image)
        return prediction