import cv2

class Sharpened:
    def __init__(self, resim):
        self.resim = resim

    def sharpened_path(self):
        blurred_image = cv2.GaussianBlur(self.resim, (5, 5), 0)
        sharpened_image = cv2.addWeighted(self.resim, 1.5, blurred_image, -0.5, 0)
        return sharpened_image
