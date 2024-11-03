import cv2

class ColorCLAHE:
    def __init__(self):
        pass

    def apply_clahe_path(self, image_path):
        image = cv2.imread(image_path)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # Daha sert uygulama için clipLimit ve tileGridSize artırıldı

        # Renkli görüntüde CLAHE uygulama
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab_image)
        l_channel_clahe = clahe.apply(l_channel)
        lab_clahe = cv2.merge((l_channel_clahe, a, b))
        optic_disc_region_clahe_color = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return optic_disc_region_clahe_color

    def apply_clahe_image(self, pic):
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

        # Renkli görüntüde CLAHE uygulama
        lab_image = cv2.cvtColor(pic, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab_image)
        l_channel_clahe = clahe.apply(l_channel)
        lab_clahe = cv2.merge((l_channel_clahe, a, b))
        optic_disc_region_clahe_color = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return optic_disc_region_clahe_color
