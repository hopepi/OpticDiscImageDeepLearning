import cv2
import numpy as np


class OpticDiscTrash:
    def __init__(self, resim_yol):
        self.resim_yol = resim_yol
        self.resim = cv2.imread(resim_yol)

    def apply_clahe(self):
        height, width = self.resim.shape[:2]

        roi = self.resim[int(height * 0.25):int(height * 0.75), int(width * 0.25):int(width * 0.75)]

        blurred_image = cv2.GaussianBlur(roi, (5, 5), 0)

        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = [c for c in contours if cv2.contourArea(c) > 500]  # Alan eşiği
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                x, y, w, h = cv2.boundingRect(approx)

                padding = 35  # Padding ekle
                optic_disc_region = roi[
                                    max(y - padding, 0):min(y + h + padding, roi.shape[0]),
                                    max(x - padding, 0):min(x + w + padding, roi.shape[1])]

                optic_disc_region_resized = cv2.resize(optic_disc_region, (224, 224))

                lab = cv2.cvtColor(optic_disc_region_resized, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))

                contrast_enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                return contrast_enhanced_image
        return None

