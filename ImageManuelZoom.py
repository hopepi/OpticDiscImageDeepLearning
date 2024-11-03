import cv2
import numpy as np


class ImageSelector:
    def __init__(self, image):
        self.start_point = None
        self.end_point = None
        self.drawing = False

        self.image = image
        self.clone = self.image.copy()

        self.scaled_image = cv2.resize(self.image, (800, 600))

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.draw_rectangle)

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            cv2.destroyWindow('Image')

    def select_region(self):
        while True:
            temp_image = self.scaled_image.copy()
            if self.start_point and self.end_point:
                cv2.rectangle(temp_image, self.start_point, self.end_point, (255, 0, 0), 2)
            cv2.imshow('Image', temp_image)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
                break

        if self.start_point and self.end_point:
            h_scale = self.image.shape[0] / 600
            w_scale = self.image.shape[1] / 800

            orig_start_point = (int(self.start_point[0] * w_scale), int(self.start_point[1] * h_scale))
            orig_end_point = (int(self.end_point[0] * w_scale), int(self.end_point[1] * h_scale))

            roi = self.clone[orig_start_point[1]:orig_end_point[1], orig_start_point[0]:orig_end_point[0]]
            return roi

        return None

