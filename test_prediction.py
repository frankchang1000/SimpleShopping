from webapp import cashierpos

import cv2
import matplotlib.pyplot as plt

from pyzbar import pyzbar

if __name__ == "__main__":
    image = cv2.imread("data/lemon.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

    print(cashierpos.prediction(image=image))