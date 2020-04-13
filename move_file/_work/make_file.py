import os

import numpy as np
import matplotlib.pyplot as plt

def make_image():
    image = np.zeros((255, 255, 3))

    for i in range(10):
        file_name = "../before/image_{}.png".format(str(i+1))
        plt.imshow(image)
        plt.savefig(file_name)

def make_text():
    for i in range(10):
        file_name = "../before/text_{}.txt".format(str(i+1))
        with open(file_name, "w") as f:
            f.write(str(i+1))

def main():
    os.makedirs("../before", exist_ok=True)
    make_image()
    make_text()

if __name__ == "__main__":
    main()