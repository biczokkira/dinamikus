import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit


#def calculate_max_pixel_values(images):
#    num_images, height, width = images.shape
#    max_pixel_values = np.zeros((height, width), dtype=np.uint8)
#
#    for y in range(height):
#        for x in range(width):
#            max_value = 0
#            for i in range(num_images):
#                pixel_value = images[i, y, x]
#                if pixel_value > max_value:
#                    max_value = pixel_value
#            max_pixel_values[y, x] = max_value
#
#    return max_pixel_values


def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.PGM'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            images.append(np.array(img))
    return np.array(images)


def calculate_linear_parameters(t, intensity):
    A = np.vstack([t, np.ones_like(t)]).T
    m, b = np.linalg.lstsq(A, intensity, rcond=None)[0]
    return m, b


def calculate_T1_half(dynamic_images):
    T1_half_values = np.zeros_like(dynamic_images[0])
    for y in range(dynamic_images.shape[1]):
        for x in range(dynamic_images.shape[2]):
            intensity_over_time = dynamic_images[:, y, x]
            print(intensity_over_time)
            t = np.arange(len(intensity_over_time))
            print(t)

            m, b = calculate_linear_parameters(t, intensity_over_time)
            T1_half_values[y, x] = 10 * m

    return T1_half_values


def segment_image(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    _, segmented_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    return segmented_image


def calculate_parametric_images(dynamic_images):
    PMax = np.max(dynamic_images, axis=0)

    segmented = segment_image(PMax)
    _, mask = cv2.threshold(segmented, 1, 255, cv2.THRESH_BINARY)

    TMax = np.argmax(dynamic_images, axis=0)
    #T1_half = np.argmax(dynamic_images >= 0.5 * TMax, axis=0)
    T1_half = calculate_T1_half(dynamic_images)

    TMax_copy = np.copy(TMax)
    T1_half_copy = np.copy(T1_half)
    TMax_copy[mask == 0] = 0
    T1_half_copy[mask == 0] = 0

    TMax_copy = cv2.convertScaleAbs(TMax_copy)
    T1_half_copy = cv2.convertScaleAbs(T1_half_copy)
    segmented_PMax = cv2.convertScaleAbs(segmented)

    TMax_overlay = cv2.addWeighted(TMax_copy, 1, segmented_PMax, 0, 0)
    T1_half_overlay = cv2.addWeighted(T1_half_copy, 1, segmented_PMax, 0, 0)
    return PMax, TMax_overlay, T1_half_overlay


def display_parametric_images(PMax, TMax, T1_half):
    # Display parametric images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(PMax, cmap='jet')
    axes[0].set_title('PMax')
    axes[0].axis('off')

    axes[1].imshow(TMax, cmap='jet')
    axes[1].set_title('TMax')
    axes[1].axis('off')

    axes[2].imshow(T1_half, cmap='jet')
    axes[2].set_title('T1/2')
    axes[2].axis('off')

    plt.show()


def main():
    folder = "sziv_maj_lep"
    dynamic_images = load_images_from_folder(folder)

    PMax, TMax, T1_half = calculate_parametric_images(dynamic_images)

    display_parametric_images(PMax, TMax, T1_half)


if __name__ == "__main__":
    main()
