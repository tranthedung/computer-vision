import cv2
import numpy as np
import os

IMAGES_DIRECTORY = 'corrupted_images/'
OUTPUT_DIRECTORY = 'restored_images/'
IMAGE_NAMES = [
    'cameraman.jpg',
]

BALANCE_ALPHA = 0.2


def save_image(display_name, save_name, image):
    cv2.imshow(display_name, image)
    cv2.imwrite(OUTPUT_DIRECTORY + save_name, image)


def get_kernel():
    return np.ones((3, 3), np.float32) / 9


def get_mean_with_kernel(filter_area, kernel):
    return np.sum(np.multiply(kernel, filter_area))


def mean_filter(image, height, width):
    kernel = get_kernel()

    for row in range(1, height + 1):
        for column in range(1, width + 1):
            filter_area = image[row - 1:row + 2, column - 1:column + 2]
            res = get_mean_with_kernel(filter_area, kernel)
            image[row][column] = res

    return image


def get_median(filter_area):
    res = np.median(filter_area)
    return res

def median_filter(image, height, width):
    for row in range(1, height + 1):
        for column in range(1, width + 1):
            filter_area = image[row - 1:row + 2, column - 1:column + 2]
            image[row][column] = get_median(filter_area)

    return image

def mean_median_balanced_filter(image, height, width):
    for row in range(1, height + 1):
        for column in range(1, width + 1):
            filter_area = image[row - 1:row + 2, column - 1:column + 2]
            mean_filter_vector = get_mean_with_kernel(filter_area, get_kernel())
            median_filter_vector = get_median(filter_area)
            image[row][column] = BALANCE_ALPHA * mean_filter_vector + (1 - BALANCE_ALPHA) * median_filter_vector
    return image


def filter_image(image, image_name, filter_name, filtering_function):
    height, width = image.shape[:2]

    # Thêm phần đệm được phản chiếu 1px để cho phép các kernel hoạt động bình thường
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)

    res = filtering_function(image, height, width)

    return res


def main():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    for image_name in IMAGE_NAMES:
        image = cv2.imread(IMAGES_DIRECTORY + image_name, 0)
        cv2.imshow('Original Image: %s' % image_name, image)

        filtered_image = filter_image(image, image_name, 'mean filter', mean_filter)
        save_image('Mean filtered Image: %s' % image_name, '%s_mean.jpg' % image_name, filtered_image)

        filtered_image = filter_image(image, image_name, 'median filter', median_filter)
        save_image('Median filtered Image: %s' % image_name, '%s_median.jpg' % image_name, filtered_image)

        filtered_image = filter_image(image, image_name, 'balanced filter', mean_median_balanced_filter)
        save_image('Mean & Median with balance %s filtered Image: %s' % (BALANCE_ALPHA, image_name),
                   '%s_mean_median%s.jpg' % (image_name, str(BALANCE_ALPHA)), filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
