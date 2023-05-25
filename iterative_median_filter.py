import cv2
from numpy import divide, int8, multiply, ravel, sort, zeros_like


def median_filter(gray_img, mask=3):
    """
    :param gray_img: gray image
    :param mask: mask size
    :return: image with median filter
    """

    # set image borders
    bd = int(mask / 2)
    # copy image size
    median_img = zeros_like(gray_img)
    for i in range(bd, gray_img.shape[0] - bd):
        for j in range(bd, gray_img.shape[1] - bd):
            # get mask according with mask
            kernel = ravel(gray_img[i - bd : i + bd + 1, j - bd : j + bd + 1])
            # calculate mask median
            median = sort(kernel)[int8(divide((multiply(mask, mask)), 2) + 1)]
            median_img[i, j] = median
    return median_img



if __name__ == "__main__":
    # read original image
    img = cv2.imread("corrupted_images/image_1.png")
    # turn image in gray scale value
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get values with two different mask size
    median3x3 = median_filter(gray, 3)
    median5x5 = median_filter(gray, 5)


    # show result images
    cv2.imshow("median filter with 3x3 mask", median3x3)
    cv2.imshow("median filter with 5x5 mask", median5x5)
    cv2.imshow("original image", gray)

    cv2.waitKey(0)