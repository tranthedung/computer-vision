import cv2
import numpy as np

"""Tính giá trị trung vị của 1 mảng các giá trị"""
def calculate_median(array):
    sorted_array = np.sort(array)
    median = sorted_array[len(array)//2]
    return median
"""
    S_xy: vùng cục bộ mà bộ lọc làm việc trên đó
    z_min: giá trị cực tiểu của mức độ xám trong S_xy
    z_max: giá trị cực đại của mức độ xám trong S_xy
    z_med: giá trị trung vị của mức độ xám trong S_xy
    z_xy: giá trị mức độ xám tại (x,y)
    S_max: kích thước lớn nhất của S_xy
"""
"""Trạng thái A"""
def state_A(z_min, z_med, z_max, z_xy, S_xy, S_max):
    if(z_min < z_med < z_max):
        return state_B(z_min, z_med, z_max, z_xy, S_xy, S_max)
    else:
        S_xy += 2 #increase the size of S_xy to the next odd value.
        if(S_xy <= S_max):
            return state_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
        else:
            return z_med

"""Trạng thái B"""
def state_B(z_min, z_med, z_max, z_xy, S_xy, S_max):
    if(z_min < z_xy < z_max):
        return z_xy
    else:
        return z_med


def adaptive_median_filter(image, initial_window, max_window):

    xlength, ylength = image.shape  # get the shape of the image.

    z_min, z_med, z_max, z_xy = 0, 0, 0, 0
    S_max = max_window
    S_xy = initial_window

    output_image = image.copy()

    for row in range(S_xy, xlength - S_xy - 1):
        for col in range(S_xy, ylength - S_xy - 1):
            filter_window = image[row - S_xy: row + S_xy + 1, col - S_xy: col + S_xy + 1]
            target = filter_window.reshape(-1)  # make 1-dimensional
            z_min = np.min(target)
            z_max = np.max(target)
            z_med = calculate_median(target)
            z_xy = image[row, col]


            new_intensity = state_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
            output_image[row, col] = new_intensity
    return output_image

if __name__ == "__main__":
    # read original image
    img = cv2.imread("corrupted_images/image_1.png")
    # turn image in gray scale value
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get values with two different mask size


    # show result images
    cv2.imshow("original image", gray)
    output_img = adaptive_median_filter(gray, 3, 19)
    cv2.imshow("output image", output_img)
    status = cv2.imwrite("restored_images/restored_image_1.png", output_img)
    print("Image written to file-system : ", status)
    cv2.waitKey(0)