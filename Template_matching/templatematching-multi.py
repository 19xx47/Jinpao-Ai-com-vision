import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
def template_matching(image_path, template_path, output_folder, threshold=0.8):
    img_rgb = cv.imread(image_path)
    assert img_rgb is not None, f"File {image_path} could not be read, check with os.path.exists()"

    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    assert template is not None, f"File {template_path} could not be read, check with os.path.exists()"

    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv.imwrite(output_path, img_rgb)

if __name__ == "__main__":
    image_folder = "/home/worakan/template/images/"  # Replace with the path to the folder containing images
    template_folder = "/home/worakan/template/images-crop/"  # Replace with the path to the folder containing templates
    output_folder = "/home/worakan/template/images-output/"  # Replace with the path where you want to save the output images

    threshold = 0.8

    # Loop through all images in the folder
   # Get the first image and template file paths
    image_files = os.listdir(image_folder)
    template_files = os.listdir(template_folder)

    if len(image_files) > 0 and len(template_files) > 0:
        image_path = os.path.join(image_folder, image_files[0])
        template_path = os.path.join(template_folder, template_files[0])

        # Perform template matching and save the result
        template_matching(image_path, template_path, output_folder, threshold)

        # Show the result
        img = cv.imread(os.path.join(output_folder, os.path.basename(image_path)))
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    else:
        print("Image folder or template folder is empty.")



# Display images from the image_folder
for i, image_files in enumerate(os.listdir(image_folder)[:3]):
    image_path = os.path.join(image_folder, image_files)
    img = cv.imread(image_path)
    plt.subplot(2, 3, i+1)

    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(f"Image {i+1}")
    plt.axis('off')

# Display images from the output_folder
for i, result_file in enumerate(os.listdir(output_folder)[:3]):
    result_path = os.path.join(output_folder, result_file)
    img = cv.imread(result_path)
    plt.subplot(2, 3, i+4)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(f"Result {i+1}")
    plt.axis('off')

plt.show()
