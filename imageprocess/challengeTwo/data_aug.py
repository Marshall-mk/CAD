import cv2
import glob


def vertical_flip(image_path):
    img = cv2.imread(image_path)
    img = cv2.flip(img, 0)
    return img


def zoom(image_path, zoom_factor=0.5):
    img = cv2.imread(image_path)
    (height, width) = img.shape[:2]
    center = (width / 2, height / 2)
    zoom_matrix = cv2.getRotationMatrix2D(center, 0, zoom_factor)
    return cv2.warpAffine(img, zoom_matrix, (width, height))


def horizontal_flip(image_path):
    img = cv2.imread(image_path)
    img = cv2.flip(img, 1)
    return img


if __name__ == "__main__":
    image_list = glob.glob("../../dataset/preprocessed_3_class_train/scc/*.jpg")

    for i in range(len(image_list)):
        orig_image_name = image_list[i].split(".jpg")[0]

        hor = horizontal_flip(image_path=str(image_list[i]))
        cv2.imwrite(f"{orig_image_name}_hor.jpg", hor)

        ver = vertical_flip(image_list[i])
        cv2.imwrite(f"{orig_image_name}_ver.jpg", ver)

        zoomed = zoom(image_list[i], zoom_factor=1.5)
        cv2.imwrite(f"{orig_image_name}_zoom.jpg", zoomed)
    print("Data augmentation complete")