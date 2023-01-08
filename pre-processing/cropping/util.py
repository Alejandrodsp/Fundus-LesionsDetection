import albumentations as A
import cv2
import numpy as np

from util_keypoints import transform_keypoints, remap_labels

np.seterr(all='ignore')


def save(image, filename_image, filename_label=None, labels=None):
    cv2.imwrite(filename=f'{filename_image}', img=image)
    if labels is None:
        return
    f = open(f'{filename_label}', 'w')
    for line in labels:
        label, cords = line[0], line[1:]
        f.write(f'{label:.0f} {" ".join(map(lambda x: str(x), line[1:]))}\n')
    f.close()


def read(image, labels):
    image = cv2.imread(filename=image)
    labels = open(labels, 'r')
    labels_list = []
    for line in labels:
        label, cords = line[:-1].split(' ')[0], line[:-1].split(' ')[1:]
        labels_list.append([float(label), *[float(c) for c in cords]])
    labels = np.array([np.array(_) for _ in labels_list])
    return image, labels


def parse_cordinates(labels):
    return_list = []
    for item in labels:
        return_list.append(item[1:])
    return np.array(return_list)


def parse_labels(labels):
    return_list = []
    for item in labels:
        return_list.append(item[0])
    return np.array(return_list)


def albumentations_crop(image, labels, x_min, y_min, x_max, y_max):
    cordinates, class_labels = parse_cordinates(labels), parse_labels(labels)

    is_bbox = sum(map(lambda x: x.shape[0], cordinates)) / len(cordinates) == 4

    if is_bbox:
        transform = A.Compose(
            transforms=[A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, always_apply=True)],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']),
        )
        new = transform(image=image, bboxes=cordinates, class_labels=class_labels)
        image, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return image, labels

    # is_keypoints
    transform = A.Compose(
        transforms=[A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, always_apply=True)],
        keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']),
    )
    cordinates, labels, remap_ref = transform_keypoints(cordinates, class_labels, image)
    new = transform(image=image, keypoints=cordinates, class_labels=labels)
    image, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['keypoints'])])
    return image, remap_labels(labels, remap_ref, image)


def sanitize_pixels(x_min, y_min, x_max, y_max, height, width):
    if y_min > height:
        y_min = 0
    if x_min > width:
        x_min = 0
    if y_max > height:
        y_max = height
    if x_max > width:
        x_max = width
    return x_min, y_min, x_max, y_max


def hough_crop(image, labels):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 31)

    circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=70, param2=50)

    if circles is None:
        return image, labels

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]

    height, width, channels = image.shape
    x_min, y_min = int(x - r), int(y - r)
    x_max, y_max = int(x_min + 2 * r), int(y_min + 2 * r)

    x_min, y_min, x_max, y_max = sanitize_pixels(
        x_min=x_min, y_min=y_min,
        x_max=x_max, y_max=y_max,
        width=width, height=height,
    )

    return albumentations_crop(image=image, labels=labels, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def crop_tiles(image, labels, num_rows=2, num_cols=2, percent=0.15):
    original_height, original_width, _ = image.shape
    height = int(original_height / num_rows)
    width = int(original_width / num_cols)
    image_tiles, labels_tiles = [], []
    for row in range(num_rows):
        for col in range(num_cols):
            y_min = row * height
            y_max = y_min + height
            x_min = col * width
            x_max = x_min + width
            x_min, y_min, x_max, y_max = sanitize_pixels(
                x_min=round(x_min * (1 - percent)), y_min=round(y_min * (1 - percent)),
                x_max=round(x_max * (1 + percent)), y_max=round(y_max * (1 + percent)),
                width=original_width, height=original_height,
            )
            image_tile, labels_tile = albumentations_crop(
                image=image, labels=labels,
                x_min=x_min, y_min=y_min,
                x_max=x_max, y_max=y_max,
            )
            image_tiles.append(image_tile)
            labels_tiles.append(labels_tile)
    return image_tiles, labels_tiles


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    image_clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(8, 8))
    cl = image_clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def apply_pre_process(image):
    image_median_filter = cv2.medianBlur(src=image, ksize=5)
    image_clahe = clahe(image_median_filter)
    image_bilateral_filter = cv2.bilateralFilter(src=image_clahe, d=9, sigmaColor=75, sigmaSpace=75)
    return image_bilateral_filter
