import numpy as np


def pairwise(it):
    it = iter(it)
    while True:
        try:
            yield next(it), next(it)
        except StopIteration:
            # no more elements in the iterator
            return


def normalize_coordinates(x, y, height, width):
    x = x / (width - 1.)
    y = y / (height - 1.)
    return x, y


def denomarlize_cordinates(x, y, height, width):
    x = x * (width - 1.)
    y = y * (height - 1.)
    return x, y


def transform_keypoints(cordinates, labels, image):
    height, width, channels = image.shape
    labels_ref = []
    transformed_keypoints = []
    remap_ref = []
    for (index, cordinate), label in zip(enumerate(cordinates), labels):
        remap_ref.append(len(cordinate) / 2)
        for x, y in pairwise(cordinate):
            transformed_keypoints.append(denomarlize_cordinates(x=x, y=y, height=height, width=width))
            labels_ref.append(label)
    return transformed_keypoints, labels_ref, remap_ref


def remap_labels(labels, remap_ref, image):
    height, width, channels = image.shape
    return_array = []
    current_label = []
    count = 0
    last_index = 0
    for label in labels:
        if not current_label:
            class_label, x, y = list(label)
            x, y = normalize_coordinates(x, y, height, width)
            current_label = [class_label, x, y]
        else:
            x, y = list(label)[1:]
            x, y = normalize_coordinates(x, y, height, width)
            current_label += [x, y]
        count += 1
        if count == remap_ref[last_index]:
            last_index += 1
            return_array.append(np.array(current_label))
            current_label = []
            count = 0
    return np.array(return_array)
