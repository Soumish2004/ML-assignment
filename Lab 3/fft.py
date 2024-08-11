import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_n_blocks(arr, tile_size=4):
    rows, cols = arr.shape
    tilings = []

    for i in range(0, rows, tile_size):
        for j in range(0, cols, tile_size):
            tiling = arr[i:i + tile_size, j:j + tile_size]
            tilings.append(tiling)

    return np.array(tilings)


def get_block_means_vars(arr):
    means_vars = []
    for block in arr:
        means_vars.append(np.mean(block) + np.var(block))
    return np.array(means_vars)


def get_img_features(img):
    blocks = get_n_blocks(img)
    return get_block_means_vars(blocks)


def load_img_and_extract_features(filepath, label):
    img = cv2.imread(filepath)
    gray_img = to_gray(img)
    features = get_img_features(gray_img)
    return features, label


def load_imgs_from_directory(directory, label, num_images=None):
    images = []
    labels = []
    files = os.listdir(directory)
    if num_images is not None:
        files = files[:num_images]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_img_and_extract_features, os.path.join(directory, filename), label) for filename
                   in files]
        for future in as_completed(futures):
            features, lbl = future.result()
            images.append(features)
            labels.append(lbl)
    return images, labels


base_dir = "/home/dhruv/Programming/College_lab/Sem5/MachineLearning19CSE305/train"
fake_dir = os.path.join(base_dir, 'FAKE')
real_dir = os.path.join(base_dir, 'REAL')

num_images_to_load = 8000

fake_images, fake_labels = load_imgs_from_directory(fake_dir, 0, num_images=num_images_to_load)
real_images, real_labels = load_imgs_from_directory(real_dir, 1, num_images=num_images_to_load)

features = np.vstack((fake_images, real_images))
labels = np.concatenate((fake_labels, real_labels))


shuffled_indices = np.random.permutation(len(labels))
features = features[shuffled_indices]
labels = labels[shuffled_indices]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)
print("Accuracy:", accuracy)
print(knn.predict(X_test))
