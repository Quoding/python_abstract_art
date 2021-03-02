# Code taken and adapted from http://www.reedbeta.com/blog/generating-abstract-images-with-random-functions/

import numpy as np, random
from PIL import Image
import struct
import scipy
import scipy.misc
import scipy.cluster


def get_random_color():

    return np.array([random.random(), random.random(), random.random()]).reshape(
        (1, 1, 3)
    )


def get_X():
    return x_array


def get_Y():
    return y_array


def safe_divide(a, b):
    return np.divide(a, np.maximum(b, 0.001))


def build_image(depth=0):
    # Get candidate function for this depth
    candidate_fcts = [
        f
        for f in functions
        if (f[0] > 0 and depth < max_depth) or (f[0] == 0 and depth >= min_depth)
    ]

    # Get a random function from candidate functions and it's number of args
    n_args, func = random.choice(candidate_fcts)

    # Debug
    # print(depth * " = " + ">  " + str(func))

    # Get arguments recursively for the chosen functions
    args = [build_image(depth + 1) for _ in range(n_args)]

    return func(*args)


def is_boring_image(img):
    NUM_CLUSTERS = 2

    img = img.resize((512, 512))
    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = np.histogram(vecs, len(codes))  # count occurrences

    max_presence = np.max(counts / (shape[0] * shape[1]))
    print(max_presence)
    if max_presence > 0.80:
        print("Found a boring image: regenerating...")
        return True
    return False


functions = [
    (0, get_random_color),
    (0, get_X),
    (0, get_Y),
    (1, np.sin),
    (1, np.cos),
    (1, np.tan),
    (1, np.cosh),
    (1, np.sinh),
    (1, np.tanh),
    (2, np.add),
    (2, np.subtract),
    (2, np.multiply),
    (2, safe_divide),
]


d_X, d_Y = 1920, 1080
x_array = np.linspace(0.0, 1.0, d_X).reshape((1, d_X, 1))
y_array = np.linspace(0.0, 1.0, d_Y).reshape((d_Y, 1, 1))

min_depth = 5
max_depth = 10


def make_image():
    is_boring = True
    while is_boring:
        img = build_image()
        img = np.tile(
            img, (d_Y // img.shape[0], d_X // img.shape[1], 3 // img.shape[2])
        )
        img = np.uint8(np.rint(img.clip(0.0, 1.0) * 255.0))
        img = Image.fromarray(img)
        is_boring = is_boring_image(img)
    return img


if __name__ == "__main__":
    img = make_image()
    img.save("out.png")
