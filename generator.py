import json
from os import walk
from os.path import basename, normpath
import random
from PIL import Image
import time
from PIL import ImageEnhance


def random_color():
    rgbl = [255, 0, 0]
    random.shuffle(rgbl)
    return tuple(rgbl)


def rand_augmentation(img):
    scale = random.uniform(0.9, 1.1)
    sharpness_level = random.uniform(0.9, 1.1)
    color_level = random.uniform(0.8, 1.2)
    contrast_level = random.uniform(0.9, 1.1)
    brightness_level = random.uniform(0.9, 1.1)
    width, height = img.size
    width = int(width * scale)
    height = int(height * scale)
    img = img.resize((width, height), Image.ANTIALIAS)
    sharpness = ImageEnhance.Sharpness(img)
    img = sharpness.enhance(sharpness_level)
    color = ImageEnhance.Color(img)
    img = color.enhance(color_level)
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(contrast_level)
    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(brightness_level)
    return img


def load_files_in_dir(path):
    images = []
    for dirName, subdirList, fileList in walk(path):
        def to_img(f):
            return Image.open('{}/{}'.format(dirName, f))

        for file in fileList:
            images.append(to_img(file))
    return images


def start(key, num_fake):
    background = load_files_in_dir('./fake/background')
    objects = load_files_in_dir('./fake/objects')
    noise = load_files_in_dir('./fake/noise')

    def rand_img_w_aug(bag):
        rand_bg = random.choice(bag).copy()
        rand_bg = rand_augmentation(rand_bg).copy()
        w, h = rand_bg.size
        return rand_bg, w, h

    def add_noise(rand_bg):
        rand_bb, bb_w, bb_h = rand_img_w_aug(noise)
        pos_x, pos_y = (random.randint(0, bg_w - bb_w), random.randint(0, bg_h - bb_h))
        rand_bg.paste(rand_bb, (pos_x, pos_y))

    def add_objects(rand_bg):
        rand_bb, bb_w, bb_h = rand_img_w_aug(objects)
        pos_x, pos_y = (random.randint(0, bg_w - bb_w), random.randint(0, bg_h - bb_h))
        cen_x = pos_x + bb_w / 2
        cen_y = pos_y + bb_h / 2
        rand_bg.paste(rand_bb, (pos_x, pos_y))
        return '{} {} {} {} {}\n'.format(0, cen_x / bg_w, cen_y / bg_h, bb_w / bg_w, bb_h / bg_h)

    for i in range(num_fake):
        rand_bg, bg_w, bg_h = rand_img_w_aug(background)
        name = '{}'.format(time.time())

        with open('./dataset/labels/{}/{}.txt'.format(key, name), 'w') as f:
            for i in range(6):
                line = add_objects(rand_bg)
                f.write(line)

        rand_bg.save('./dataset/images/{}/{}.jpg'.format(key, name))


if __name__ == '__main__':
    start('train2017', 2000)
    start('val2017', 200)
