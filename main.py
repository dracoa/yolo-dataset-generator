import datetime
import glob
from dataclasses import dataclass
from random import randrange, sample, choice

import albumentations as A
import cv2
from tqdm import tqdm


def paste(bg, obj, x, y):
    oh, ow, _ = obj.shape
    bg2 = bg.copy()
    bg2[y:y + oh, x:x + ow] = obj
    return bg2


@dataclass
class FakeObject:
    id: int
    name: str
    image: any


def paste_skill_object():
    pass


def background_transform():
    return A.Compose([
        A.RandomBrightnessContrast(p=0.75),
        A.GaussNoise(p=0.75),
        A.RandomGamma(p=0.75),
    ])


def object_transform():
    return A.Compose([
        A.CenterCrop(height=52, width=52, p=1),
        A.RandomBrightnessContrast(p=0.75),
        A.GaussNoise(p=0.75),
        A.RandomGamma(p=0.75),
        A.RandomScale(scale_limit=(-0.1, -0), p=0.8)
    ])


def generate(img_type, num_gen, offset):
    objects = []
    for i, file in enumerate(glob.glob('./diablo/objects/skills/*/*.png')):
        objects.append(FakeObject(id=i, name=file, image=cv2.imread(file)))

    backgrounds = list(map(lambda f: cv2.imread(f), glob.glob('./diablo/backgrounds/*')))
    bg_transform = background_transform()
    obj_transform = object_transform()

    for i in tqdm(range(num_gen)):
        bg = choice(backgrounds)
        name = f'img_{datetime.datetime.now().timestamp()}'
        bg = bg_transform(image=bg)['image']
        bh, bw, _ = bg.shape
        pw = offset * 15
        labels = []
        for skill in sample(objects, 6):
            img = obj_transform(image=skill.image)["image"]
            oh, ow, _ = img.shape
            pw += randrange(10, 20)
            ph = randrange(10, bh - oh)
            oxc, oyc = (pw + ow // 2, ph + oh // 2)
            labels.append(f'{skill.id} {oxc / bw} {oyc / bh} {ow / bw} {oh / bh}')
            bg = paste(bg, img, pw, ph)
            pw += ow

        cv2.imwrite(f'./diablo_skill/images/{img_type}/{name}.jpg', bg)
        with open(f'./diablo_skill/labels/{img_type}/{name}.txt', 'w') as f:
            for line in labels:
                f.write(f'{line}\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for i in range(1, 11):
        generate('train', 5000, i)
        generate('val', 500, i)
