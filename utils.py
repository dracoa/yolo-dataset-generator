import glob

import cv2

from sift_matcher import sift_match


def show(img):
    cv2.imshow('image', img)
    if cv2.waitKey(0) == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()


def cropBarArea():
    files = list(map(lambda f: cv2.imread(f), glob.glob('./fake/*.png')))
    target = cv2.imread('./diablo/objects/skill-bar.png')
    ind = 0
    for background in files:
        ind += 1
        x, y, w, h = sift_match(target, background)
        bh, bw, _ = background.shape
        print(x, y, w, h, bw, bh)
        crop = background[y - (h * 4):y + h, x - 50:x + w + 50]
        cv2.imwrite(f'./bar_{ind}.jpg', crop)
