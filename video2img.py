'''
[video2img.py]
10초내외의 동영상으로 구성되어 있는 Celeb-DF dataset을 이미지로 나타내기 위한 코드
'''

import cv2
from natsort import natsorted
import glob


image_path = glob.glob('C:/Users/mmclab1/Desktop/fakecheck/dataset/fake_video/*.mp4', recursive=False)
image_path = natsorted(image_path)

for path in image_path:
    vidcap = cv2.VideoCapture(path)
    count = 0

    while (vidcap.isOpened()):
        retval, image = vidcap.read()

        # 30 frame 마다 한 장씩 추출
        if (int(vidcap.get(1)) % 30 == 0):
            img_name = path.split('\\')[1]
            cv2.imwrite("C:/Users/mmclab1/Desktop/fakecheck/dataset/fake_img/%s-frame_%d.jpg" % (
            img_name.split('.')[0], count), image)
            count += 1

        if count == 4:
            break

    vidcap.release()

print('success')