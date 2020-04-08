import cv2
import glob


imgs = glob.glob( +'/image*.jpg')
imgs.sort()
def concat_pic(cls_det):
    image = []
    for idx in range(len(sample_frame)):
        image.append(show(imgs[idx], cls_det[idx], idx))
    im_list_2d = [image[:len(image)//2], image[len(image)//2:]]
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])