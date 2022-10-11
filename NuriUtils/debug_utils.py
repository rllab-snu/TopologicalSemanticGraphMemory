import time
import cv2
import numpy as np


def log_time(prev_time=None, log='', return_time=False):
    if prev_time is not None :
        delta = time.time() - prev_time
        print("[TIME] ", log, delta)
    if return_time:
        return time.time(), delta
    else:
        return time.time()


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcamA, gcamB, rawA_image, rawB_image, paper_cmap=False):
    gcamA = gcamA.squeeze().cpu().numpy()
    gcamB = gcamB.squeeze().cpu().numpy()
    cmapA = cm.jet_r(gcamA)[..., :3] * 255.0
    cmapB = cm.jet_r(gcamB)[..., :3] * 255.0
    gcamA = (cmapA.astype(np.float) + rawA_image.astype(np.float)) / 2
    gcamB = (cmapB.astype(np.float) + rawB_image.astype(np.float)) / 2
    raw_image = np.concatenate((gcamA, gcamB), 1)
    for i in range(len(gcamA)):
        cv2.imwrite(filename + '_{}.png'.format(i), np.uint8(raw_image[i]))

