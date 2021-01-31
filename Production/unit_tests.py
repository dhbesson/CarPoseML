import unittest
import cv2
import pandas as pd
import numpy as np
import logging
from math import sin, cos, pi

logging.basicConfig(level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

IMG_WIDTH = 1024
IMG_HEIGHT = IMG_WIDTH // 16 * 5
IMG_SHAPE = (2710, 3384, 3)
MODEL_SCALE = 8
DISTANCE_THRESH_CLEAR = 2

points_df_path = 'points_df.pkl'
points_df = pd.read_pickle(points_df_path)

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)

def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    
    Rot = np.dot(Y, np.dot(P, R))
    return np.dot(Y, np.dot(P, R))

def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    # logging.debug(f'(x,y,z) ({x},{y},{z}) (px, py) ({x * fx / z + cx},{y * fy / z + cy}')
    return x * fx / z + cx, y * fy / z + cy

def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:,::-1]
    return (img / 255).astype('float32')

class TestPreprocessingMethods(unittest.TestCase):

    def test_euler_to_Rot(self):
        true_arr = np.array([[1,0,0],[0,1,0],[0,0,1]])
        yaw, pitch, roll = 0, 0, 0
        test_arr = euler_to_Rot(yaw, pitch, roll)
        self.assertAlmostEqual(test_arr.tolist(), true_arr.tolist())


    def test_preprocess_image(self):
        true_shape = (2710, 3384, 3)
        impath = './ID_00ab59fa6.jpg'
        img = imread(impath)
        shape = img.shape
        self.assertEqual(shape, true_shape)

    def test_convert_3d_to_2d(self):
        true_coords = (2454.4205333333334, 2892.235366666667)
        x, y, z = 1, 2, 3
        coords = convert_3d_to_2d(1,2,3)
        self.assertAlmostEqual(coords, true_coords)


if __name__ == '__main__':
    unittest.main()
