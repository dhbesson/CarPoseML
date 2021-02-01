import sys
import json
import boto3
import cv2
import os
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from requests_toolbelt.multipart import decoder
from PIL import Image

import base64

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# import sys
# import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils

from urllib.parse import unquote_plus
from efficientnet_pytorch import EfficientNet

import numpy as np



# # from PIL import Image
# # import PIL.Image
# # import matplotlib as plt

# s3 = boto3.resource('s3')
# BUCKET_NAME = 'car-pic-uploads'

model_path = '/mnt/efs/new_model_2.pth' 
# model_path = '/mnt/efs/model.pth' 
car_pic_path = '/mnt/efs/test-image.jpg'
points_df_path = '/mnt/efs/points_df.pkl'

IMG_WIDTH = 1024
IMG_HEIGHT = IMG_WIDTH // 16 * 5
IMG_SHAPE = (2710, 3384, 3)
MODEL_SCALE = 8
DISTANCE_THRESH_CLEAR = 2

# s3_client = boto3.client('s3')

# img0 = imread(car_pic_path, True)
# IMG_SHAPE = img0.shape


# s3.download_file(BUCKET_NAME, "test_image", download_path)

# CHANGE TO EPOCH 25!!!!

# s3.Bucket(BUCKET_NAME).download_file("model.pth", model_path)
# s3.Bucket(BUCKET_NAME).download_file("test-image.jpg", car_pic_path)
# s3.Bucket(BUCKET_NAME).download_file("points_df.pkl", points_df_path)

points_df = pd.read_pickle(points_df_path)

# train = pd.read_csv('/mnt/efs/KaggleData/train.csv')
# test = pd.read_csv('/mnt/efs/KaggleData/sample_submission.csv')

# train_images_dir = '/mnt/efs/KaggleData/train_images/{}.jpg'
# test_images_dir = '/mnt/efs/KaggleData/test_images/{}.jpg'

# df_train, df_dev = train_test_split(train, test_size=0.01, random_state=42)
# print(df_train.head())
# df_test = test

# print(df_test.head())
from math import sin, cos

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)

def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    # plt.imshow(img)
    # plt.savefig('/tmp/imread.jpg')
    # s3_client.upload_file('/tmp/imread.jpg', 'car-image-downloads', 'imread.jpg')
    return img
    
# convert euler angle to rotation matrix
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
    return np.dot(Y, np.dot(P, R))
    
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x

def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh
    
class MyUNet(nn.Module):
    '''Mixture of previous classes'''
    def __init__(self, n_classes):
        super(MyUNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        
        self.conv0 = double_conv(5, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)
        
        self.mp = nn.MaxPool2d(2)
        
        self.up1 = up(1282 + 1024, 512)
        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))
        
        x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
        feats = self.base_model.extract_features(x_center)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(device)
        feats = torch.cat([bg, feats, bg], 3)
        
        # Add positional info
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        feats = torch.cat([feats, mesh2], 1)
        
        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x

def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:,::-1]
    # plt.imshow(img)
    # plt.savefig('/tmp/preprocess_img.jpg')
    # s3_client.upload_file('/tmp/imread.jpg', 'car-image-downloads', 'preprocess_img.jpg')
    return (img / 255).astype('float32')

def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy

def optimize_xy(r, c, x0, y0, z0, flipped=False):
    def distance_fn(xyz):
        x, y, z = xyz
        xx = -x if flipped else x
        slope_err = (xzy_slope.predict([[xx,z]])[0] - y)**2
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        y = (y + IMG_SHAPE[1] // 6) * IMG_WIDTH / (IMG_SHAPE[1] * 4 / 3) / MODEL_SCALE
        return max(0.2, (x-r)**2 + (y-c)**2) + max(0.4, slope_err)
    
    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new

def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]

def extract_coords(prediction, flipped=False):
    logits = prediction[0]
    regr_output = prediction[1:]
    points = np.argwhere(logits > 0)
    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(_regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = \
                optimize_xy(r, c,
                            coords[-1]['x'],
                            coords[-1]['y'],
                            coords[-1]['z'], flipped)
    coords = clear_duplicates(coords)
    return coords
    
def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict

def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)
    
    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict

def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7], dtype='float32')
    coords = str2coords(labels)
    # print('labels: ' + labels)
    xs, ys = get_img_coords(labels)
    for x, y, regr_dict in zip(xs, ys, coords):
        x, y = y, x
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
        y = (y + img.shape[1] // 6) * IMG_WIDTH / (img.shape[1] * 4/3) / MODEL_SCALE
        y = np.round(y).astype('int')
        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:
            mask[x, y] = 1
            regr_dict = _regr_preprocess(regr_dict, flip)
            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
    if flip:
        mask = np.array(mask[:,::-1])
        regr = np.array(regr[:,::-1])
    return mask, regr

def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x

def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image (row)
        ys: y coordinates in the image (column)
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys
    
def visualize(img, coords):
    # You will also need functions from the previous cells
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    
    img = img.copy()
    for point in coords:
        # Get values
        # print('Point info:' + str(point) + ' type: ' + str(type(point)))
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])
    
    # plt.imshow(img)
    # plt.savefig('/tmp/visualize.jpg')
    # s3_client.upload_file('/tmp/imread.jpg', 'car-image-downloads', 'visualize.jpg')
    
    return img
    
def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        # print('Points: ' + str([p_x, p_y, p_z]) )
        cv2.circle(image, (p_x, p_y),30, (0, 255, 0), -1) # FIX ME!
        # if p_x > image.shape[1] or p_y > image.shape[0]:
            # print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    # plt.imshow(image)
    # plt.savefig('/tmp/draw_points.jpg')
    # s3_client.upload_file('/tmp/imread.jpg', 'car-image-downloads', 'draw_points.jpg')
    return image
    
zy_slope = LinearRegression()
X = points_df[['z']]
y = points_df['y']
zy_slope.fit(X, y)

# Will use this model later
xzy_slope = LinearRegression()
X = points_df[['x', 'z']]
y = points_df['y']
xzy_slope.fit(X, y)

def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)

def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords
    

class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        IMG_WIDTH = 1024
        IMG_HEIGHT = IMG_WIDTH // 16 * 5
        MODEL_SCALE = 8
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)
        
        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(10) == 1
        
        # Read image
        img0 = imread(img_name, True)
        img = preprocess_image(img0, flip=flip)
        plt.imshow(img)
        plt.savefig('/mnt/efs/preprocess_img.jpg')
        img = np.rollaxis(img, 2, 0)
        
        # Get mask and regression maps
        mask, regr = get_mask_and_regr(img0, labels, flip=flip)
        regr = np.rollaxis(regr, 2, 0)
        
        return [img, mask, regr]

# train_dataset = CarDataset(df_train, train_images_dir, training=True)
# dev_dataset = CarDataset(df_dev, train_images_dir, training=False)
# test_dataset = CarDataset(df_test, test_images_dir, training=False)

# print(dev_dataset.df.head())
flip = False 

# img = preprocess_image(img0, flip=flip)
# img = np.rollaxis(img, 2, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('/mnt/efs/full_model.pt', map_location=torch.device('cpu'))
# model = MyUNet(8).to(device)
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu') ))
# torch.save(model, '/mnt/efs/full_model.pt' )

# for param_tensor in model.state_dict():
    # print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
model.eval()

# output = model(torch.tensor(img[None]).to(device)).data.cpu().numpy()

# predictions = []
# for out in output:
#     coords = extract_coords(out)
#     predictions.append(coords)
    
# output = model(torch.tensor(img[None]).to(device)).data.cpu().numpy()
# coords_pred = extract_coords(output[0])

# plt.imshow(visualize(img0, coords_pred))
# plt.savefig('/mnt/efs/pred_img.jpg')
    
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    try:
        # print(event['headers']['Content-Type'])
        body = event["body"]
        # print(event['headers'])
        content_type = event['headers']['Content-Type']
        
        body_dec = base64.b64decode(body)
        part = decoder.MultipartDecoder(body_dec, content_type).parts[0]
        # print(part.headers)
        # print(part.content)
        filename = '/mnt/efs/uploaded_img.jpg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(part.content)
        # s3_client.upload_file('/mnt/efs/uploaded_img.jpg', 'car-image-downloads', 'uploaded_img.jpg')
        
        # img_id = os.path.splitext(key)[0]
        img_id = 'img_69'
        # img_row = df_dev[df_dev['ImageId'] == img_id].index[0]
        # print('img row: ' + str(img_row))
        # upload_path = '/tmp/resized-{}'.format(tmpkey)
        
        img0 = imread(filename, True)
        img = preprocess_image(img0, flip=flip)
        plt.imshow(img)
        plt.savefig('/mnt/efs/preprocess_img.jpg')
        img = np.rollaxis(img, 2, 0)
        
        # Get mask and regression maps
        # mask, regr = get_mask_and_regr(img0, labels, flip=flip)
        # regr = np.rollaxis(regr, 2, 0)
        
        # img, mask, regr = dev_dataset[0]
        
        with torch.no_grad():
            output = model(torch.tensor(img[None]).to(device))
        output = output.data.cpu().numpy()
        
        coords_pred = []
        # output = model(torch.tensor(img[None]).to(device)).data.cpu().numpy()
        
        for out in output:
            coords_pred.append(extract_coords(out))
            
        # coords_true = extract_coords(np.concatenate([mask[None], regr], 0))
        
        with open('/tmp/coords_pred.npy', 'wb') as f:
            np.save(f, coords_pred)
            
        # with open('/tmp/coords_true.npy', 'wb') as f:
        #     np.save(f, coords_true)
        
        # print('coords_true: ' + str(coords_true))
        # print('coords_pred: ' + str(coords_pred[0]))
        plt.clf()
        with open(filename, 'wb') as f:
            f.write(part.content)
        img0 = imread(filename)
        # img = imread(train_images_dir.format(str(img_id)))
        plt.imshow(visualize(img0, coords_pred[0]))
        # fig, axes = plt.subplots(1, 2, figsize=(30,30))
        # axes[0].set_title('Ground truth')
        # axes[0].imshow(visualize(img, coords_true))
        # axes[1].set_title('Prediction')
        # axes[1].imshow(visualize(img, coords_pred[0]))
        plt.savefig('/tmp/pred_img.jpg')
        # final_image = Image.open('/tmp/pred_img.jpg')
        # k = bucket.new_key('abc/123/')
        # k.set_contents_from_string('')
        
        # s3_client.upload_file('/tmp/pred_img.jpg', 'car-image-downloads', 'new_pred_img.jpg')
        # s3_client.upload_file('/tmp/coords_pred.npy', 'car-image-downloads', img_id + '_coords_pred')
        # s3_client.upload_file('/tmp/coords_true.npy', 'car-image-downloads', img_id + '_coords_true')
        with open("/tmp/pred_img.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return {
            'headers': { "Content-Type": "image/jpg" },
            'statusCode': 200,
            'body': encoded_string.decode('utf-8'),
            'isBase64Encoded': True
        }
    except:
        print("Unexpected error:", sys.exc_info()[0])
        # return {
        #     'statusCode': 200,
        #     # 'message' : 'failure',
        # # 'body': json.dumps(str(coords_pred))
        # }
    
    # for record in event['Records']:
    #     bucket = record['s3']['bucket']['name']
    #     key = record['s3']['object']['key']
    #     # tmpkey = key.replace('/', '')
    #     download_path = '/tmp/pred_img.jpg'
    #     # print('key: ' + str(key))
    #     s3_client.download_file(bucket, key, download_path)
        
    #     img_id = os.path.splitext(key)[0]
    #     # print(df_dev.head())
    #     img_row = df_dev[df_dev['ImageId'] == img_id].index[0]
    #     # print('img row: ' + str(img_row))
    #     # upload_path = '/tmp/resized-{}'.format(tmpkey)
        
    #     # FIX THIS img_row problem!!!
    #     img, mask, regr = dev_dataset[0]
        
    #     with torch.no_grad():
    #         output = model(torch.tensor(img[None]).to(device))
    #     output = output.data.cpu().numpy()
        
    #     coords_pred = []
    #     # output = model(torch.tensor(img[None]).to(device)).data.cpu().numpy()
        
    #     for out in output:
    #         coords_pred.append(extract_coords(out))
            
    #     coords_true = extract_coords(np.concatenate([mask[None], regr], 0))
        
    #     with open('/tmp/coords_pred.npy', 'wb') as f:
    #         np.save(f, coords_pred)
            
    #     with open('/tmp/coords_true.npy', 'wb') as f:
    #         np.save(f, coords_true)
        
    #     # print('coords_true: ' + str(coords_true))
    #     # print('coords_pred: ' + str(coords_pred[0]))
        
    #     img = imread(train_images_dir.format(str(img_id)))
        
    #     fig, axes = plt.subplots(1, 2, figsize=(30,30))
    #     axes[0].set_title('Ground truth')
    #     axes[0].imshow(visualize(img, coords_true))
    #     axes[1].set_title('Prediction')
    #     axes[1].imshow(visualize(img, coords_pred[0]))
    #     plt.savefig('/tmp/pred_img.jpg')
        
    #     # k = bucket.new_key('abc/123/')
    #     # k.set_contents_from_string('')
        
    #     s3_client.upload_file(download_path, 'car-image-downloads', key)
    #     s3_client.upload_file('/tmp/coords_pred.npy', 'car-image-downloads', img_id + '_coords_pred')
    #     s3_client.upload_file('/tmp/coords_true.npy', 'car-image-downloads', img_id + '_coords_true')
        
    
