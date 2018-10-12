import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2

import random
import matplotlib.pyplot as plt

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - image.shape[0]/2)/(image.shape[0]/4)


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
            
        img = cv2.resize(image, (new_w, new_h))
                 
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]
        
        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}
    
class FaceCrop(object):
    """ Crop out face using the keypoints as reference

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """       
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        image_copy = np.copy(image)
        
        h, w = image.shape[:2]
        
        x_max = 0
        x_min = 10000
        y_max = 0
        y_min = 10000
        
        # Find the coordinates to keypoints at the far left, far right, top and bottom
        # Also check that no keypoints are outside the image
        for coord in key_pts:
            if coord[0] > x_max:
                if coord[0] >= w:
                    x_max = w
                else:
                    x_max = coord[0]
            if coord[0] < x_min:
                if coord[0] < 0:
                    x_min = 0
                else:
                    x_min = coord[0]
            if coord[1] > y_max:
                if coord[1] >= h:
                    y_max = h
                else:
                    y_max = coord[1]
            if coord[1] < y_min:
                if coord[1] < 0:
                    y_min = 0
                else:
                    y_min = coord[1]
        
        # Set the the left corner keypoint as out crop cooridnate
        x = int(x_min)
        y = int(y_min)
        
        # Get height and width of keypoint area
        new_h = int(y_max - y_min)
        new_w = int(x_max - x_min)
        
        # Set the smallest side equal to the largest since we want a square
        if new_h > new_w:
            new_w = new_h
        else:
            new_h = new_w       
         
         # Check that padding dosent go outside the frame
        padding = 0
        if(y - 10 > 0 and x - 10 > 0 and x + new_w + 10 < w and y + new_h + 10 < h):
            padding = 10
            
        image_copy = image_copy[y - padding: y + new_h + padding, x - padding: x + new_w + padding]     
        
        key_pts = key_pts - [x - padding, y - padding] 
        
        return {'image': image_copy, 'keypoints': key_pts}
    
class RandomVerticalFlip(object):
    """Random vertical flip of image in sample"""
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        if random.choice([0, 1]) < .5:
            image = np.fliplr(image)
            key_pts = np.fliplr(key_pts)

        return {'image': image, 'keypoints': key_pts}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}
    
class Random90DegFlip(object):
    """Random vertical flip of image in sample"""
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        if random.choice([0, 1]) < .25:
            image_copy = np.rot90(image_copy,1)
            image_copy = np.flipud(image_copy)
            key_pts_copy = np.fliplr(key_pts_copy)
            

        return {'image': image_copy, 'keypoints': key_pts_copy}
    
class RandomVertiacalFlip(object):
    """Random vertical flip of image in sample"""
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        if random.choice([0, 1]) < .5:
            image_copy = np.fliplr(image_copy)
            key_pts_copy[:,0] = -key_pts_copy[:, 0]
            key_pts_copy[:,0] = key_pts_copy[:, 0] + 224
            
        return {'image': image_copy, 'keypoints': key_pts_copy}
    
class Random180DegFlip(object):
    """Random vertical flip of image in sample"""
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        if random.choice([0, 1]) < .25:
            image_copy = np.flipud(image_copy)
            key_pts_copy[:,1] = -key_pts_copy[:,1]
            key_pts_copy[:,1] = key_pts_copy[:,1] + 224
            
        return {'image': image_copy, 'keypoints': key_pts_copy}