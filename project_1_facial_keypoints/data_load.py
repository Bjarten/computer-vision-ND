import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
from PIL import Image

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
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values    
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class Normalize(object):
    """Normalize the color range to [0,1] and convert a color image to grayscale if needed"""        
    def __init__(self, color = False):
        self.color = color
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        if not self.color:
            image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-2, 2]
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
        
        #Set the smallest side equal to the largest since we want a square
        if new_h > new_w:
            new_w = new_h
        else:
            new_h = new_w       
        
        randsize1 = [2, 70]
        randsize2 = [2, 30]
        randsize3 = [1, 10]

        # Check that padding dosent go outside the frame
        padding_x_1 = 0
        padding_x_2 = 0
        padding_y_1 = 0
        padding_y_2 = 0
        
        padding_size_x_1 = random.randint(randsize1[0],randsize1[1])
        padding_size_x_2 = random.randint(randsize1[0],randsize1[1])
        padding_size_y_1 = random.randint(randsize1[0],randsize1[1])
        padding_size_y_2 = random.randint(randsize1[0],randsize1[1])
        if(y - padding_size_y_1 > 0 and x - padding_size_x_1 > 0
           and x + new_w + padding_size_x_2 < w and y + new_h + padding_size_y_2 < h):
            padding_x_1 = padding_size_x_1
            padding_x_2 = padding_size_x_2
            padding_y_1 = padding_size_y_1
            padding_y_2 = padding_size_y_2
        else:
            padding_size_x_1 = random.randint(randsize2[0],randsize2[1])
            padding_size_x_2 = random.randint(randsize2[0],randsize2[1])
            padding_size_y_1 = random.randint(randsize2[0],randsize2[1])
            padding_size_y_2 = random.randint(randsize2[0],randsize2[1])
         
            if(y - padding_size_y_1 > 0 and x - padding_size_x_1 > 0
               and x + new_w + padding_size_x_2 < w and y + new_h + padding_size_y_2 < h):
                padding_x_1 = padding_size_x_1
                padding_x_2 = padding_size_x_2
                padding_y_1 = padding_size_y_1
                padding_y_2 = padding_size_y_2
                
            else:
                padding_size_x_1 = random.randint(randsize3[0],randsize3[1])
                padding_size_x_2 = random.randint(randsize3[0],randsize3[1])
                padding_size_y_1 = random.randint(randsize3[0],randsize3[1])
                padding_size_y_2 = random.randint(randsize3[0],randsize3[1])
         
                if(y - padding_size_y_1 > 0 and x - padding_size_x_1 > 0
                   and x + new_w + padding_size_x_2 < w and y + new_h + padding_size_y_2 < h):
                    padding_x_1 = padding_size_x_1
                    padding_x_2 = padding_size_x_2
                    padding_y_1 = padding_size_y_1
                    padding_y_2 = padding_size_y_2 
        
        
        image_copy = image_copy[y - padding_y_1: y + new_h + padding_y_2, x - padding_x_1: x + new_w + padding_x_2]     
        
        key_pts = key_pts - [x - padding_x_1, y - padding_y_1] 
        
        return {'image': image_copy, 'keypoints': key_pts}
    
class FaceCropTight(object):
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
        
        #Set the smallest side equal to the largest since we want a square
        if new_h > new_w:
            new_w = new_h
        else:
            new_h = new_w       
        
        randsize1 = [5, 10]

        # Check that padding dosent go outside the frame
        padding_x_1 = 0
        padding_x_2 = 0
        padding_y_1 = 0
        padding_y_2 = 0
        
        padding_size_x_1 = random.randint(randsize1[0],randsize1[1])
        padding_size_x_2 = random.randint(randsize1[0],randsize1[1])
        padding_size_y_1 = random.randint(randsize1[0],randsize1[1])
        padding_size_y_2 = random.randint(randsize1[0],randsize1[1])
        
        if(y - padding_size_y_1 > 0 and x - padding_size_x_1 > 0
           and x + new_w + padding_size_x_2 < w and y + new_h + padding_size_y_2 < h):
            padding_x_1 = padding_size_x_1
            padding_x_2 = padding_size_x_2
            padding_y_1 = padding_size_y_1
            padding_y_2 = padding_size_y_2
                 
        image_copy = image_copy[y - padding_y_1: y + new_h + padding_y_2, x - padding_x_1: x + new_w + padding_x_2]     
        
        key_pts = key_pts - [x - padding_x_1, y - padding_y_1] 
        
        return {'image': image_copy, 'keypoints': key_pts}

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
    """Random 90 degree flip of image in sample"""
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        if random.choice([0, 1]) < .25:
            image_copy = np.rot90(image_copy,1)
            image_copy = np.flipud(image_copy)
            key_pts_copy = np.fliplr(key_pts_copy)
            

        return {'image': image_copy, 'keypoints': key_pts_copy}
    
class RandomGamma(object):
    """Random gamma of image in sample"""
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        image_copy = adjust_gamma(image_copy, gamma=random.uniform(0.8, 1.1)) 
        
        return {'image': image_copy, 'keypoints': key_pts_copy}
    
class ColorJitter(object):
    """ColorJitter image in sample"""
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,)
        
        image_copy = np.copy(image)
        
        key_pts_copy = np.copy(key_pts)

        image_copy = color_jitter(Image.fromarray(image_copy)) 
       
        image_copy = np.array(image_copy)
        
        return {'image': image_copy, 'keypoints': key_pts_copy}
    
def adjust_gamma(image, gamma=1.0):# build a lookup table mapping the pixel values [0, 255] to 
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

class RandomHorizontalFlip(object):
    """Random horizontal flip of image in sample"""
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        key_pts_copy_2 = np.copy(key_pts_copy)

        if random.choice([0, 1]) <= 0.5:
            # horizontally flip image
            image_copy = np.fliplr(image_copy)          
            # keypoints (x,y) = (-x,y)
            key_pts_copy[:,0] = -key_pts_copy[:, 0]
            # move keypoints form 2 kvadrant to 1 kvadrant
            key_pts_copy[:,0] = key_pts_copy[:, 0] + image_copy.shape[1]

            # since the keypoints are fliped around the y axis
            # their placment are wrong int the keypoint array.
            # E.g. the right eye and left eye is in the wrong place,
            # so the keypoints need to be correctly mirrord in the list
            
            key_pts_copy_2 = np.copy(key_pts_copy)
            
            # mirror jawline 
            key_pts_copy_2[16] = key_pts_copy[0]
            key_pts_copy_2[15] = key_pts_copy[1]
            key_pts_copy_2[14] = key_pts_copy[2]
            key_pts_copy_2[13] = key_pts_copy[3]
            key_pts_copy_2[12] = key_pts_copy[4]
            key_pts_copy_2[11] = key_pts_copy[5]
            key_pts_copy_2[10] = key_pts_copy[6]
            key_pts_copy_2[9]  = key_pts_copy[7]
            key_pts_copy_2[8]  = key_pts_copy[8]
            key_pts_copy_2[7] = key_pts_copy[9] 
            key_pts_copy_2[6] = key_pts_copy[10] 
            key_pts_copy_2[5] = key_pts_copy[11]
            key_pts_copy_2[4] = key_pts_copy[12]
            key_pts_copy_2[3] = key_pts_copy[13]
            key_pts_copy_2[2] = key_pts_copy[14]
            key_pts_copy_2[1] = key_pts_copy[15]
            key_pts_copy_2[0]  = key_pts_copy[16]
            
            # mirror eyebrowns
            key_pts_copy_2[26] = key_pts_copy[17] 
            key_pts_copy_2[25] = key_pts_copy[18] 
            key_pts_copy_2[24] = key_pts_copy[19]
            key_pts_copy_2[23] = key_pts_copy[20]
            key_pts_copy_2[22] = key_pts_copy[21]
            key_pts_copy_2[21] = key_pts_copy[22]
            key_pts_copy_2[20] = key_pts_copy[23]
            key_pts_copy_2[19] = key_pts_copy[24]
            key_pts_copy_2[18] = key_pts_copy[25] 
            key_pts_copy_2[17] = key_pts_copy[26]
            
            # mirror nose tip
            key_pts_copy_2[35] = key_pts_copy[31] 
            key_pts_copy_2[34] = key_pts_copy[32] 
            key_pts_copy_2[33] = key_pts_copy[33]
            key_pts_copy_2[32] = key_pts_copy[34]
            key_pts_copy_2[31] = key_pts_copy[35]
            
            # mirror eyes
            key_pts_copy_2[45] = key_pts_copy[36]
            key_pts_copy_2[44] = key_pts_copy[37] 
            key_pts_copy_2[43] = key_pts_copy[38]
            key_pts_copy_2[42] = key_pts_copy[39]
            key_pts_copy_2[47] = key_pts_copy[40]
            key_pts_copy_2[46] = key_pts_copy[41] 
            key_pts_copy_2[39] = key_pts_copy[42] 
            key_pts_copy_2[38] = key_pts_copy[43]
            key_pts_copy_2[37] = key_pts_copy[44]
            key_pts_copy_2[36] = key_pts_copy[45]
            key_pts_copy_2[41] = key_pts_copy[46] 
            key_pts_copy_2[40] = key_pts_copy[47] 
   
            # mirror lips
            key_pts_copy_2[54] = key_pts_copy[48]
            key_pts_copy_2[53] = key_pts_copy[49] 
            key_pts_copy_2[52] = key_pts_copy[50]
            key_pts_copy_2[51] = key_pts_copy[51]
            key_pts_copy_2[50] = key_pts_copy[52]
            key_pts_copy_2[49] = key_pts_copy[53] 
            key_pts_copy_2[48] = key_pts_copy[54]
            
            key_pts_copy_2[59] = key_pts_copy[55]
            key_pts_copy_2[58] = key_pts_copy[56]
            key_pts_copy_2[57] = key_pts_copy[57]
            key_pts_copy_2[56] = key_pts_copy[58] 
            key_pts_copy_2[55] = key_pts_copy[59]
            
            key_pts_copy_2[64] = key_pts_copy[60]
            key_pts_copy_2[63] = key_pts_copy[61] 
            key_pts_copy_2[62] = key_pts_copy[62]
            key_pts_copy_2[61] = key_pts_copy[63]
            key_pts_copy_2[60] = key_pts_copy[64]
                 
            key_pts_copy_2[67] = key_pts_copy[65] 
            key_pts_copy_2[66] = key_pts_copy[66] 
            key_pts_copy_2[65] = key_pts_copy[67]

            
        return {'image': image_copy, 'keypoints': key_pts_copy_2}

# inspired by https://github.com/macbrennan90/facial-keypoint-detection/blob/master/CV_project.ipynb and
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
class Rotate(object):
    """Rotate image in sample by an angle"""
    
    def __init__(self, rotation):
        self.rotation = rotation
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        
        rows = image.shape[0]
        cols = image.shape[1]
        
        M = cv2.getRotationMatrix2D((rows/2,cols/2),90,1)
        image_copy = cv2.warpAffine(image_copy,M,(cols,rows))
                
        
        key_pts_copy = key_pts_copy.reshape((1,136))
        new_keypoints = np.zeros(136)
        
        for i in range(68):
            coord_idx = 2*i
            old_coord = key_pts_copy[0][coord_idx:coord_idx+2]
            new_coord = np.matmul(M,np.append(old_coord,1))
            new_keypoints[coord_idx] += new_coord[0]
            new_keypoints[coord_idx+1] += new_coord[1]
        
        new_keypoints = new_keypoints.reshape((68,2))
        
        return {'image': image_copy, 'keypoints': new_keypoints}
    
class RandomRotate(object):
    """Rotate image in sample by an angle"""
    
    def __init__(self, rotation=30):
        self.rotation = rotation
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        
        rows = image.shape[0]
        cols = image.shape[1]
        
        M = cv2.getRotationMatrix2D((rows/2,cols/2),random.choice([-self.rotation, self.rotation]),1)
        image_copy = cv2.warpAffine(image_copy,M,(cols,rows))
                
        
        key_pts_copy = key_pts_copy.reshape((1,136))
        new_keypoints = np.zeros(136)
        
        for i in range(68):
            coord_idx = 2*i
            old_coord = key_pts_copy[0][coord_idx:coord_idx+2]
            new_coord = np.matmul(M,np.append(old_coord,1))
            new_keypoints[coord_idx] += new_coord[0]
            new_keypoints[coord_idx+1] += new_coord[1]
        
        new_keypoints = new_keypoints.reshape((68,2))
        
        return {'image': image_copy, 'keypoints': new_keypoints}