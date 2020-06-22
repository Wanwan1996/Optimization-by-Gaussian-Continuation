import os
import torch
from PIL import Image
import numpy as np
import scipy.io as sio
import torchvision.transforms.functional as F
import torchvision as tv
from torch.utils.data import Dataset
import cv2


class TrainValSet(Dataset):
    def __init__(self, root_dir, dataset='LSUN17', transform=None):
        """
        Get training set or validation set from 3 datasets, LSUN17, MIT1003 and CAT2000.

        :param root_dir: root directory to training set or validation set.
        :param dataset: string, 'LSUN17', 'MIT1003' or 'CAT2000'
        :param transform: Optional transform to be applied on a sample.

        """
        self.root_dir = root_dir
        self.dataset = dataset
        self.images_path = os.path.join(root_dir, "images")
        self.fix_path = os.path.join(root_dir, "fixation")
        self.sal_path = os.path.join(root_dir, "maps")

        self.images = [f for f in os.listdir(self.images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.images.sort()
        self.fixation = [f for f in os.listdir(self.fix_path) if f.endswith('.mat')]
        self.fixation.sort()
        self.saliency = [f for f in os.listdir(self.sal_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.saliency.sort()

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_path, self.images[idx])
        image = Image.open(img_name)
        sal_name = os.path.join(self.sal_path, self.saliency[idx])
        sal = Image.open(sal_name)
        # print('images name is: {}\n'.format(img_name))
        # print('maps name is: {}\n'.format(sal_name))
        # print('image size is {}\n'.format(image.size))
        # print('saliency map size is {}\n'.format(sal.size))

        if self.dataset == 'LSUN17':
            fix_name = os.path.join(self.fix_path, self.fixation[idx])
            fix_temp = sio.loadmat(fix_name)["I"]
            fix = np.expand_dims(fix_temp, axis=0)

        elif self.dataset == 'CAT2000':
            fix_name = os.path.join(self.fix_path, self.fixation[idx])
            fix_temp = sio.loadmat(fix_name)["fixLocs"]
            fix = np.expand_dims(fix_temp, axis=0)

        elif self.dataset == 'MIT1003':
            image = np.array(image)
            sal = np.array(sal)
            image = padding(image)
            sal = padding(sal, channels=1)
            fix_name = os.path.join(self.fix_path, self.fixation[idx])
            fix_temp = sio.loadmat(fix_name)["fixLocs"]
            fix = padding_fixation(fix_temp)
            fix = np.expand_dims(fix, axis=0)

        else:
            raise RuntimeError('Dataset not supported.')

        # print('padding image size on mit1003 is {}\n'.format(image.shape))
        # print('padding sal size on mit1003 is {}\n'.format(sal.shape))
        # print('fixations name is: {}\n'.format(fix_name))
        # print('fixation map size is {}\n'.format(fix.shape))

        sample = {"image": image, "sal": sal, "fix": fix}

        if self.transform:
            sample = self.transform(sample)

        return sample


class TestSet(Dataset):
    def __init__(self, root_dir, dataset='LSUN17', transform=None):
        """
        Get test set of LSUN17.

        :param root_dir: root directory to test set.
        :param transform: Optional transform to be applied on a sample.

        """
        self.root_dir = root_dir
        self.dataset = dataset
        self.images = [f for f in os.listdir(self.root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.images.sort()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        if self.dataset in ['LSUN17', 'CAT2000']:
            img_name = os.path.join(self.root_dir, self.images[idx])
            image = Image.open(img_name)
        elif self.dataset in ['MIT1003', 'MIT300']:
            img_name = os.path.join(self.root_dir, self.images[idx])
            image = Image.open(img_name)
            image = np.array(image)
            image = padding(image)
        else:
            raise RuntimeError('Dataset not supported.')

        sample = {"image": image, "img_name": self.images[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out


def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c))

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


class ToTensor(object):
    def __call__(self, samples):
        samples["image"] = F.to_tensor(samples["image"])
        if 'sal' in samples.keys() and 'fix' in samples.keys():
            samples["sal"] = F.to_tensor(samples["sal"])
            samples["fix"] = torch.from_numpy(samples["fix"].copy()).type(torch.FloatTensor)

        if samples["image"].shape[0] == 1:
            samples["image"] = samples["image"].expand(3, samples["image"].shape[1], samples["image"].shape[2])

        # print(samples["image"].shape)
        # print(samples["image"].dtype)
        # print(samples["image"].max())

        return samples


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


class Normalize(object):
    def __init__(self, mean, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, samples):
        image = samples["image"]
        if not _is_tensor_image(image):
            raise TypeError('tensor is not a torch image.')

        if not self.std:
            for t, mean_val in zip(image, self.mean):
                t.sub_(mean_val)
        else:
            for t, mean_val, std_val in zip(image, self.mean, self.std):
                t.sub_(mean_val).div_(std_val)

        samples["image"] = image

        return samples


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, samples):
        image = samples["image"]
        if not len(self.size) == 2:
            raise TypeError('size is not a sequence.')
        samples["image"] = image.resize(self.size[::-1], self.interpolation)

        return samples


if __name__ == '__main__':

    train_path = '/Users/leehao/dataset/MIT1003/train'
    val_path = '/Users/leehao/dataset/MIT1003/val'
    test_path = '/Users/leehao/dataset/MIT1003/IMAGES'

    mean_ = [0.485, 0.456, 0.406]
    std_ = [0.229, 0.224, 0.225]

    train_trans = tv.transforms.Compose([
        # Resize((240, 320)),
        ToTensor(),
        Normalize(mean=mean_)
    ])
    val_trans = tv.transforms.Compose([
        ToTensor(),
        Normalize(mean=mean_)
    ])
    test_trans = tv.transforms.Compose([
        ToTensor(),
        Normalize(mean=mean_)
    ])

    train_data = TrainValSet(train_path, dataset='MIT1003', transform=train_trans)
    val_data = TrainValSet(val_path, dataset='MIT1003', transform=val_trans)
    test_data = TestSet(test_path, dataset='MIT1003', transform=test_trans)

    # print(train_data.__len__())
    # print(val_data.__len__())
    print(test_data.__len__())

    # samples_ = train_data.__getitem__(1)
    # for key, val in samples_.items():
    #     print('key is {}\n'.format(key))
    #     print('the type of {} is {}\n'.format(key, type(val)))
    #     print('{} tensor size is {}\n'.format(key, val.shape))
    #     print('the max value of {} is {}\n'.format(key, val.max()))
    #     print('the min value of {} is {}\n'.format(key, val.min()))
    #     print('the size of {} is {}\n'.format(key, val.size()))
    #     print('the data type of {} is {}\n'.format(key, val.dtype))

    samples_ = test_data.__getitem__(0)
    for key, val in samples_.items():
        print('key is {}\n'.format(key))
        print('the value of {} is {}\n'.format(key, val))
        print('the type of {} is {}\n'.format(key, type(val)))
        if key == 'image':
            print('the shape of {} is {}\n'.format(key, val.shape))
        else:
            print('the shape of {} is {}\n'.format(key, len(val)))
