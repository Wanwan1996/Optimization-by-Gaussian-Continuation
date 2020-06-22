import os
import numpy as np
import cv2
import scipy.ndimage
from tqdm import tqdm
import imageio as io
from salicon_data import TestSet, ToTensor, Normalize, Resize
from salicon_resnet_models import resnet50_pred, resnet101_pred, dilated_resnet50_pred, \
    dilated_resnet101_pred, ml_dilated_resnet50_pred
from salicon_densenet_models import densenet121_pred, dilated_densenet121_pred, densenet169_pred, \
    densenet169_pool_pred, dilated_densenet169_pred, dilated_densenet169_pool_pred, ml_densenet169_pred, \
    ml_densenet169_pool_pred, ml_dilated_densenet169_pred, ml_dilated_densenet169_pool_pred
import torch
import torchvision as tv
from torch.utils.data import DataLoader


# Set model
# ---------------------------------------
model_framework = ['resnet_50', 'resnet_101', 'densenet_121', 'densenet_169']
model_choice = 3
# ---------------------------------------

# set version (0 for no_dilation and 1 for dilation)
# ---------------------------------------
version = ['no_dilation', 'dilation']
version_choice = 1
# ---------------------------------------

# set multi-scale (0 for no_mul and 1 for mul)
# ---------------------------------------
multi_scale = ['no_mul', 'mul']
multi_choice = 1
# ---------------------------------------

# set pooling layer (0 for no_pooling and 1 for pooling)
# ---------------------------------------
pooling_layer = ['no_pool', 'pool']
pooling_choice = 1
# ---------------------------------------


# Set Hyper Parameters
# ---------------------------------------
batch_size = 5
use_gpu_ = True
gpu_device = 0
drop_rate = 0.0
sigma = 0
# ---------------------------------------


# Set dataset
# ---------------------------------------
dataset = ['LSUN17', 'CAT2000', 'MIT1003', 'MIT300']
data_choice = 0
split_set = ['val', 'test']
split_choice = 0
# ---------------------------------------


# trained model path and model type
# ---------------------------------------
if version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'no_mul' and \
        pooling_layer[pooling_choice] == 'no_pool':
    if model_framework[model_choice] == 'densenet_169':
        model_path = '/home/leehao/video_weights/DCN169/1/dcn169-sal.05-mix_loss_-1.8588.pth'
        model_ = densenet169_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    elif model_framework[model_choice] == 'densenet_121':
        model_path = '/home/leehao/video_weights/DCN121/4/dcn121-sal.04-mix_loss_-1.8034'
        model_ = densenet121_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    elif model_framework[model_choice] == 'resnet_101':
        model_path = '/home/leehao/video_weights/RN121/4/rn121-sal.04-mix_loss_-1.8034.pth'
        model_ = resnet101_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    elif model_framework[model_choice] == 'resnet_50':
        model_path = '/home/leehao/video_weights/RN50/4/rn50-sal.09-mix_loss_-1.8123.pth'
        model_ = resnet50_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    else:
        raise RuntimeError('Model not exists')
elif version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'no_mul' and \
        pooling_layer[pooling_choice] == 'pool':
    if model_framework[model_choice] == 'densenet_169':
        model_path = '/home/leehao/video_weights/DCN169_P/4/dcn169_p-sal.04-mix_loss_-1.8034.pth'
        model_ = densenet169_pool_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    else:
        raise RuntimeError('Model not exists')
elif version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'mul' and \
        pooling_layer[pooling_choice] == 'no_pool':
    if model_framework[model_choice] == 'densenet_169':
        model_path = '/home/leehao/video_weights/ML_DCN169/4/ml_dcn169-sal.04-mix_loss_-1.8034.pth'
        model_ = ml_densenet169_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    # elif model_framework[model_choice] == 'densenet_121':
    #     model_path = '/home/leehao/video_weights/ML_DCN121/4/ml_dcn121-sal.04-mix_loss_-1.8034.pth'
    #     model_ = ml_densenet121_pred(pretrained=True)
    # elif model_framework[model_choice] == 'resnet_101':
    #     model_path = '/home/leehao/video_weights/ML_RN121/4/ml_rn169-sal.04-mix_loss_-1.8034.pth'
    #     model_ = ml_resnet101_pred(pretrained=True)
    # elif model_framework[model_choice] == 'resnet_50':
    #     model_path = '/home/leehao/video_weights/ML_RN50/4/ml_rn50-sal.04-mix_loss_-1.8034.pth'
    #     model_ = ml_resnet50_pred(pretrained=True)
    else:
        raise RuntimeError('Model not exists')
elif version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'mul' and \
        pooling_layer[pooling_choice] == 'pool':
    if model_framework[model_choice] == 'densenet_169':
        model_path = '/home/leehao/video_weights/ML_DCN169_P/4/ml_dcn169_p-sal.04-mix_loss_-1.8034.pth'
        model_ = ml_densenet169_pool_pred(pretrained=False)
    else:
        raise RuntimeError('Model not exists')
elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'no_mul' and \
        pooling_layer[pooling_choice] == 'no_pool':
    if model_framework[model_choice] == 'densenet_169':
        model_path = '/home/leehao/video_weights/DDCN169/12/ddcn169-sal.08-mix_loss_-1.8986.pth'
        model_ = dilated_densenet169_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    elif model_framework[model_choice] == 'densenet_121':
        model_path = '/home/leehao/video_weights/DDCN121/7/ddcn121-sal.04-mix_loss_-1.9043.pth'
        model_ = dilated_densenet121_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    elif model_framework[model_choice] == 'resnet_101':
        model_path = '/home/leehao/video_weights/DRN101/7/drn101-sal.04-mix_loss_-1.9043.pth'
        model_ = dilated_resnet101_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    elif model_framework[model_choice] == 'resnet_50':
        model_path = '/home/leehao/video_weights/DRN50/2/drn50-sal.06-mix_loss_-1.8827.pth'
        model_ = dilated_resnet50_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    else:
        raise RuntimeError('Model not exists')
elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'no_mul' and \
        pooling_layer[pooling_choice] == 'pool':
    if model_framework[model_choice] == 'densenet_169':
        model_path = '/home/leehao/video_weights/DDCN169_P/4/ddcn169_p-sal.06-mix_loss_-1.8712.pth'
        model_ = dilated_densenet169_pool_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    else:
        raise RuntimeError('Model not exists')
elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'mul' and \
        pooling_layer[pooling_choice] == 'no_pool':
    if model_framework[model_choice] == 'densenet_169':
        if dataset[data_choice] == 'LSUN17':
            model_path = '/home/leehao/video_weights/ML_DDCN169/14/ml_ddcn169-sal.09-mix_loss_-1.9050.pth'
        elif dataset[data_choice] == 'CAT2000':
            model_path = '/home/leehao/video_weights/ML_DDCN169_CAT2000/13/ml_ddcn169-sal.03-mix_loss_-2.1453.pth'
        elif dataset[data_choice] == 'MIT1003':
            if split_set[split_choice] == 'val':
                model_path = '/home/leehao/video_weights/ML_DDCN169_MIT1003/3/ml_ddcn169-sal.03-mix_loss_-2.8066.pth'
            else:
                model_path = '/home/leehao/video_weights/ML_DDCN169/14/ml_ddcn169-sal.09-mix_loss_-1.9050.pth'
        elif dataset[data_choice] == 'MIT300':
            model_path = '/home/leehao/video_weights/ML_DDCN169_MIT1003/3/ml_ddcn169-sal.03-mix_loss_-2.8066.pth'
        else:
            raise NotImplementedError
        model_ = ml_dilated_densenet169_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    # elif model_framework[model_choice] == 'densenet_121':
    #     model_path = '/home/leehao/video_weights/ML_DDCN121/7/ml_ddcn121-sal.04-mix_loss_-1.9043.pth'
    #     model_ = ml_dilated_densenet121_pred(pretrained=True)
    # elif model_framework[model_choice] == 'resnet_101':
    #     model_path = '/home/leehao/video_weights/ML_DRN101/7/ml_drn101-sal.04-mix_loss_-1.9043.pth'
    #     model_ = ml_dilated_resnet101_pred(pretrained=True)
    elif model_framework[model_choice] == 'resnet_50':
        model_path = '/home/leehao/video_weights/ML_DRN50/1/ml_drn50-sal.04-mix_loss_-1.8819.pth'
        model_ = ml_dilated_resnet50_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    else:
        raise RuntimeError('Model not exists')
elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'mul' and \
        pooling_layer[pooling_choice] == 'pool':
    if model_framework[model_choice] == 'densenet_169':
        model_path = '/home/leehao/video_weights/ML_DDCN169_P/6/ml_ddcn169_p-sal.04-mix_loss_-1.8986.pth'
        model_ = ml_dilated_densenet169_pool_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    else:
        raise RuntimeError('Model not exists')
else:
    raise NotImplemented

save_path = model_path.replace('video_weights', 'image_results').replace('.pth', '')
# ---------------------------------------

# test set path and save folder
# ---------------------------------------
if dataset[data_choice] == 'LSUN17':
    if split_set[split_choice] == 'val':
        test_set_path = '/data/LSUN17/val/images'
        save_path_folder = os.path.join(save_path, 'salicon_val')
    else:
        test_set_path = '/data/LSUN17/images'
        save_path_folder = os.path.join(save_path, 'salicon_test')
elif dataset[data_choice] == 'CAT2000':
    if split_set[split_choice] == 'val':
        test_set_path = '/data/CAT2000/val/images'
        save_path_folder = os.path.join(save_path, 'cat2000_val')
    else:
        test_set_path = '/data/CAT2000/testSet/Stimuli'
        save_path_folder = os.path.join(save_path, 'cat2000_test')
elif dataset[data_choice] == 'MIT1003':
    if split_set[split_choice] == 'val':
        test_set_path = '/data/MIT1003/val/images'
        save_path_folder = os.path.join(save_path, 'mit1003_val')
    else:
        test_set_path = '/data/MIT1003/IMAGES'
        save_path_folder = os.path.join(save_path, 'mit1003_all')
elif dataset[data_choice] == 'MIT300':
    test_set_path = '/data/MIT300/BenchmarkIMAGES'
    save_path_folder = os.path.join(save_path, 'mit300')
else:
    raise RuntimeError('dataset not exists')

if sigma:
    save_path_folder = os.path.join(save_path_folder, 'sigma{}'.format(sigma))
else:
    save_path_folder = os.path.join(save_path_folder, 'no_sigma')
# ---------------------------------------

if not os.path.exists(save_path_folder):
    os.makedirs(save_path_folder)
if not os.path.isdir(save_path_folder):
    raise Exception('{} is not a dir'.format(save_path_folder))


if use_gpu_:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    print("Using gpu{}".format(os.getenv("CUDA_VISIBLE_DEVICES")))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

if dataset[data_choice] == 'LSUN17':
    test_trans = tv.transforms.Compose([
        ToTensor(),
        Normalize(mean=mean)
    ])
elif dataset[data_choice] == 'CAT2000':
    test_trans = tv.transforms.Compose([
        Resize((270, 480)),
        ToTensor(),
        Normalize(mean=mean)
    ])
elif dataset[data_choice] == 'MIT1003':
    test_trans = tv.transforms.Compose([
        ToTensor(),
        Normalize(mean=mean)
    ])
elif dataset[data_choice] == 'MIT300':
    test_trans = tv.transforms.Compose([
        ToTensor(),
        Normalize(mean=mean)
    ])
else:
    raise RuntimeError('dataset not exists')


def predict(model, dataloader, use_gpu, save_folder, test_path):
    model.eval()
    for data in tqdm(dataloader, desc="predicting: {}".format(model_framework[model_choice]), ncols=100):
        with torch.no_grad():
            images = data['image']
            img_names = data['img_name']
            if use_gpu:
                images = images.cuda()
            outputs = model(images).data.cpu().numpy()
            # outputs = model(images).data.cpu()
            output = np.zeros(outputs.shape, dtype=np.uint8)
            # print(output.shape)
            # print(output.dtype)
            for i in range(outputs.shape[0]):
                # print(outputs[i, 0, :, :].dtype)
                # print(outputs[i, 0, :, :].shape)
                # print(outputs[i, 0, :, :].max())
                if dataset[data_choice] in ['MIT1003', 'MIT300']:
                    original_images = cv2.imread('{}/{}'.format(test_path, img_names[i]), 0)
                    outputs[i, 0, :, :] = outputs[i, 0, :, :] / np.max(outputs[i, 0, :, :]) * 255
                    image_ = postprocess_predictions(outputs[i, 0, :, :], original_images.shape[0],
                                                     original_images.shape[1])
                    if sigma:
                        image_ = scipy.ndimage.filters.gaussian_filter(image_, sigma=sigma)
                        image_ = image_ / np.max(outputs[i, 0, :, :]) * 255
                        image_ = image_.astype(np.uint8)
                        io.imwrite(os.path.join(save_folder, img_names[i].split('.')[0] + '.jpg'), image_)
                    else:
                        image_ = image_ / np.max(outputs[i, 0, :, :]) * 255
                        image_ = image_.astype(np.uint8)
                        io.imwrite(os.path.join(save_folder, img_names[i].split('.')[0] + '.jpg'), image_)
                elif dataset[data_choice] in ['LSUN17', 'CAT2000']:
                    if sigma:
                        outputs[i, 0, :, :] = scipy.ndimage.filters.gaussian_filter(outputs[i, 0, :, :], sigma=sigma)
                        outputs[i, 0, :, :] = outputs[i, 0, :, :] / np.max(outputs[i, 0, :, :]) * 255
                        output[i, 0, :, :] = outputs[i, 0, :, :].astype(np.uint8)
                        io.imwrite(os.path.join(save_folder, img_names[i].split('.')[0] + '.png'), output[i, 0, :, :])
                    else:
                        outputs[i, 0, :, :] = outputs[i, 0, :, :] / np.max(outputs[i, 0, :, :]) * 255
                        output[i, 0, :, :] = outputs[i, 0, :, :].astype(np.uint8)
                        io.imwrite(os.path.join(save_folder, img_names[i].split('.')[0] + '.png'), output[i, 0, :, :])
                        # print(output[i, 0, :, :].dtype)
                        # io.imwrite(os.path.join(save_folder, img_names[i].split('.')[0] + '.png'),
                        #            outputs[i, 0, :, :])
                else:
                    raise RuntimeError('dataset not exists')
            del images, img_names


def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    # pred = pred / np.max(pred) * 255

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    # img = scipy.ndimage.filters.gaussian_filter(img, sigma=5)
    # img = img / np.max(img) * 255

    return img


model_.load_state_dict(torch.load(model_path))
if use_gpu_:
    model_.cuda()

if dataset[data_choice] == 'CAT2000' and split_set[split_choice] == 'test':
    sub_folder = os.listdir(test_set_path)
    for sub in sub_folder:
        test_set_path_ = os.path.join(test_set_path, sub)
        save_path_folder_ = os.path.join(save_path_folder, sub)
        if not os.path.exists(save_path_folder_):
            os.makedirs(save_path_folder_)
        if not os.path.isdir(save_path_folder):
            raise Exception('{} is not a dir'.format(save_path_folder_))
        test_set = TestSet(test_set_path_, transform=test_trans)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
        predict(model_, test_loader, use_gpu_, save_path_folder_, test_set_path)
else:
    test_set = TestSet(test_set_path, dataset=dataset[data_choice], transform=test_trans)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
    predict(model_, test_loader, use_gpu_, save_path_folder, test_set_path)
