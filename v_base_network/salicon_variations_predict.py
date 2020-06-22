import os
import numpy as np
import cv2
import scipy.ndimage
from tqdm import tqdm
import imageio as io
from salicon_data import TestSet, ToTensor, Normalize
from salicon_variations import plain_dense169_pred, two_block_dense169_pred, two_dilated_block_dense169_pred, \
    three_block_dense169_pred, three_dilated_block_dense169_pred, \
    low_high_concat_dense169_pred, middle_high_concat_dense169_pred
import torch
import torchvision as tv
from torch.utils.data import DataLoader


# set variations
# ---------------------------------------
variations_version = ['plain', 'block', 'multiscale']
variation_choice = 0

# Set model
# ---------------------------------------
blocks = ['Two', 'Three']
block_choice = 1
# ---------------------------------------

# set version (0 for no_dilation and 1 for dilation)
# ---------------------------------------
dilation = ['no_dilation', 'dilation']
dilation_choice = 1
# ---------------------------------------

# set multi-scale (0 for no_mul and 1 for mul)
# ---------------------------------------
multi_scale = ['low_high', 'middle_high']
multi_choice = 1
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
dataset = 'LSUN17'
split_set = 'val'
# ---------------------------------------


# trained model path and model type
# ---------------------------------------
if variations_version[variation_choice] == 'plain':
    model_path = '/home/leehao/video_weights/Plain169/2/plain169-sal.08-mix_loss_-1.8668.pth'
    model_ = plain_dense169_pred(pretrained=False)
    model_.load_state_dict(torch.load(model_path))
elif variations_version[variation_choice] == 'block':
    if blocks[block_choice] == 'Two' and dilation[dilation_choice] == 'no_dilation':
        model_path = '/home/leehao/video_weights/Two_DCN169/1/two_block_dcn169-sal.09-mix_loss_-1.7578.pth'
        model_ = two_block_dense169_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    elif blocks[block_choice] == 'Two' and dilation[dilation_choice] == 'dilation':
        model_path = '/home/leehao/video_weights/Two_DDCN169/1/two_block_ddcn169-sal.04-mix_loss_-1.8135.pth'
        model_ = two_dilated_block_dense169_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    elif blocks[block_choice] == 'Three' and dilation[dilation_choice] == 'no_dilation':
        model_path = '/home/leehao/video_weights/Three_DCN169/1/three_block_dcn169-sal.07-mix_loss_-1.8487.pth'
        model_ = three_block_dense169_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    elif blocks[block_choice] == 'Three' and dilation[dilation_choice] == 'dilation':
        model_path = '/home/leehao/video_weights/Three_DDCN169/1/three_block_ddcn169-sal.09-mix_loss_-1.8811.pth'
        model_ = three_dilated_block_dense169_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    else:
        raise RuntimeError('Model not exists')
elif variations_version[variation_choice] == 'multiscale':
    if multi_scale[multi_choice] == 'low_high':
        model_path = '/home/leehao/video_weights/Low_High_DDCN169/1/lh_ddcn169-sal.02-mix_loss_-1.9011.pth'
        model_ = low_high_concat_dense169_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    elif multi_scale[multi_choice] == 'middle_high':
        model_path = '/home/leehao/video_weights/Middle_High_DDCN169/1/mh_ddcn169-sal.09-mix_loss_-1.9029.pth'
        model_ = middle_high_concat_dense169_pred(pretrained=False)
        model_.load_state_dict(torch.load(model_path))
    else:
        raise RuntimeError('Model not exists')
else:
    raise NotImplemented

save_path = model_path.replace('video_weights', 'image_results').replace('.pth', '')
# ---------------------------------------

# test set path and save folder
# ---------------------------------------
test_set_path = '/data/LSUN17/val/images'
save_path_folder = os.path.join(save_path, 'salicon_val')

if sigma:
    save_path_folder = os.path.join(save_path_folder, 'sigma{}'.format(sigma))
else:
    save_path_folder = os.path.join(save_path_folder, 'no_sigma')
# ---------------------------------------

if not os.path.exists(save_path_folder):
    os.makedirs(save_path_folder)
if not os.path.isdir(save_path_folder):
    raise Exception('{} is not a dir'.format(save_path_folder))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_trans = tv.transforms.Compose([
    ToTensor(),
    Normalize(mean=mean)
])


if use_gpu_:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    print("Using gpu{}".format(os.getenv("CUDA_VISIBLE_DEVICES")))


def predict(model, dataloader, use_gpu, save_folder):
    model.eval()
    for data in tqdm(dataloader, desc="predicting: {}".format(variations_version[variation_choice]), ncols=100):
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

test_set = TestSet(test_set_path, dataset=dataset, transform=test_trans)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
predict(model_, test_loader, use_gpu_, save_path_folder)
