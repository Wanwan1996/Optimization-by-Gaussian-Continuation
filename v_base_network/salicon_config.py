#########################################################################
# MODEL PARAMETERS														#
#########################################################################
import torchvision as tv
from salicon_loss import NSSLoss, CCLoss, KLDLoss
from salicon_data import TrainValSet, ToTensor, Normalize, Resize
import platform

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
pooling_choice = 0
# ---------------------------------------


# Set Hyper Parameters
# ---------------------------------------
use_gpu_ = True 
gpu_device_ = 0
batch_size = 5
n_epochs_ = 10
# input data down_sampling
downsampling_ = False

# ---------------------------------------


# Set dataset
# ---------------------------------------
dataset = ['LSUN17', 'MIT1003', 'CAT2000']
data_choice = 0
# ---------------------------------------


# Set dataset path
# ---------------------------------------
if platform.node() == 'node002':
    if dataset[data_choice] == 'LSUN17':
        train_set_path = "/data/LSUN17/train/"
        val_set_path = "/data/LSUN17/val/"
    elif dataset[data_choice] == 'MIT1003':
        train_set_path = "/data/MIT1003/train/"
        val_set_path = "/data/MIT1003/val/"
    elif dataset[data_choice] == 'CAT2000':
        train_set_path = "/data/CAT2000/train"
        val_set_path = "/data/CAT2000/val"
    else:
        raise RuntimeError()
elif platform.node() == 'leehaodeMacBook-Pro.local':
    if dataset[data_choice] == 'LSUN17':
        train_set_path = "/Users/leehao/dataset/LSUN17/train/"
        val_set_path = "//Users/leehao/dataset/LSUN17/val/"
    elif dataset[data_choice] == 'MIT1003':
        train_set_path = "//Users/leehao/dataset/MIT1003/train/"
        val_set_path = "//Users/leehao/dataset/MIT1003/val/"
    elif dataset[data_choice] == 'CAT2000':
        train_set_path = "//Users/leehao/dataset/CAT2000/train"
        val_set_path = "//Users/leehao/dataset/CAT2000/val"
    else:
        raise RuntimeError()
else:
    raise RuntimeError()

# final upsampling factor
if dataset[data_choice] == 'CAT2000':
    shape_out_ = (1080, 1920)
elif dataset[data_choice] in ['LSUN17', 'MIT1003']:
    shape_out_ = (480, 640)
else:
    raise RuntimeError()
# ---------------------------------------

# Set loss function
# ---------------------------------------
# 'nss', 'cc', 'kld', 'mix_loss'
loss_choice = 'mix_loss'
nss_ratio = 0.5
cc_ratio = 1
eval_choice = ['nss', 'cc', 'kld']
loss_fnuc = {'nss': NSSLoss(size_average=True), 'kld': KLDLoss(size_average=True), 'cc': CCLoss(size_average=True)}
# ---------------------------------------


# Set learning rate and optimizer
# ---------------------------------------
if dataset[data_choice] == 'LSUN17':
    lr = 1e-2
    milestones = [2, 4, 6, 8]
elif dataset[data_choice] in ['MIT1003', 'CAT2000']:
    lr = 1e-4
    milestones = [3, 6, 8]
else:
    raise NotImplementedError

learning_mode = ['pretrained-same_lr', 'pretrained-diff_lr']
mode_choice = 0
gamma = 0.1
momentum = 0.9
w_d = 0.0001
nesterov = True
drop_rate_ = 0.0
# ---------------------------------------


# pre-trained: params are initialized by $pre-trained_model$
# ---------------------------------------
if dataset[data_choice] == 'LSUN17':
    if model_framework[model_choice] == 'resnet_50':
        pretrained_model = '../models/resnet50-19c8e357.pth'
    elif model_framework[model_choice] == 'resnet_101':
        pretrained_model = '../models/resnet101-5d3b4d8f.pth'
    elif model_framework[model_choice] == 'densenet_121':
        pretrained_model = '../models/densenet121-a639ec97.pth'
    elif model_framework[model_choice] == 'densenet_169':
        pretrained_model = '../models/densenet169-b2777c0a.pth'
    else:
        raise NotImplementedError

elif dataset[data_choice] in ['MIT1003', 'CAT2000']:
    if model_framework[model_choice] == 'resnet_50':
        pretrained_model = '../weights/resnet50'
    elif model_framework[model_choice] == 'resnet_101':
        pretrained_model = '../weights/resnet101'
    elif model_framework[model_choice] == 'densenet_121':
        pretrained_model = '../weights/densenet121'
    elif model_framework[model_choice] == 'densenet_169':
        pretrained_model = '../weights/ml_densenet169_14_-1.9050.pth'
    else:
        raise NotImplementedError

else:
    raise NotImplementedError
# ---------------------------------------


# trained model are save to this path
# ---------------------------------------
if dataset[data_choice] == 'LSUN17':
    if version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'no_mul' and \
            pooling_layer[pooling_choice] == 'no_pool':
        if model_framework[model_choice] == 'resnet_50':
            save_path_ = '/home/leehao/video_weights/RN50/'
        elif model_framework[model_choice] == 'resnet_101':
            save_path_ = '/home/leehao/video_weights/RN101/'
        elif model_framework[model_choice] == 'densenet_121':
            save_path_ = '/home/leehao/video_weights/DCN121/'
        elif model_framework[model_choice] == 'densenet_169':
            save_path_ = '/home/leehao/video_weights/DCN169/'
        else:
            raise NotImplementedError
    elif version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'no_mul' and \
            pooling_layer[pooling_choice] == 'pool':
        if model_framework[model_choice] == 'densenet_169':
            save_path_ = '/home/leehao/video_weights/DCN169_P/'
        else:
            raise NotImplementedError
    elif version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'mul' and \
            pooling_layer[pooling_choice] == 'no_pool':
        if model_framework[model_choice] == 'resnet_50':
            save_path_ = '/home/leehao/video_weights/ML_RN50/'
        elif model_framework[model_choice] == 'resnet_101':
            save_path_ = '/home/leehao/video_weights/ML_RN101/'
        elif model_framework[model_choice] == 'densenet_121':
            save_path_ = '/home/leehao/video_weights/ML_DCN121/'
        elif model_framework[model_choice] == 'densenet_169':
            save_path_ = '/home/leehao/video_weights/ML_DCN169/'
        else:
            raise NotImplementedError
    elif version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'mul' and \
            pooling_layer[pooling_choice] == 'pool':
        if model_framework[model_choice] == 'densenet_169':
            save_path_ = '/home/leehao/video_weights/ML_DCN169_P/'
        else:
            raise NotImplementedError
    elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'no_mul' and \
            pooling_layer[pooling_choice] == 'no_pool':
        if model_framework[model_choice] == 'resnet_50':
            save_path_ = '/home/leehao/video_weights/DRN50/'
        elif model_framework[model_choice] == 'resnet_101':
            save_path_ = '/home/leehao/video_weights/DRN101/'
        elif model_framework[model_choice] == 'densenet_121':
            save_path_ = '/home/leehao/video_weights/DDCN121/'
        elif model_framework[model_choice] == 'densenet_169':
            save_path_ = '/home/leehao/video_weights/DDCN169/'
        else:
            raise NotImplementedError
    elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'no_mul' and \
            pooling_layer[pooling_choice] == 'pool':
        if model_framework[model_choice] == 'densenet_169':
            save_path_ = '/home/leehao/video_weights/DDCN169_P/'
        else:
            raise NotImplementedError
    elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'mul' and \
            pooling_layer[pooling_choice] == 'no_pool':
        if model_framework[model_choice] == 'resnet_50':
            save_path_ = '/home/leehao/video_weights/ML_DRN50/'
        elif model_framework[model_choice] == 'resnet_101':
            save_path_ = '/home/leehao/video_weights/ML_DRN101/'
        elif model_framework[model_choice] == 'densenet_121':
            save_path_ = '/home/leehao/video_weights/ML_DDCN121/'
        elif model_framework[model_choice] == 'densenet_169':
            save_path_ = '/home/leehao/video_weights/ML_DDCN169/'
        else:
            raise NotImplementedError
    elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'mul' and \
            pooling_layer[pooling_choice] == 'pool':
        if model_framework[model_choice] == 'densenet_169':
            save_path_ = '/home/leehao/video_weights/ML_DDCN169_P/'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

elif dataset[data_choice] == 'CAT2000':
    if version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'mul' and \
            pooling_layer[pooling_choice] == 'no_pool':
        if model_framework[model_choice] == 'resnet_50':
            save_path_ = '/home/leehao/video_weights/ML_DRN50_CAT2000/'
        elif model_framework[model_choice] == 'resnet_101':
            save_path_ = '/home/leehao/video_weights/ML_DRN101_CAT2000/'
        elif model_framework[model_choice] == 'densenet_121':
            save_path_ = '/home/leehao/video_weights/ML_DDCN121_CAT2000/'
        elif model_framework[model_choice] == 'densenet_169':
            save_path_ = '/home/leehao/video_weights/ML_DDCN169_CAT2000/'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

elif dataset[data_choice] == 'MIT1003':
    if version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'mul' and \
            pooling_layer[pooling_choice] == 'no_pool':
        if model_framework[model_choice] == 'resnet_50':
            save_path_ = '/home/leehao/video_weights/ML_DRN50_MIT1003/'
        elif model_framework[model_choice] == 'resnet_101':
            save_path_ = '/home/leehao/video_weights/ML_DRN101_MIT1003/'
        elif model_framework[model_choice] == 'densenet_121':
            save_path_ = '/home/leehao/video_weights/ML_DDCN121_MIT1003/'
        elif model_framework[model_choice] == 'densenet_169':
            save_path_ = '/home/leehao/video_weights/ML_DDCN169_MIT1003/'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

else:
    raise NotImplementedError
# ---------------------------------------


# Set data iterator
# ---------------------------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
if downsampling_:
    if dataset[data_choice] in ['LSUN17', 'MIT1003']:
        train_trans = tv.transforms.Compose([
            Resize((240, 320)),
            ToTensor(),
            Normalize(mean=mean)
        ])
        val_trans = tv.transforms.Compose([
            Resize((240, 320)),
            ToTensor(),
            Normalize(mean=mean)
        ])
    elif dataset[data_choice] == 'CAT2000':
        train_trans = tv.transforms.Compose([
            Resize((270, 480)),
            ToTensor(),
            Normalize(mean=mean)
        ])
        val_trans = tv.transforms.Compose([
            Resize((270, 480)),
            ToTensor(),
            Normalize(mean=mean)
        ])
    else:
        raise NotImplementedError
else:
    train_trans = tv.transforms.Compose([
        ToTensor(),
        Normalize(mean=mean)
    ])
    val_trans = tv.transforms.Compose([
        ToTensor(),
        Normalize(mean=mean)
    ])

train_set = TrainValSet(train_set_path, dataset=dataset[data_choice], transform=train_trans)
val_set = TrainValSet(val_set_path, dataset=dataset[data_choice], transform=val_trans)
# ---------------------------------------
