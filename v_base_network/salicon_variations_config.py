#########################################################################
# MODEL PARAMETERS														#
#########################################################################
import torchvision as tv
from salicon_loss import NSSLoss, CCLoss, KLDLoss
from salicon_data import TrainValSet, ToTensor, Normalize, Resize

# set variations
# ---------------------------------------
variations_version = ['plain', 'block', 'multiscale']
variation_choice = 0

# Set model
# ---------------------------------------
blocks = ['two', 'three']
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
use_gpu_ = True
gpu_device_ = 0 
batch_size = 4
n_epochs_ = 10
# input data down_sampling
downsampling_ = False
# final upsampling factor
shape_out_ = (480, 640)
# ---------------------------------------


# Set dataset
# ---------------------------------------
dataset = 'LSUN17'
train_set_path = "/data/LSUN17/train/"
val_set_path = "/data/LSUN17/val/"
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
lr = 1e-2
milestones = [5]

learning_mode = ['pretrained-same_lr', 'pretrained-diff_lr']
mode_choice = 0
gamma = 0.1
momentum = 0.9
w_d = 0.0001
nesterov = True
drop_rate_ = 0.0
# ---------------------------------------


# pre-trained: params are initialized
# ---------------------------------------
pretrained_model = '../models/densenet169-b2777c0a.pth'

# trained model are save to this path
if variations_version[variation_choice] == 'plain':
    save_path_ = '/home/leehao/video_weights/Plain169/'

elif variations_version[variation_choice] == 'block':
    if blocks[block_choice] == 'two' and dilation[dilation_choice] == 'no_dilation':
        save_path_ = '/home/leehao/video_weights/Two_DCN169/'
    elif blocks[block_choice] == 'two' and dilation[dilation_choice] == 'dilation':
        save_path_ = '/home/leehao/video_weights/Two_DDCN169/'
    elif blocks[block_choice] == 'three' and dilation[dilation_choice] == 'no_dilation':
        save_path_ = '/home/leehao/video_weights/Three_DCN169/'
    elif blocks[block_choice] == 'three' and dilation[dilation_choice] == 'dilation':
        save_path_ = '/home/leehao/video_weights/Three_DDCN169/'
    else:
        raise NotImplementedError

elif variations_version[variation_choice] == 'multiscale':
    if multi_scale[multi_choice] == 'low_high' and dilation[dilation_choice] == 'dilation':
        save_path_ = '/home/leehao/video_weights/Low_High_DDCN169/'
    elif multi_scale[multi_choice] == 'middle_high' and dilation[dilation_choice] == 'dilation':
        save_path_ = '/home/leehao/video_weights/Middle_High_DDCN169/'
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

else:
    train_trans = tv.transforms.Compose([
        ToTensor(),
        Normalize(mean=mean)
    ])
    val_trans = tv.transforms.Compose([
        ToTensor(),
        Normalize(mean=mean)
    ])

train_set = TrainValSet(train_set_path, dataset=dataset, transform=train_trans)
val_set = TrainValSet(val_set_path, dataset=dataset, transform=val_trans)
# ---------------------------------------
