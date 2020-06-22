from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from datetime import datetime
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from salicon_config import *
from salicon_resnet_models import resnet50_pred, resnet101_pred,\
    dilated_resnet50_pred, dilated_resnet101_pred, ml_dilated_resnet50_pred
from salicon_densenet_models import densenet121_pred, dilated_densenet121_pred,\
    densenet169_pred, dilated_densenet169_pred, densenet169_pool_pred, dilated_densenet169_pool_pred,\
    ml_densenet169_pred, ml_densenet169_pool_pred, ml_dilated_densenet169_pred, ml_dilated_densenet169_pool_pred


def train_model(model, dataloaders, optimizer, scheduler, use_gpu, num_epochs, save_path):
    since = time.time()
    best_model = model.state_dict()

    # set loss into a csv file
    # record the three metrics on train and val phase of one epoch in a row
    recorder = np.zeros((num_epochs, 2*len(eval_choice)))
    lr_list = []
    best_loss = 0

    try:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 20)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                # Set model to training mode
                if phase == 'train':
                    scheduler.step()
                    model.train(True)
                    lr_list.append('{:.2e}'.format(scheduler.get_lr()[0]))
                    # print(lr_list)
                # Set model to evaluate mode
                else:
                    # model.train(False)
                    model.eval()

                # running_loss indicates the loss summation for train or val phase of every epoch
                running_loss = 0.0
                # running_val indicates the every metric summation for train or val phase of every epoch
                running_val = dict()
                for metric in eval_choice:
                    running_val[metric] = 0.0

                # for a batch data.
                for data in tqdm(dataloaders[phase], desc="{:6}: ".format(phase), ncols=100):
                    images, sals, fixs = data["image"], data["sal"], data["fix"]

                    if phase == "train":
                        images, sals, fixs = Variable(images, requires_grad=True), \
                                             Variable(sals, requires_grad=False), \
                                             Variable(fixs, requires_grad=False)
                    else:
                        images, sals, fixs = Variable(images), Variable(sals), Variable(fixs)
                    if use_gpu:
                        images, sals, fixs = images.cuda(), sals.cuda(), fixs.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    outputs = model(images)

                    # loss mean of a batch
                    if loss_choice == 'nss':
                        loss = loss_fnuc[loss_choice](outputs, fixs)
                    elif loss_choice in ['cc', 'kld']:
                        loss = loss_fnuc[loss_choice](outputs, sals)
                    elif loss_choice == 'mix_loss':
                        loss = nss_ratio * loss_fnuc['nss'](outputs, fixs) + cc_ratio * loss_fnuc['cc'](outputs, sals)
                    else:
                        raise RuntimeError()

                    # each metric mean of a batch
                    val = dict()

                    for metric in eval_choice:
                        if metric == 'nss':
                            val[metric] = loss_fnuc[metric](outputs, fixs)
                        elif metric in ['cc', 'kld']:
                            val[metric] = loss_fnuc[metric](outputs, sals)
                        else:
                            raise RuntimeError()

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # accumulate the loss of each batch
                    running_loss += loss.item()*images.shape[0]
                    # accumulated the metric of each batch
                    for metric in eval_choice:
                        running_val[metric] += val[metric].item() * images.shape[0]

                    del images, sals, fixs, outputs, loss, val
                # number of images
                num_images = len(dataloaders[phase].dataset.images)
                # mean loss of train or val phase in one epoch
                epoch_loss = running_loss / num_images
                # each mean metric of train or val phase in one epoch
                for metric in eval_choice:
                    if metric in ['nss', 'cc']:
                        running_val[metric] /= -num_images
                    else:
                        running_val[metric] /= num_images
                    print('{}-{}: {:.4f}, lr: {}\n'.format(phase, metric, running_val[metric], lr_list[epoch]))

                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model = model.state_dict()
                if phase == "val":
                    current_model = model.state_dict()
                    if version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'no_mul' and \
                            pooling_layer[pooling_choice] == 'no_pool':
                        if model_framework[model_choice] == 'densenet_169':
                            model_name = "dcn169-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        elif model_framework[model_choice] == 'densenet_121':
                            model_name = "dcn121-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        elif model_framework[model_choice] == 'resnet_101':
                            model_name = "rn101-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        elif model_framework[model_choice] == 'resnet_50':
                            model_name = "rn50-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        else:
                            raise RuntimeError('Model not exists')
                    elif version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'no_mul' and \
                            pooling_layer[pooling_choice] == 'pool':
                        if model_framework[model_choice] == 'densenet_169':
                            model_name = "dcn169_p-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        else:
                            raise RuntimeError('Model not exists')
                    elif version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'mul' and \
                            pooling_layer[pooling_choice] == 'no_pool':
                        if model_framework[model_choice] == 'densenet_169':
                            model_name = "ml_dcn169-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        elif model_framework[model_choice] == 'densenet_121':
                            model_name = "ml_dcn121-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        elif model_framework[model_choice] == 'resnet_101':
                            model_name = "ml_rn101-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        elif model_framework[model_choice] == 'resnet_50':
                            model_name = "ml_rn50-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        else:
                            raise RuntimeError('Model not exists')
                    elif version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'mul' and \
                            pooling_layer[pooling_choice] == 'pool':
                        if model_framework[model_choice] == 'densenet_169':
                            model_name = "ml_dcn169_p-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        else:
                            raise RuntimeError('Model not exists')
                    elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'no_mul' and \
                            pooling_layer[pooling_choice] == 'no_pool':
                        if model_framework[model_choice] == 'densenet_169':
                            model_name = "ddcn169-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        elif model_framework[model_choice] == 'densenet_121':
                            model_name = "ddcn121-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        elif model_framework[model_choice] == 'resnet_101':
                            model_name = "drn101-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        elif model_framework[model_choice] == 'resnet_50':
                            model_name = "drn50-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        else:
                            raise RuntimeError('Model not exists')
                    elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'no_mul' and \
                            pooling_layer[pooling_choice] == 'pool':
                        if model_framework[model_choice] == 'densenet_169':
                            model_name = "ddcn169_p-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        else:
                            raise RuntimeError('Model not exists')
                    elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'mul' and \
                            pooling_layer[pooling_choice] == 'no_pool':
                        if model_framework[model_choice] == 'densenet_169':
                            model_name = "ml_ddcn169-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        elif model_framework[model_choice] == 'densenet_121':
                            model_name = "ml_ddcn121-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        elif model_framework[model_choice] == 'resnet_101':
                            model_name = "ml_drn101-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        elif model_framework[model_choice] == 'resnet_50':
                            model_name = "ml_drn50-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        else:
                            raise RuntimeError('Model not exists')
                    elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'mul' and \
                            pooling_layer[pooling_choice] == 'pool':
                        if model_framework[model_choice] == 'densenet_169':
                            model_name = "ml_ddcn169_p-sal.{:0>2d}-{}_{:.4f}.pth".format(epoch, loss_choice, epoch_loss)
                            torch.save(current_model, os.path.join(save_path, model_name))
                        else:
                            raise RuntimeError('Model not exists')
                    else:
                        raise NotImplemented

                if phase == "train":
                    for i in range(len(eval_choice)):
                        recorder[epoch, i] = running_val[eval_choice[i]]
                else:
                    for i in range(len(eval_choice)):
                        recorder[epoch, i + len(eval_choice)] = running_val[eval_choice[i]]

    finally:
        print('finally')
        # save configure parameters
        config_para = {'learning_strategy': learning_mode[mode_choice], 'loss': loss_choice,
                       'batch': batch_size, 'downsampling': downsampling_}
        config = pd.DataFrame(data=config_para, index=['config_para'])
        config_file_name = '{}_model_configure.csv'.format(str(datetime.now()).split('.')[0].replace(" ", "-"))
        config.to_csv(os.path.join(save_path, config_file_name))
        # save training results
        train_index = ['train_' + i for i in eval_choice]
        val_index = ['val_' + i for i in eval_choice]
        metric_index = train_index + val_index
        results = pd.DataFrame(data=recorder[:len(lr_list), :], index=lr_list, columns=metric_index)
        file_name = '{}-{}-{}_{:.4f}.csv'.format(str(datetime.now()).split('.')[0].replace(" ", "-"),
                                                 learning_mode[mode_choice].split('-')[1], loss_choice, best_loss)
        results.to_csv(os.path.join(save_path, file_name))
        if loss_choice == 'mix_loss':
            loss_ratio_choice = {'nss_ratio': nss_ratio, 'cc_ratio': cc_ratio}
            loss_ratio = pd.DataFrame(data=loss_ratio_choice, index=['loss_ratio_choice'])
            loss_file_name = '{}_loss_ratio.csv'.format(str(datetime.now()).split('.')[0].replace(" ", "-"))
            loss_ratio.to_csv(os.path.join(save_path, loss_file_name))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model)
    model_name = '{}-{}-{}_{:.4f}.pth'.format(str(datetime.now()).split('.')[0].replace(" ", "-"),
                                              learning_mode[mode_choice].split('-')[1], loss_choice, best_loss)
    torch.save(best_model, os.path.join(save_path, model_name))

    return model


if __name__ == "__main__":

    if not os.path.exists(save_path_):
        os.makedirs(save_path_)
    if not os.path.isdir(save_path_):
        raise Exception('%s is not a directory.' % save_path_)

    # data loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # structure a new network
    if version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'no_mul' and \
            pooling_layer[pooling_choice] == 'no_pool':
        if model_framework[model_choice] == 'densenet_169':
            model_ = densenet169_pred(pretrained=True)
        elif model_framework[model_choice] == 'densenet_121':
            model_ = densenet121_pred(pretrained=True)
        elif model_framework[model_choice] == 'resnet_101':
            model_ = resnet101_pred(pretrained=True)
        elif model_framework[model_choice] == 'resnet_50':
            model_ = resnet50_pred(pretrained=True)
        else:
            raise RuntimeError('Model not found')
    elif version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'no_mul' and \
            pooling_layer[pooling_choice] == 'pool':
        if model_framework[model_choice] == 'densenet_169':
            model_ = densenet169_pool_pred(pretrained=True)
        else:
            raise RuntimeError('Model not found')
    elif version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'mul' and \
            pooling_layer[pooling_choice] == 'no_pool':
        if model_framework[model_choice] == 'densenet_169':
            model_ = ml_densenet169_pred(pretrained=True)
        # elif model_framework[model_choice] == 'densenet_121':
        #     model_ = ml_densenet121_pred(pretrained=True)
        # elif model_framework[model_choice] == 'resnet_101':
        #     model_ = ml_resnet101_pred(pretrained=True)
        # elif model_framework[model_choice] == 'resnet_50':
        #     model_ = ml_resnet50_pred(pretrained=True)
        else:
            raise RuntimeError('Model not found')
    elif version[version_choice] == 'no_dilation' and multi_scale[multi_choice] == 'mul' and \
            pooling_layer[pooling_choice] == 'pool':
        if model_framework[model_choice] == 'densenet_169':
            model_ = ml_densenet169_pool_pred(pretrained=True)
        else:
            raise RuntimeError('Model not found')
    elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'no_mul' and \
            pooling_layer[pooling_choice] == 'no_pool':
        if model_framework[model_choice] == 'densenet_169':
            model_ = dilated_densenet169_pred(pretrained=True)
        elif model_framework[model_choice] == 'densenet_121':
            model_ = dilated_densenet121_pred(pretrained=True)
        elif model_framework[model_choice] == 'resnet_101':
            model_ = dilated_resnet101_pred(pretrained=True)
        elif model_framework[model_choice] == 'resnet_50':
            model_ = dilated_resnet50_pred(pretrained=True)
        else:
            raise RuntimeError('Model not found')
    elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'no_mul' and \
            pooling_layer[pooling_choice] == 'pool':
        if model_framework[model_choice] == 'densenet_169':
            model_ = dilated_densenet169_pool_pred(pretrained=True)
        else:
            raise RuntimeError('Model not found')
    elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'mul' and \
            pooling_layer[pooling_choice] == 'no_pool':
        if model_framework[model_choice] == 'densenet_169':
            if dataset[data_choice] == 'LSUN17':
                model_ = ml_dilated_densenet169_pred(pretrained=True)
            elif dataset[data_choice] in ['MIT1003', 'CAT2000']:
                model_ = ml_dilated_densenet169_pred(pretrained=False)
                model_.load_state_dict(torch.load(pretrained_model))
            else:
                raise RuntimeError('Model not found')
        # elif model_framework[model_choice] == 'densenet_121':
        #     model_ = ml_dilated_densenet121_pred(pretrained=True)
        # elif model_framework[model_choice] == 'resnet_101':
        #     model_ = ml_dilated_resnet101_pred(pretrained=True)
        elif model_framework[model_choice] == 'resnet_50':
            model_ = ml_dilated_resnet50_pred(pretrained=True)
        else:
            raise RuntimeError('Model not found')
    elif version[version_choice] == 'dilation' and multi_scale[multi_choice] == 'mul' and \
            pooling_layer[pooling_choice] == 'pool':
        if model_framework[model_choice] == 'densenet_169':
            model_ = ml_dilated_densenet169_pool_pred(pretrained=True)
        else:
            raise RuntimeError('Model not found')
    else:
        raise RuntimeError('Model not found')

    if use_gpu_:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device_)
        print("Using gpu{}".format(os.getenv("CUDA_VISIBLE_DEVICES")))
        model_.cuda()

    # Finetune
    for param in model_.parameters():
        param.requires_grad = True

    # set a optimizer
    if dataset[data_choice] == 'LSUN17':
        if learning_mode[mode_choice] == 'pretrained-same_lr':
            if model_framework[model_choice] in ['resnet_50', 'resnet_101']:
                optimizer_ = optim.SGD([{'params': model_.encoder.parameters(), 'lr': lr},
                                        {'params': model_.predictor.parameters(), 'lr': lr}],
                                       momentum=momentum, nesterov=nesterov, weight_decay=w_d)
            elif model_framework[model_choice] in ['densenet_121', 'densenet_169']:
                if multi_scale[multi_choice] == 'no_mul':
                    optimizer_ = optim.SGD([{'params': model_.encoder.features.parameters(), 'lr': lr},
                                            {'params': model_.predictor.parameters(), 'lr': lr}],
                                           momentum=momentum, nesterov=nesterov, weight_decay=w_d)
                elif multi_scale[multi_choice] == 'mul':
                    optimizer_ = optim.SGD([{'params': model_.encoder.features.parameters(), 'lr': lr},
                                            {'params': model_.encoder.integ2.parameters(), 'lr': lr},
                                            {'params': model_.encoder.integ3.parameters(), 'lr': lr},
                                            # {'params': model_.encoder.integ4.parameters(), 'lr': lr},
                                            {'params': model_.predictor.parameters(), 'lr': lr}],
                                           momentum=momentum, nesterov=nesterov, weight_decay=w_d)
                else:
                    raise NotImplemented
            else:
                raise NotImplementedError
        elif learning_mode[mode_choice] == 'pretrained-diff_lr':
            if model_framework[model_choice] in ['resnet_50', 'resnet_101']:
                optimizer_ = optim.SGD([{'params': model_.encoder.parameters(), 'lr': lr},
                                        {'params': model_.predictor.parameters(), 'lr': 10 * lr}],
                                       momentum=momentum, nesterov=nesterov, weight_decay=w_d)
            elif model_framework[model_choice] in ['densenet_121', 'densenet_169']:
                if multi_scale[multi_choice] == 'no_mul':
                    optimizer_ = optim.SGD([{'params': model_.encoder.features.parameters(), 'lr': lr},
                                            {'params': model_.predictor.parameters(), 'lr': 10 * lr}],
                                           momentum=momentum, nesterov=nesterov, weight_decay=w_d)
                elif multi_scale[multi_choice] == 'mul':
                    optimizer_ = optim.SGD([{'params': model_.encoder.features.parameters(), 'lr': lr},
                                            {'params': model_.encoder.integ2.parameters(), 'lr': 10 * lr},
                                            {'params': model_.encoder.integ3.parameters(), 'lr': 10 * lr},
                                            # {'params': model_.encoder.integ4.parameters(), 'lr': 10 * lr},
                                            {'params': model_.predictor.parameters(), 'lr': 10 * lr}],
                                           momentum=momentum, nesterov=nesterov, weight_decay=w_d)
                else:
                    raise NotImplemented
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    elif dataset[data_choice] in ['CAT2000', 'MIT1003']:
        if learning_mode[mode_choice] == 'pretrained-same_lr':
            if model_framework[model_choice] in ['resnet_50', 'resnet_101']:
                optimizer_ = optim.SGD([{'params': model_.encoder.parameters(), 'lr': lr},
                                        {'params': model_.predictor.parameters(), 'lr': lr}],
                                       momentum=momentum, nesterov=nesterov, weight_decay=w_d)
            elif model_framework[model_choice] in ['densenet_121', 'densenet_169']:
                if multi_scale[multi_choice] == 'no_mul':
                    optimizer_ = optim.SGD([{'params': model_.encoder.features.parameters(), 'lr': lr},
                                            {'params': model_.predictor.parameters(), 'lr': lr}],
                                           momentum=momentum, nesterov=nesterov, weight_decay=w_d)
                elif multi_scale[multi_choice] == 'mul':
                    optimizer_ = optim.SGD([{'params': model_.encoder.features.parameters(), 'lr': lr},
                                            {'params': model_.encoder.integ2.parameters(), 'lr': lr},
                                            {'params': model_.encoder.integ3.parameters(), 'lr': lr},
                                            # {'params': model_.encoder.integ4.parameters(), 'lr': lr},
                                            {'params': model_.predictor.parameters(), 'lr': lr}],
                                           momentum=momentum, nesterov=nesterov, weight_decay=w_d)
                else:
                    raise NotImplemented
            else:
                raise NotImplementedError
        elif learning_mode[mode_choice] == 'pretrained-diff_lr':
            if model_framework[model_choice] in ['resnet_50', 'resnet_101']:
                optimizer_ = optim.SGD([{'params': model_.encoder.parameters(), 'lr': lr},
                                        {'params': model_.predictor.parameters(), 'lr': 10 * lr}],
                                       momentum=momentum, nesterov=nesterov, weight_decay=w_d)
            elif model_framework[model_choice] in ['densenet_121', 'densenet_169']:
                if multi_scale[multi_choice] == 'no_mul':
                    optimizer_ = optim.SGD([{'params': model_.encoder.features.parameters(), 'lr': lr},
                                            {'params': model_.predictor.parameters(), 'lr': 10 * lr}],
                                           momentum=momentum, nesterov=nesterov, weight_decay=w_d)
                elif multi_scale[multi_choice] == 'mul':
                    optimizer_ = optim.SGD([{'params': model_.encoder.features.parameters(), 'lr': lr},
                                            {'params': model_.encoder.integ2.parameters(), 'lr': 10 * lr},
                                            {'params': model_.encoder.integ3.parameters(), 'lr': 10 * lr},
                                            # {'params': model_.encoder.integ4.parameters(), 'lr': 10 * lr},
                                            {'params': model_.predictor.parameters(), 'lr': 10 * lr}],
                                           momentum=momentum, nesterov=nesterov, weight_decay=w_d)
                else:
                    raise NotImplemented
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    scheduler_ = optim.lr_scheduler.MultiStepLR(optimizer_, milestones=milestones, gamma=gamma)

    # Train model
    # ---------------------------------------
    best_model_ = train_model(model=model_, dataloaders={"train": train_loader, "val": val_loader},
                              optimizer=optimizer_, scheduler=scheduler_, use_gpu=use_gpu_,
                              num_epochs=n_epochs_, save_path=save_path_)
