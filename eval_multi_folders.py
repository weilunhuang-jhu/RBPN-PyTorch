from __future__ import print_function
import argparse

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rbpn import Net as RBPN
from data import get_test_set
from functools import reduce
import numpy as np

# from scipy.misc import imsave
import scipy.io as sio
import time
import cv2
import math
import pdb


def prepare_data(opt, file_list):
    print('===> Loading datasets')
    opt.file_list = file_list
    test_set = get_test_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.file_list, opt.other_dataset, opt.future_frame)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    return testing_data_loader

def prepare_model(opt):

    gpus_list=range(opt.gpus)
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    print('===> Building model ', opt.model_type)
    if opt.model_type == 'RBPN':
        model = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=opt.nFrames, scale_factor=opt.upscale_factor)
    if cuda:
        model = torch.nn.DataParallel(model, device_ids=gpus_list)
    model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
    print('Pre-trained SR model is loaded.')
    if cuda:
        model = model.cuda(gpus_list[0])

    return model

def eval(model, testing_data_loader, opt):
    t_load_begin = time.perf_counter()
    model.eval()
    count=1
    avg_psnr_predicted = 0.0
    for batch in testing_data_loader:
        t_load_end = time.perf_counter()
        print("===> time for loading batch:" + str(t_load_end - t_load_begin))
        t_load_begin = t_load_end
        input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]

        # # get downsample imgs
        # save_img(input, str(count), True, opt)
        
        with torch.no_grad():
            input = Variable(input).cuda(0)
            bicubic = Variable(bicubic).cuda(0)
            neigbor = [Variable(j).cuda(0) for j in neigbor]
            flow = [Variable(j).cuda(0).float() for j in flow]

        t0 = time.time()
        with torch.no_grad():
            prediction = model(input, neigbor, flow) 

        if opt.residual:
            prediction = prediction + bicubic
        
        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
        save_img(prediction.cpu().data, str(count), True, opt)
        count+=1
    
def save_img(img, img_name, pred_flag, opt):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

    # save img
    save_dir=opt.output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_fn = save_dir +'/'+ img_name.zfill(8)+'.png'
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    gt = gt[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
    
def main():

    # Arguments settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--chop_forward', type=bool, default=False)
    parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
    parser.add_argument('--data_dir', type=str, default='./LumoImg/lesions')
    parser.add_argument('--file_list', type=str, default='foliage.txt')
    parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
    parser.add_argument('--future_frame', type=bool, default=False, help="use future frame")
    parser.add_argument('--nFrames', type=int, default=7)
    parser.add_argument('--model_type', type=str, default='RBPN')
    parser.add_argument('--residual', type=bool, default=False)
    parser.add_argument('--output', default='Results/RBPN', help='Location to save checkpoint models')
    parser.add_argument('--model', default='weights/RBPN_4x.pth', help='sr pretrained base model')

    opt = parser.parse_args()
    out = opt.output
    model = prepare_model(opt)

    directory = 'LumoImg/lesions'
    txt_files = glob.glob(directory+'/*.txt')
    txt_files.sort()

    for txt_file in txt_files:
        file = os.path.split(txt_file)[-1]
        data = prepare_data(opt, file)
        opt.output = os.path.join(out, file.split('.')[0])
        eval(model, data, opt)

if __name__ == "__main__":
    main()