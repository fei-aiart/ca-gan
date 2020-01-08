# -*- coding: utf-8 -*-
# @Author: JacobShi777

import argparse


def init():
    parser = argparse.ArgumentParser(description='PyTorch ')
    # parser.add_argument('--root', type=str, default='/data/jacob/photosketch', help='image source folder')
    parser.add_argument('--root', type=str, default='/home/jacob/dataset/photosketch', help='image source folder')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='checkpoint folder')
    parser.add_argument('--gen_root', type=str, default='./Gen_images', help='images generated to')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--input_nc', type=int, default=9, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--gpu_ids', default=[0], help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda?')
    parser.add_argument('--no_lsgan', action='store_true',
                        help='do *not* use least square GAN, if false, use vanilla GAN')
    parser.add_argument('--lambda1', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_g', type=float, default=5, help='qing')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--n_epoch', type=int, default=700, help='training epoch')
    parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
    parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
    # parser.add_argument('--infofile', default=['./data/list_train.txt', './data/list_test.txt'], help='infofile')
    parser.add_argument('--infofile', default=['./data/list_train_F.txt', './data/list_test_F.txt'], help='infofile')
    parser.add_argument('--batchSize', type=int, default=1, help='training batchSize')
    parser.add_argument('--test_period', type=int, default=100, help='test_period')
    parser.add_argument('--save_period', type=int, default=700, help='save_period')
    parser.add_argument('--myGpu', default='2', help='GPU Number')

    parser.add_argument('--alpha1', type=float, default=0.7, help='alpha for global L1 loss ')
    parser.add_argument('--rec_weig1', type=float, default=0.3, help='weights when applying recognition')
    parser.add_argument('--rec_weig2', type=float, default=1.7, help='weights when applying recognition')
    parser.add_argument('--rec_weig3', type=float, default=3.4, help='weights when applying recognition')
    parser.add_argument('--style_weight', type=float, default=60.0, help="weight for style-loss, default is 5.0")
    parser.add_argument("--content_weight", type=float, default=0.3, help="weight for content-loss, default is 1.0")
    parser.add_argument("--model_dir", type=str, default="models/",
                        help="directory for vgg, if model is not present in the directory it is downloaded")
    parser.add_argument('--model_vgg', type=str, default='../pre_trained_models/vgg.model', help='vgg model')
    parser.add_argument('--styleParam', type=float, default=5.0, help='')

    parser.add_argument('--lambda_vgg16', type=float, default=5.0, help='')
    parser.add_argument('--vgg16', type=str, default='../pre_trained_models/vgg.model', help='vgg model')
    parser.add_argument('--output', type=str, default="./output", help=' ')
    parser.add_argument('--test_epoch', type=str, default='700', help='test')
    opt = parser.parse_args()
    return opt
