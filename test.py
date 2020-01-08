import cv2
import torch
from torch.autograd import Variable
import numpy as np
import os
import torchvision.utils as vutils
from data import *
from model import *
import option
from torch.utils.data import DataLoader
from myutils.Unet2 import *
opt = option.init()
norm_layer = get_norm_layer(norm_type='batch')
netG = MyUnetGenerator(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, \
                           use_dropout=False, gpu_ids=opt.gpu_ids)

netE = MyEncoder(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, \
                 use_dropout=False, gpu_ids=opt.gpu_ids)
fold = opt.test_epoch
netG.load_state_dict(torch.load('./checkpoint/netG_epoch_'+fold+'.weight'))
netE.load_state_dict(torch.load('./checkpoint/netE_epoch_'+fold+'.weight'))

netE.cuda()
netG.cuda()


test_set = DatasetFromFolder(opt, False)

testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
# netG = UnetGenerator(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, use_dropout=False, gpu_ids=opt.gpu_ids)
if not os.path.exists(opt.output):
    os.makedirs(opt.output)

save_dir_A = opt.output + "/" + fold
if not os.path.exists(save_dir_A):
    os.makedirs(save_dir_A)
for i, batch in enumerate(testing_data_loader):
    real_p, real_s, identity = Variable(batch[0]), Variable(batch[1]), Variable(batch[2].squeeze(1))
    real_p, real_s, identity = real_p.cuda(), real_s.cuda(), identity.cuda()
    # parsing = real_p[:, 3:, :, :]
    # real_p, real_s = real_s, real_p[:, 0:3, :, :]
    # real_p = torch.cat([real_p, parsing], 1)

    parsing_feature = netE(real_p[:, 3:, :, :])
    fake_s1 = netG.forward(real_p[:, 0:3, :, :], parsing_feature)
    # fake_s1[:, 1, :, :],fake_s1[:, 2, :, :], fake_s1[:, 0, :, :] = fake_s1[:, 0, :, :], fake_s1[:, 1, :, :], fake_s1[:, 2, :, :]
    output_name_A = '{:s}/{:s}{:s}'.format(
        save_dir_A, str(i + 1), '.jpg')
    vutils.save_image(fake_s1[:, :, 3:253, 28:228], output_name_A, normalize=True, scale_each=True)
    # fake_s1 = fake_s1.squeeze(0)

    # fake_s1 = np.transpose(fake_s1.data.cpu().numpy(), (1, 2, 0)) / 2 + 0.5

    # img = fake_s1[3:253, 28:228, :]
    # cc = (img * 255).astype(np.uint8)
    # cv2.imwrite(output_name_A, cc)

print " saved"



