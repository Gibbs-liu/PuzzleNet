
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import open3d as o3d
import numpy as np
import dataset
import datetime
import argparse
import os, sys
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping


#  import model6_boundary as import_model
#  import model6 as import_model
import model5_b as import_model
#  import model3_c3 as import_model
#  import model3_e2 as import_model


def parse_arg(argv=None):
    parser = argparse.ArgumentParser('touched regis')

    # required args
    parser.add_argument('--output_path', help='path to store output_folders', type=str, 
            default='TRG')
    parser.add_argument('--device', help='choose device to use', type=int, default=2)
    parser.add_argument('--dim-k', default=1024, type=int, help='length of feature')
    parser.add_argument('--epochs', default=100000, type=int, help='epochs')
    parser.add_argument('--max_iter', default=50, type=int, help='max-iter on IC algorithm')
    parser.add_argument('--dt', help='data type of input, default=fr', type=str, default='fr')
    parser.add_argument('--mode', help='mode for now', default='train')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--save_on_epochs', default=1, type=int, help='save on how many epochs')
    parser.add_argument('--loss_mode', default=0, type=int, help='mode of loss using')
    parser.add_argument('--valp', default=0, type=int, help='how often check valuation per epochs')
    parser.add_argument('-m', help='leave a message for this round of training/valuation', type=str, 
            default='default string')
    parser.add_argument('--loss_sum', default=False, action='store_true', 
            help="是使用sum还是mean计算loss，sum default")
    parser.add_argument('--pretrain_epochs', default=700, type=int)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--dataset', default='cadr', type=str, help='dataset type, fr or cad or bs')
    parser.add_argument('--look', default=False, action='store_true', help='look or not')
    parser.add_argument('--random', default=False, action='store_true', help='use random dataset')
    parser.add_argument('--use_emd2', action='store_true', help='look or not', default=False)
    parser.add_argument('--use_cd2', action='store_true', help='look or not', default=False)
    parser.add_argument('--use_emd3', action='store_true', help='use emd as loss for fpcb', default=False)
    parser.add_argument('--random_slice', action='store_true', help='look or not', default=False)
    parser.add_argument('--verbose', action='store_true', help='use verbose', default=False)
    args = parser.parse_args(argv)
    return args


def main(opt):
    # save setting
    argsDic = opt.__dict__
    dir = os.path.join(opt.output_path)
    opt.output_path = os.path.join('/home/code/transReg/val/test_out/')
    model_path = os.path.join('/home/code/transReg/val/models_5')


    # get Model
    # model 的超参能不能拿出来用？
    #  model = import_model.TouchedRegraster.load_from_checkpoint(os.path.join(model_path, 'cadr.ckpt'))
    #  model = import_model.TouchedRegraster.load_from_checkpoint(os.path.join(model_path, 'cone.ckpt'))
    #  model = import_model.TouchedRegraster.load_from_checkpoint(os.path.join(model_path, 'cyl.ckpt'))
    #  model = import_model.TouchedRegraster.load_from_checkpoint(os.path.join(model_path, 'sphere.ckpt'))
    #  model = import_model.TouchedRegraster.load_from_checkpoint(os.path.join(model_path, 'fr5.ckpt'))

    #  model = import_model.TouchedRegraster.load_from_checkpoint(os.path.join(model_path, 'trans.ckpt'))

    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Aug20_22-48-38/epoch=16730-losstrain_loss=0.45130691.ckpt') # r 1
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Sep29_12-47-06/epoch=8801-losstrain_loss=0.58531296.ckpt')
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Sep29_14-08-15/epoch=14389-losstrain_loss=0.34025502.ckpt')

    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Aug23_14-18-02/epoch=14842-losstrain_loss=2.67890167.ckpt')
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Aug23_14-18-02/epoch=13842-losstrain_loss=2.57903862.ckpt')
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Sep26_02-25-56/epoch=9265-losstrain_loss=4.65855312.ckpt')
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Sep26_02-14-35/epoch=17703-losstrain_loss=3.33939028.ckpt')
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Sep06_19-52-59/epoch=28545-losstrain_loss=2.55200005.ckpt')
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Sep29_14-57-52/epoch=647-losstrain_loss=0.00951214.ckpt')
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Sep29_14-54-28/epoch=1123-losstrain_loss=14.95236874.ckpt')
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Sep30_16-40-57/epoch=9269-losstrain_loss=0.88293684.ckpt') # 2
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct08_13-08-07/epoch=1541-losstrain_loss=0.36748514.ckpt') # 0
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct08_16-05-39/epoch=1173-losstrain_loss=11.86771774.ckpt')
#     model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct08_16-05-39/epoch=1173-losstrain_loss=11.86771774.ckpt')


    # For FR
#     model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct14_16-21-48/epoch=24294-losstrain_loss=0.61680794.ckpt') # 1
#     model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct14_16-21-48/epoch=24234-losstrain_loss=0.62542528.ckpt') # 1
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct14_16-19-11/epoch=21878-losstrain_loss=0.41205645.ckpt') # 0
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct14_16-19-57/epoch=19751-losstrain_loss=0.67370081.ckpt') # 2
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct14_16-20-29/epoch=18731-losstrain_loss=0.39350516.ckpt') # 5
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct14_16-20-29/epoch=18731-losstrain_loss=0.39350516.ckpt') # 6

    # For cadr
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct14_16-08-36/epoch=5111-losstrain_loss=3.38435793.ckpt') # 1
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct14_16-08-36/epoch=5104-losstrain_loss=3.45459270.ckpt') # 1
    #  model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct14_16-08-36/epoch=5881-losstrain_loss=2.97773314.ckpt') # 1
    model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct14_16-08-36/epoch=7070-losstrain_loss=1.56574667.ckpt') # 1
    model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Oct14_16-08-36/epoch=7728-losstrain_loss=1.64379442.ckpt') # 1
    model = import_model.TouchedRegraster.load_from_checkpoint('/home/code/transReg/TRG/Dec26_14-46-41/epoch=11579-losstrain_loss=0.72871453.ckpt') # 1
    
    



    print(model.C.dataset)
    print(model.C.loss_mode)
    
    trainset, _, testdataset = dataset.get_dataset(model.C.dataset, random=False, random_slice=False)
    num_workers = torch.cuda.device_count()*4
    traindataloder = torch.utils.data.DataLoader(trainset, batch_size=1, drop_last=False, shuffle=False,
            num_workers=num_workers)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, drop_last=False, shuffle=False,
            num_workers=num_workers)


    # TODO: trainer 的配置需要增加一些
    trainer = pl.Trainer(accelerator="gpu", devices=[opt.device],
            default_root_dir=opt.output_path,
            max_epochs=opt.epochs,
            #  callbacks=[checkpoint_callback, early_stop],
            #
            log_every_n_steps=3,
            check_val_every_n_epoch=10,
            )
    trainer.test(model, dataloaders=testdataloader)


if __name__ == '__main__':
    opt = parse_arg()
    main(opt)


