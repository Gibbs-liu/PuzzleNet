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
#  import model5_b as import_model  # 论文用的模型
import model8 as import_model
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
    parser.add_argument('--dataset', default='fr', type=str, help='dataset type, fr or cad or bs')
    parser.add_argument('--look', default=False, action='store_true', help='look or not')
    parser.add_argument('--random', default=False, action='store_true', help='use random dataset')
    parser.add_argument('--use_emd2', action='store_true', help='look or not', default=False)
    parser.add_argument('--use_cd2', action='store_true', help='look or not', default=False)
    parser.add_argument('--use_emd3', action='store_true', help='use emd as loss for fpcb', default=False)
    parser.add_argument('--random_slice', action='store_true', help='look or not', default=False)
    parser.add_argument('--verbose', action='store_true', help='use verbose', default=False)



    args = parser.parse_args(argv)
    return args

def per_parse(opt):
    if opt.output_path == 'TRG':
        cpath = os.path.join(os.path.dirname(__file__), 'TRG')
        if not os.path.exists(cpath):
            os.makedirs(cpath)
        dpath = os.path.join(cpath, datetime.datetime.now().strftime('%b%d_%H-%M-%S'))
        opt.output_path = dpath
    opt.swap_axis = True



def main(opt):
    # save setting
    argsDic = opt.__dict__
    dir = os.path.join(opt.output_path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(os.path.join(dir, 'setting.txt'), 'w+') as f:
        f.writelines('-----------starting---------------\n')
        f.writelines(__file__+ '\n')
        f.writelines(time.asctime(time.localtime(time.time()))+'\n')
        for eachArg, value in argsDic.items():
            f.writelines(eachArg+"   "*8+str(value)+'\n'  )
        f.writelines('---------------ending-------------\n')

    print('************************************')
    print('     ', opt.output_path)
    print('************************************')


    # get Model
    model = import_model.TouchedRegraster(opt)
    with open(os.path.join(dir, 'model.txt'), 'w+') as ofile2:
        ofile2.writelines(import_model.__name__+'\n')
    
    traindataset, valdataset, testdataset = dataset.get_dataset(opt.dataset, random=opt.random, random_slice=opt.random_slice)
    num_workers = torch.cuda.device_count()*4
    num_workers = 64
    #  num_workers = 4
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=opt.batch_size, drop_last=True, shuffle=True,
            num_workers=num_workers)
    valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=opt.batch_size, drop_last=True, shuffle=False,
            num_workers=num_workers)
    # TODO: batch_size 应该是1去做
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, drop_last=False, shuffle=False,
            num_workers=num_workers)
    print(len(valdataset))
    print(len(traindataset))


    # TODO: trainer 的配置需要增加一些
    #  trainer = pl.Trainer(accelerator='gpu',
            #  devices=[opt.device],
            #  default_root_dir=opt.output_path,
            #  max_steps=opt.epochs)
    checkpoint_callback = ModelCheckpoint(
            dirpath=opt.output_path,
            filename='''{epoch:02d}-loss{train_loss:.8f}''',
            every_n_epochs=opt.save_on_epochs,save_top_k=2,save_on_train_epoch_end=True,
            monitor='train_loss'
            )
    early_stop = early_stopping.EarlyStopping(monitor='epoch_loss', mode='min', stopping_threshold=7.0, check_on_train_epoch_end=True, verbose=opt.verbose, patience=99999)
    trainer = pl.Trainer(accelerator="gpu", devices=[opt.device],
            default_root_dir=opt.output_path,
            max_epochs=opt.epochs,
            #  callbacks=[checkpoint_callback, early_stop],
            callbacks=[checkpoint_callback],
            #  log_every_n_steps=3,
            check_val_every_n_epoch=10,
            #  num_sanity_val_steps=0,
            )
    trainer.fit(model, traindataloader, val_dataloaders=valdataloader)
    trainer.test(model, dataloaders=testdataloader)


if __name__ == '__main__':
    opt = parse_arg()
    per_parse(opt)
    main(opt)

    # 测一下出来的形状的对不对
    #  m=PCTransformer(opt)
    #  a = torch.randn(64,1024,3)
    #  f = m(a)
    #  print(f.shape)

