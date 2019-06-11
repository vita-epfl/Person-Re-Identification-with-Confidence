import train_imgreid_xent_smoothing as xent_script
import sys
import argparse
from torchreid import models
import os.path as osp
import itertools
from torchreid.utils.logger import Logger
import pdb
from torchreid import data_manager
import math
import numpy as np
from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser(description='HyperParameter Search')
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
parser.add_argument('--root', type=str, default='/data/george-data/Dataset',
                    help="root path to data directory")
parser.add_argument('--gpu-devices', default='0', type=str,
help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())

parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")

parser.add_argument('--suffix', default='', type=str,
                    help='suffix to add to directory')

parser.add_argument('--multiplier', default=1, type=int,
                    help='number of learning rate in multiple of 5')

parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")

parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")

parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="use cuhk03-metric (default: False)")
parser.add_argument('--use-smoothing', action='store_true',
                    help="use label smoothing (default: False)")
def main(args):
    src = '/data/george-data/survey/hyperLearningRate/'
    args = parser.parse_args(args)
    root = ["--root", args.root]
    arch_choice = ["-a", args.arch]
    data_choice = ["-d", args.dataset]
    optim_choice = ["--optim", args.optim]
    epoch_choice = ["--max-epoch", "100"]
    train_batch = ["--train-batch", str(args.train_batch)]
    test_batch = ["--test-batch", "100"]
    height_width = ["--height", "256", "--width", "128"]
    label_smooth = []
    if args.use_smoothing:
        label_smooth =  ["--label-smooth"]
    stepSize_choice = ["--stepsize", "20", "40"]
    gpu_choice = ["--gpu-devices", args.gpu_devices]
    weight_loading = ["--load-weights",args.load_weights]
    eval_steps = ["--eval-step", "10"]
    enable_scheduler = ["--scheduler","1"]
    fixBase = ["--fixbase-epoch","6" ,"--fixbase-lr","0.0003"]
    multiplier = args.multiplier
    cuhk03_choice = []
    cuhk03_name = ""
    if args.cuhk03_labeled:
        cuhk03_choice.append('--cuhk03-labeled')
        if cuhk03_name != "":
            cuhk03_name = "_".join([cuhk03_name,"cuhk03Labeled"])
        else:
            cuhk03_name = cuhk03_name + "cuhk03Labeled"
    if args.cuhk03_classic_split:
        cuhk03_choice.append('--cuhk03-classic-split')
        if cuhk03_name != "":
            cuhk03_name = "_".join([cuhk03_name,"cuhk03ClassicSplit"])
        else:
            cuhk03_name = cuhk03_name + "cuhk03ClassicSplit"
    if args.use_metric_cuhk03:
        cuhk03_choice.append('--use-metric-cuhk03')
        if cuhk03_name != "":
            cuhk03_name = "_".join([cuhk03_name,"cuhk03Metric"])
        else:
            cuhk03_name = cuhk03_name + "cuhk03Metric"
    #lr_search = ["0.0002","0.0003","0.00035","0.0004","0.0005","0.0006","0.0007","0.001"]

    aa = math.log(0.0003, 10)
    bb = math.log(0.001, 10)
    log_lr_list = np.random.uniform(aa, bb, int(5*multiplier)).tolist()
    lr_search = [10**xx for xx in log_lr_list]
    # Beta
    xent_list = np.random.uniform(1, 7, int(5)).tolist()*multiplier

    saved_folders = []
    final_results = []

    writer_dict={}
    writer = SummaryWriter(log_dir=osp.join(src,'Accuracy_vs_LR_xent',"_".join([args.arch,data_choice[1],optim_choice[1], "e"+epoch_choice[1], "b"+train_batch[1],"{}lr".format(5*multiplier),cuhk03_name,args.suffix]), 'tensorboard'))

    dict_acc = {}
    best_lr_acc = 0
    best_lr = -1
    best_xent = -1
    best_lr_acc = -1
    best_arg = None
    print("Learning Rates to train: {}".format(lr_search))
    for idx,lr_elem in enumerate(lr_search):
        try:
            sys.stdout = sys.__stdout__
            arg_list = []
            lr_choice = ["--learning-rate", str(lr_elem)]

            xent = xent_list[idx]
            xent_choice = ["--lambda-xent", str(xent)]
            folder_save = osp.join(src,'log_resnet50', "_".join([args.arch,data_choice[1],optim_choice[1], "e"+epoch_choice[1], "b"+train_batch[1],"lr"+lr_choice[1], "xentLambda"+str(xent),'NotPretrained',cuhk03_name,args.suffix]))

            save_dir = ["--save-dir",folder_save]
            saved_folders.append(folder_save)
            arg_list.extend(root+ arch_choice+data_choice+optim_choice+epoch_choice+train_batch+test_batch+height_width+stepSize_choice+gpu_choice+eval_steps+enable_scheduler+lr_choice+label_smooth+save_dir+fixBase+xent_choice+weight_loading+cuhk03_choice)

            if "siamese" in args.arch:
                best_rank, best_epoch = xent_siamese_script.main(arg_list)
            else:
                best_rank, best_epoch = xent_script.main(arg_list)
            dict_acc[lr_elem] = (xent,best_rank, best_epoch)
            writer.add_scalars(
            'Acc vs LR',
            dict(rank_1= best_rank,
                 epoch = best_epoch),
            int(lr_search.index(lr_elem)))

            writer.add_scalars(
            'LR_Xent',
            dict(learning_rate= lr_elem,
                 xent_lambda = xent),
            int(lr_search.index(lr_elem)))


            if best_lr_acc< best_rank:
                best_lr = lr_elem
                best_xent = xent
                best_lr_acc = best_rank
                best_arg = arg_list
            print(dict_acc)
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)
            pass
    print("#################################")
    print("Best Learning rate and xent lambda:")
    print("Learning Rate: {}".format(best_lr))
    print("Lambda Xent: {}".format(best_xent))
    print("Best Rank1: {}".format(best_lr_acc))
    print("Argument list: {}".format(best_arg))

if __name__ == '__main__':
    main(sys.argv[1:])
