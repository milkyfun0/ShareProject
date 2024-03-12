"""
@File  :main.py
@Author:CaoQiXuan
@Date  :23/12/314:52
@Desc  :
"""
import time

from torch.utils.tensorboard import SummaryWriter

from models.base_clip.Base_Clip import Network
from train import train
from utils import get_options, same_seeds

"""
最后一层全连接层对于transformer的影响？

r1i:8.13953488372093 r5i:22.093023255813954 r10i:30.46511627906977 medri:25.0 meanri:103.60697674418604
 r1t:5.906976744186046 r5t:26.790697674418606 r10t:43.95348837209303 medrt:13.0 meanrt:45.34046511627907
 sum:137.34883720930233
"""

if __name__ == "__main__":
    """
    tensorboard --logdir=logs/base_clip/20231204-2015
    """
    same_seeds(114514)
    opt = get_options()
    writer = SummaryWriter(log_dir=opt["logs"]["store_path"] + "test/", comment="")

    model = Network(opt=opt, writer=writer).cuda()
    train(opt=opt, model=model, writer=writer)
