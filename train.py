"""
@File  :train.py
@Author:CaoQiXuan
@Date  :23/12/4 19:57
@Desc  :
"""
import gc
import random
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

import utils
from DataSet import get_loader
from Optimizer import Optimizer
from models.base_clip.Base_Clip import Network
from models.base_cnn.Base_CNN import Base_CNN
from models.base_peg.Base_Peg import Base_Peg
from models.base_trans.Base_Trans import Base_Trans
from models.base_register.Base_Register import Base_Register
from utils import validate_with_no_cross, save_log_txt, mark_validate, same_seeds, get_options, convert_weights


def train(opt, model, train_loader, test_loader, writer):
    model.train()
    optimizer = Optimizer(opt=opt["optimizer"], model=model, writer=writer)
    # train_loader, test_loder = get_loader(opt)

    count = 0
    max_mr = 0
    print_i = len(train_loader) // opt["logs"]["print_freq"]
    best_metric_dict = None

    for epoch in range(opt['train']['epoch']):
        start_time = time.time()

        for data_pair, i in zip(train_loader, (range(len(train_loader)))):
            torch.cuda.empty_cache()
            gc.collect()
            loss = model(data_pair)["total loss"]
            loss.backward()
            optimizer.step(epoch=epoch)
            end_time = time.time()

            if i % print_i == 0:
                mark_log = time.strftime(
                    '[%Y-%m-%d %H:%M:%S]',
                    time.localtime()) + " Epoch: [{:0>3d}/{:0>3d}] [{:d}/{:d}] total loss: {:<5.2f} time: {:<5.2f}s\n".format(
                    epoch, opt["train"]["epoch"], i, len(train_loader), loss.cpu().item(), end_time - start_time)
                save_log_txt(writer, mark_log)
                writer.add_scalar("total_loss", loss.item(), count)
                count += 1
                start_time = time.time()
        if epoch % opt["logs"]["eval_step"] == 0:
            torch.cuda.empty_cache()
            metric_dict = validate_with_no_cross(test_loader, model, opt)
            if metric_dict["mR"] > max_mr:
                max_mr = metric_dict["mR"]
                best_metric_dict = metric_dict
                model.clip.state_dict(),
                print(opt["logs"]["analysis_path"] + opt["model"]["CLIP"]["save_name"])
                if opt["logs"]["save_state_dict"]:
                    torch.save(model.clip.state_dict(),
                               opt["logs"]["analysis_path"] + opt["model"]["CLIP"]["save_name"])
            mark_validate(opt, epoch // opt["logs"]["eval_step"], metric_dict, best_metric_dict, writer)


if __name__ == "__main__":
    """
    tensorboard --logdir=logs/base_clip/20231204-2015
    """
    same_seeds(114514)

    opt = get_options(model_name="base_trans")
    utils.generate_random_samples(opt)
    train_loader, test_loder = get_loader(opt)
    writer = SummaryWriter(log_dir=opt["logs"]["store_path"] + "test/", comment="")

    # model = Network(opt=opt, writer=writer).cuda()
    if opt["model"]["name"] == "base_clip":
        model = Network(opt=opt, writer=writer)
    elif opt["model"]["name"] == "base_cnn":
        model = Base_CNN(opt=opt, writer=writer)
    elif opt["model"]["name"] == "base_trans":
        model = Base_Trans(opt=opt, writer=writer)
    elif opt["model"]["name"] == "base_register":
        model = Base_Register(opt=opt, writer=writer)
    elif opt["model"]["name"] == "base_peg":
        model = Base_Peg(opt=opt, writer=writer)
    else:
        print("Unknown model")
        sys.exit()
    model.show()
    # convert_weights(model)
    model.cuda()
    print(model)
    train(opt=opt, model=model, train_loader=train_loader, test_loader=test_loder, writer=writer)
