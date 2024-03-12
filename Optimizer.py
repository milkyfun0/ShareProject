"""
@File  :Optimizer.py
@Author:CaoQiXuan
@Date  :23/12/3 16:15
@Desc  :
"""

import torch.optim
from torchvision.models import resnet34


class Optimizer:
    def __init__(self, opt, model, writer=None):
        self.writer = writer
        self.params = filter(lambda p: p.requires_grad, model.parameters())

        print("---------requires_grad=False---------")
        for param in filter(lambda p: not p[1].requires_grad, model.named_parameters()):
            print(param[0])
        print("---------requires_grad=False---------")

        self.opt = opt
        self.optimizer = torch.optim.Adam(
            params=self.params,
            lr=opt["lr"],
            betas=eval(opt["betas"]),
            eps=eval(opt["eps"])
        )
        self.optimizer.zero_grad()
        if self.opt["lr_scheduler_type"] == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=opt["step"]["decay_epoch"],
                gamma=opt["step"]["decay_param"],
            )
        elif self.opt["lr_scheduler_type"] == "cos":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0=opt["cos"]["T_0"],
                T_mult=opt["cos"]["T_mult"],
                eta_min=opt["cos"]["eta_min"]
            )
        else:
            self.lr_scheduler = None

        self.norm_type = opt["norm_type"]
        self.max_grad_clip = opt["max_grad_clip"]
        self.stepNum = 0

    def step(self, epoch: int = None):
        if self.max_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.max_grad_clip, norm_type=self.norm_type)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            if self.opt["lr_scheduler_type"] == "step" and self.opt["step"]["restart_epoch"] != -1 and epoch % \
                    self.opt["step"]["restart_epoch"] == 0 and epoch != 0:
                self.lr_scheduler.__init__(optimizer=self.optimizer,
                                           step_size=self.opt["step"]["decay_epoch"],
                                           gamma=self.opt["step"]["decay_param"])

        self.optimizer.zero_grad()
        if self.writer is not None:
            self.writer.add_scalar("Optimizer", scalar_value=self.optimizer.state_dict()['param_groups'][0]['lr'],
                                   global_step=self.stepNum)
        self.stepNum += 1


if __name__ == "__main__":
    from main import get_options

    resnet = resnet34(pretrained=True)
    Optimizer(get_options()["optimizer"], resnet).step(10)
