import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def adjust_learning_rate(optimizer, orig_lr, batch_size, n_triplets, epochs):
    out = 0
    for group in optimizer.param_groups:
        if "no_grad" in group.keys():
            continue
        if "step" not in group:
            group["step"] = 0.0
        else:
            group["step"] += 1.0
        group["lr"] = orig_lr * (1.0 - float(group["step"]) * float(batch_size) / (n_triplets * float(epochs)))
        out = group["lr"]
    return out


def create_optimizer(model, new_lr, optimizer_name, wd):
    # setup optimizer
    if optimizer_name == "sgd": 
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=new_lr, 
                              momentum=0.9,
                              dampening=0.9,
                              weight_decay=wd)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=new_lr, weight_decay=wd)
    else:
        raise Exception("Not supported optimizer: {0}".format(optimizer_name))
    return optimizer
