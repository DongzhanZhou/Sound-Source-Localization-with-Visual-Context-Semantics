import os
import sys
import yaml
import easydict
import argparse
import torch
import numpy as np
import json

import utils
import dataset
import model

def train(epoch, network, optimizer, loader, logger, args):
    logger_meters = {}
    for key in args.loss_names:
        logger_meters[key] = utils.AverageMeter(key)
    logger_meters['acc'] = utils.AverageMeter('acc')
    network.train()

    for i, data in enumerate(loader):
        image, spec, name = data
        optimizer.zero_grad()
        loss_dict = network(image.cuda(), spec.cuda())
        loss = loss_dict['loc'].mean()
        for key in args.loss_names[1:]: # the first name is loc by default
            loss += getattr(args, '%s_weight' %(key)) * loss_dict[key].mean()

        loss.backward()
        optimizer.step()

        for key in args.loss_names:
            logger_meters[key].update(loss_dict[key].mean().item())

        _, top5 = loss_dict['logits'].topk(5, 1, True, True)
        target = torch.arange(top5.shape[0]).view(-1, 1).to(top5.device)
        acc = (top5 == target).sum()
        logger_meters['acc'].update(acc)

        if i % args.display_iter == 0:
            info = 'Epoch [{}][{}/{}]: loss {:.4f}, '.format(epoch, i, len(loader), loss.item())
            for key in args.loss_names:
                info += '{} {:.4f}, '.format(key, loss_dict[key].mean())
            info += 'Acc {:.3f}'.format(acc)
            logger.info(info)

    info = 'Epoch [{}]: '.format(epoch)
    for key in args.loss_names + ['acc']:
        info += '{} {:.4f}, '.format(key, logger_meters[key].avg)
    logger.info(info)

def main():
    args = easydict.EasyDict(yaml.safe_load(open("configs/train.yaml"))['common'])

    logfile = utils.record_handler(args)
    logger = utils.get_logger(__name__, logfile)
    train_data = dataset.LmdbDataset(args, logger)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True,num_workers=4,drop_last=True)
    network = model.LocModel(args)
    network.cuda()
    network = torch.nn.DataParallel(network)
    if len(args.pretrained_path):
        network = utils.load_pretrained(network, args.pretrained_path, logger)

    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    for epoch in range(args.num_epoch):
        train(epoch, network, optimizer, train_loader, logger, args)
        torch.save(network.state_dict(), os.path.join(logfile.replace('log.txt', 'model.pth')))
    logger.info("Finish training")

if __name__ == '__main__':
    main()