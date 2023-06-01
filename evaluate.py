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

def evaluate(network, loader, logger, args):
    evaluator = utils.Evaluator(logger)
    network.eval()
    results = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            image, spec, name = data
            heatmap, _ = network(image.cuda(), spec.cuda())
            heatmap = heatmap.detach().cpu().numpy()
            for b_ix in range(len(name)):
                gt = utils.testset_gt(args, name[b_ix])
                results.append({'av': heatmap[b_ix][0], 'gt':gt, 'name': name[b_ix]})
    auc = evaluator.eval(results, 'av')
    del results
    return auc

def main():
    args = easydict.EasyDict(yaml.safe_load(open("configs/test.yaml"))['common'])
    logfile = utils.record_handler(args)
    logger = utils.get_logger(__name__, logfile)
    # load model
    network = model.LocModel(args)
    network.cuda()
    network = torch.nn.DataParallel(network)
    network = utils.load_pretrained(network, args.pretrained_path, logger)

    if args.testset == 'flickr':
        val_data = dataset.FlickrTestDataset()
    else: # vggss
        val_data = dataset.LmdbDataset(args, logger, mode='eval')

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                             shuffle=False,num_workers=4,drop_last=False)
    
    evaluate(network, val_loader, logger, args)

if __name__ == '__main__':
    main()