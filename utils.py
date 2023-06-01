import csv
import os
import torch
import cv2
import shutil
import datetime
import logging
import sklearn.metrics
import numpy as np
import json
import xml.etree.ElementTree as ET

ospj = os.path.join
ospe = os.path.exists

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def load_pretrained(network, path, logger):
    ckpt = torch.load(path)
    if 'model_state_dict' in ckpt.keys():
        ckpt = ckpt['model_state_dict']
    model_keys = set(network.state_dict().keys())
    ckpt_keys = set(ckpt.keys())
    common_keys = model_keys & ckpt_keys
    diff_keys = model_keys - ckpt_keys
    network.load_state_dict(ckpt, strict=False)
    logger.info("Loaded pretrained models, {} keys in common, {} keys missing.".format(len(common_keys), len(diff_keys)))
    if len(diff_keys):
        logger.info("Missing keys: {}".format(diff_keys))
    return network

def get_logger(name,logfile=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler_console = logging.StreamHandler()
    handler_console.setLevel(logging.DEBUG)
    handler_console.setFormatter(formatter)
    logger.addHandler(handler_console)

    if logfile:
        handler_file = logging.FileHandler(logfile)
        handler_file.setLevel(logging.DEBUG)
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    return logger

def record_handler(args):
    TIMEFORMAT = '_%Y%m%d_%H:%M:%S,%f'
    theTime = datetime.datetime.now().strftime(TIMEFORMAT)
    fold_name = args.exp_name + theTime
    fold_name = ospj('experiments', args.exp_prefix, fold_name[:-3])
    if not ospe(fold_name):os.makedirs(fold_name)
    code_folder = ospj(fold_name, 'codes')
    if not ospe(code_folder):os.makedirs(code_folder)
    for file in os.listdir('.'):
        if file.endswith('.py'):
            shutil.copyfile(file, ospj(code_folder, file))
    with open(ospj(fold_name,'args.json'),'w') as f:
        f.write(json.dumps(args.__dict__))
    logfile = ospj(fold_name, 'log.txt')
    return logfile

class Evaluator():
    def __init__(self, logger=None):
        super(Evaluator, self).__init__()
        self.logger = logger
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres):
        infer_map = np.zeros((224, 224))
        infer_map[infer>=thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))
        #return ciou, np.sum(infer_map*gtmap),(np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))
        return ciou

    def cal_AUC(self, ciou):
        results = []
        for i in range(21):
            result = np.sum(np.array(ciou)>=0.05*i)
            result = result / len(ciou)
            results.append(result)
        x = [0.05*i for i in range(21)]
        auc = sklearn.metrics.auc(x, results)
        ciou_acc = np.sum(np.array(ciou) >= 0.5)/len(ciou)
        return auc, ciou_acc

    def final(self):
        ciou = np.mean(np.array(self.ciou)>=0.5)
        return ciou

    def clear(self):
        self.ciou = []
    
    def eval(self, outputs, eval_key):
        for item in outputs:
            pred = item[eval_key]
            gtmap = item['gt']
            heatmap = cv2.resize(pred, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            vggss_thres = np.median(heatmap.flatten())
            self.ciou.append(self.cal_CIOU(heatmap, gtmap, vggss_thres))
        
        auc, ciou_acc = self.cal_AUC(self.ciou)
        try:
            self.logger.info("AUC: {:.3f}, IoU@0.5: {:.3f}".format(auc, ciou_acc))
        except:
            print("AUC: {:.3f}, IoU@0.5: {:.3f}".format(auc, ciou_acc))
        return auc


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value

def testset_gt(args,name):

    if args.testset == 'flickr':
        gt = ET.parse(args.gt_path + '%s.xml' % name).getroot()
        gt_map = np.zeros([224,224])
        bboxs = []
        for child in gt: 
            for childs in child:
                bbox = []
                if childs.tag == 'bbox':
                    for index,ch in enumerate(childs):
                        if index == 0:
                            continue
                        bbox.append(int(224 * int(ch.text)/256))
                bboxs.append(bbox)
        for item_ in bboxs:
            temp = np.zeros([224,224])
            (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
            temp[item_[1]:item_[3],item_[0]:item_[2]] = 1
            gt_map += temp
        gt_map /= 2
        gt_map[gt_map>1] = 1
        
    elif args.testset == 'vggss':
        gt = args.gt_all[name[:-4]]
        gt_map = np.zeros([224,224])
        for item_ in gt:
            item_ =  list(map(lambda x: int(224* max(x,0)), item_) )
            temp = np.zeros([224,224])
            (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
            temp[ymin:ymax,xmin:xmax] = 1
            gt_map += temp
        gt_map[gt_map>0] = 1
    return gt_map

