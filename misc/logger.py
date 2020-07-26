import csv
import os
import shutil
import numpy as np
from queue import PriorityQueue
from tqdm import tqdm
import torch


class CsvLogger:
    def __init__(self, filepath='./', filename='validate_record.csv', data=None, fieldsnames=['epoch', 'train_loss', 'val_loss', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']):
        self.log_path = filepath
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        if filename:
            self.log_name = filename
            self.csv_path = os.path.join(self.log_path, self.log_name)
            self.fieldsnames = fieldsnames

            if not os.path.exists(self.csv_path):
                with open(self.csv_path, 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
                    writer.writeheader()

            self.data = {}
            for field in self.fieldsnames:
                self.data[field] = []
            if data is not None:
                for d in data:
                    d_num = {}
                    for key in d:
                        d_num[key] = float(d[key]) if key != 'epoch' else int(d[key])
                    self.write(d_num)

    def write(self, data):
        for k in self.data:
            self.data[k].append(data[k])
        with open(self.csv_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writerow(data)

    def write_text(self, text, print_t=True):
        with open(os.path.join(self.log_path, 'log.txt'), 'a') as f:
            f.write('{}\n'.format(text))
        if print_t:
            tqdm.write(text)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, multiply=True):
        self.val = val
        if multiply:
            self.sum += val * n
        else:
            self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class ModelNode(object):
    def __init__(self, res, model_path, key='Sum'):
        self.res = res
        self.model_path = model_path
        self.key = key

    def __lt__(self, other): 
        return self.res[self.key] < other.res[self.key]   

class k_PriorityQueue(object):
    def __init__(self, k_best_model, folder_path, standard=['METEOR', 'CIDEr'], init_res={}):
        self.k_best_model = k_best_model
        self.queue = PriorityQueue()
        self.folder_path = folder_path

        self.continuous_failed_count = 0
        self.key = 'Sum'
        if len(init_res):
            assert len(init_res) == 5
            assert self.key in init_res.keys()
            self.best_res = init_res
            self.queue.put(ModelNode(init_res, ''))
        else:
            self.best_res = {self.key: 0, 'Bleu_4':0, 'METEOR':0, 'ROUGE_L':0, 'CIDEr':0} # rethe best overall score
        self.best_ = {k: self.best_res[k] for k in standard}


    def score(self, res):
        out = 0
        # update the best score for each metric
        for k in self.best_.keys():
            if res[k] > self.best_[k]:
                self.best_[k] = res[k]
            # calculate the relative score for each metric
            out += res[k] / self.best_[k]

        # calculate overall score
        res[self.key] = out / len(self.best_.keys())
        
    def update(self, res):
        self.score(res)
        self.score(self.best_res)
        
        new_queue = PriorityQueue()
        while self.queue.qsize() > 0:
            node = self.queue.get()
            self.score(node.res)
            new_queue.put(node)
        
        self.queue = new_queue
        
    def check(self, res, model_path, model_name, opt):
        # save at most k models
        self.update(res)

        if self.queue.qsize() == self.k_best_model:
            node = self.queue.get()

            if res['Sum'] > node.res['Sum']:
                self.continuous_failed_count = 0
                self.queue.put(ModelNode(res, model_path))
                if model_path:
                    # move current checkpoint to the folder_path
                    shutil.copy(
                        os.path.join(opt["checkpoint_path"], 'checkpoint.pth.tar'), 
                        os.path.join(self.folder_path, model_name)
                        )
                if node.model_path:
                    # remove the previous checkpoint to save disk space
                    os.remove(os.path.join(self.folder_path, 'model_%04d.pth.tar' % node.res['epoch']))
            else:
                self.queue.put(node)
                self.continuous_failed_count += 1
                if self.continuous_failed_count >= opt['tolerence']:
                    #logger.write_text("Have reached maximun tolerence {}!".format(opt['tolerence']))
                    return False, self.continuous_failed_count
        else:
            self.queue.put(ModelNode(res, model_path))
            if model_path:
                shutil.copy(
                    os.path.join(opt["checkpoint_path"], 'checkpoint.pth.tar'), 
                    os.path.join(self.folder_path, model_name)
                    )

        info = "{:2d}, {:6.2f} {} {:6.2f}\tB {:5.2f}({:5.2f})\tM {:5.2f}({:5.2f})\tR {:5.2f}({:5.2f})\tC {:5.2f}({:5.2f})".format(
                        self.continuous_failed_count, 100 * res['Sum'], res['Sum'] > self.best_res['Sum'], 100 * self.best_res['Sum'],
                        100 * res["Bleu_4"], 100 * (res["Bleu_4"] - self.best_res["Bleu_4"]),
                        100 * res["METEOR"], 100 * (res["METEOR"] - self.best_res["METEOR"]),
                        100 * res["ROUGE_L"], 100 * (res["ROUGE_L"] - self.best_res["ROUGE_L"]),
                        100 * res["CIDEr"], 100 * (res["CIDEr"] - self.best_res["CIDEr"])
                        )
        if res['Sum'] > self.best_res['Sum']:
            self.best_res = res

        return True, info

    def load(self):
        # load the saved results in the folder_path and then evaluate them
        file_list = os.listdir(self.folder_path)
        for file in file_list:
            pth = os.path.join(self.folder_path, file)
            res = torch.load(pth)['validate_result']
            self.queue.put(ModelNode(res, pth))

    def qsize(self):
        return self.queue.qsize()

    def get(self):
        return self.queue.get()

    def load_the_best_checkpoint(self, model):
        assert self.qsize() == 1
        node = self.get()
        model.load_state_dict(torch.load(node.model_path)['state_dict'])
        self.queue.put(node)