import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
import wandb
from .data_loader import SentenceRELoader
from .utils import AverageMeter
from transformers import AdamW



class SentenceRE(nn.Module):

    def __init__(self, 
                 train_txt_file,
                 val_txt_file,
                 test_txt_file, 
                 train_img_dir,
                 val_img_dir,
                 test_img_dir,
                 model,
                 ckpt, 
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 warmup_step=300,
                 opt='sgd'):
        self.device = 0
        super().__init__()
        self.max_epoch = max_epoch
        # Load data 数据加载
        #  组成 文本路径 图片路径 编号的关系
        if train_txt_file != None:            
            self.train_loader = SentenceRELoader(
                train_txt_file,
                train_img_dir,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True
            )

        if val_txt_file != None:
            self.val_loader = SentenceRELoader(
                val_txt_file,
                val_img_dir,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False)

        if test_txt_file != None:
            self.test_loader = SentenceRELoader(
                test_txt_file,
                test_img_dir,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )
        # Model
        self.model = model
        # self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw': # Optimizer for BERT
            
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda(self.device)         # 直接改成1
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='acc'):
        best_metric = 0
        global_step = 0
        for epoch in range(self.max_epoch):
            self.train()
            logging.info("=== Epoch %d train ===" % epoch)
            
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)-1):
                        try:
                            data[i] = data[i].cuda(self.device)    # mark
                        except:
                            pass
                label = data[0]  # 长度64
                args = data[1:]  # 4 64张3*600*800图片  0 tokens2id64*128 1mask64*128 2，3都是64*1位置编码 
                # logits = self.parallel_model(*args)
                logits = self.model(*args)  # 64 23
                loss = self.criterion(logits, label)  # 交叉熵
                score, pred = logits.max(-1) # (B) 最大值
                acc = float((pred == label).long().sum()) / label.size(0)  # 查看预测正确的数目
                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                #wandb.log({"avg_loss":avg_loss.avg,"avg_acc":avg_acc.avg})
                # Optimize
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
            # Val 
            logging.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader) 
           # wandb.log({"eval_acc":result["acc"],"micro_p":result["micro_p"],"micro_r":result["micro_r"],"micro_f1":result["micro_f1"]})
            logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                logging.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]
        logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)-1):
                        try:
                            data[i] = data[i].cuda(self.device)
                        except:
                            pass
                label = data[0]
                args = data[1:]        
                # logits = self.parallel_model(*args)
                logits = self.model(*args)
                score, pred = logits.max(-1) # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                t.set_postfix(acc=avg_acc.avg)
        result = eval_loader.dataset.eval(pred_result)
        return result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
