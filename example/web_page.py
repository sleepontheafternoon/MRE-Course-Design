# coding:utf-8
import torch
import numpy as np
import json
import os,sys
import os,  json

import torch
from torch import nn, optim

from transformers import AdamW

from transformers import BertModel, BertTokenizer
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import torch.nn as nn
from transformers import BeitFeatureExtractor, BeitModel
import argparse
import logging
import random
import wandb
import jpeg4py as jpeg4py
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    




class BERTEntityEncoder_MY(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768 * 2
        self.mask_entity = mask_entity
      
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.device = "cpu"

    def forward(self, token, att_mask, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
        Return:
            (B, 2H), representations for sentences
        """
        # outputs = self.bert(token, attention_mask=att_mask)
        outputs = self.bert(token, attention_mask=att_mask, return_dict=True) # 64 768
        hidden = outputs.last_hidden_state  # 64 128 728
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(torch.device(self.device))  # (B, L) 64 128#hidden.device
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(torch.device(self.device))  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        x = torch.cat([head_hidden, tail_hidden], 1)  # (B, 2H)
        x = self.linear(x)
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        
        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2


class BEiTEncoder_MY(nn.Module):
    def __init__(self, beit_path):
        super().__init__()
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(beit_path)
        self.model = BeitModel.from_pretrained(beit_path)
       
    def forward(self, images):
        inputs = self.feature_extractor(images, return_tensors="pt")  # inputs.pixel_values64 3 244 244
        feat_img = self.model(inputs.pixel_values.to(torch.device("cpu")), return_dict=True)  #self.model.device
        return feat_img.last_hidden_state  # 64 197 168


class SentenceRE_MY(nn.Module):

    def __init__(self, 
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
        
        
        
        # Model
        self.model = model   # softmax
        # self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
         
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
    

        self.ckpt = ckpt

       

    def eval_model(self, eval_loader):
        self.eval()
     
        pred_result = []
        with torch.no_grad():
            data = eval_loader
            args = data[1:]        
            # logits = self.parallel_model(*args)
            logits = self.model(*args)
            score, pred = logits.max(-1) # (B)
            # Save result
        return pred.item()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

class SentenceRE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def infer(self, item):
        """
        Args:
            item: {'text' or 'token', 'h': {'pos': [start, end]}, 't': ...}
        Return:
            (Name of the relation of the sentence, score)
        """
        raise NotImplementedError


class SoftmaxNN_MY(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, image_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.image_encoder = image_encoder
        self.select_device = "cpu"
        self.num_class = num_class
        
        self.fc_att = nn.Linear(self.sentence_encoder.hidden_size, 768)  # 1536
        
        self.fc_txt = nn.Linear(self.sentence_encoder.hidden_size, 768)
        
        self.fc_image = nn.Linear(self.sentence_encoder.hidden_size, 768)
        
        
        self.fc_all = nn.Linear(768, num_class)
        
        self.fc = nn.Linear(768*3, num_class)
        
        
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        self.device = torch.device(self.select_device)
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        _item = self.sentence_encoder.tokenize(item)
        item = []
        for x in _item:
            item.append(x.to(self.device))    # device  next(self.parameters()).device
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    
    def forward(self, *args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        feat_txt = self.sentence_encoder(*args[:-1])  # (B, H) 文本特征 64 1536
        
        # 获取图片特征
        feat_img = self.image_encoder(args[-1])  # 64 197 768
        feat_cls = feat_img[:, 0, :]  # 64 768
        feat_tokens = feat_img[:, 1:, :]  # 64 196 768
        weighted_feat = self.cross_modality_att(feat_txt, feat_tokens)  # 64  768
        
        ft = self.fc_txt(feat_txt)
        fi = self.fc_image(torch.cat((feat_cls, weighted_feat), dim=1))  # 64 768
        f_final = ft+fi   
        #rep = self.drop(f_final)   #原先是这两行
        #logits = self.fc_all(rep)  # 层 768 23   输出64 23
        
        rep = self.drop(torch.cat((feat_txt, feat_cls), dim=1))   # 目前使用这一行 simple-cat
        #rep = self.drop(torch.cat((feat_txt,weighted_feat),dim=1))  # 自己改的
        
        # rep = self.sentence_encoder(*args) # (B, H)
        
        # rep = self.drop(feat_txt)
        logits = self.fc(rep) # (B, N)   # simple——cat
        return logits

    def logit_to_score(self, logits):
        return torch.softmax(logits, -1)
    
    def cross_modality_att(self, feat_txt, feat_tokens):
        query = self.fc_att(feat_txt)  # 64 768
        att_score = torch.einsum('bth,bh->bt',[feat_tokens, query]) # 左边是等式右边是进行操作的矩阵  64 196 768  #  生成64 196
        att_p = F.softmax(att_score, dim=1)  # 求得每行的softmax
        
        weighted_feat = torch.einsum('bth,bt->bh',[feat_tokens, att_p])
        return weighted_feat






parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased', 
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--beit_path', default='microsoft/beit-base-patch16-224-pt22k',
                    help='beit ckpt path / model name (hugginface)')

parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--pooler', default='entity', choices=['cls', 'entity'], 
        help='Sentence representation pooler')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')
parser.add_argument('--mask_entity', action='store_true', 
        help='Mask entity mentions')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='mega', choices=['none', 'semeval', 'wiki80', 'tacred', 'mega'], 
        help='Dataset. If not none, the following args can be ignored')


# Hyper-parameters
parser.add_argument('--batch_size', default=1, type=int,
        help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
        help='Learning rate')
parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=30, type=int,
        help='Max number of training epochs')

# Seed
parser.add_argument('--seed', default=42, type=int,
        help='Seed')

args = parser.parse_args()

# Set random seed
set_seed(args.seed)

# Some basic settings
# root_path = '/data/home/zengdj/workspaces/cyx/megaBaseline/'
# sys.path.append(root_path)
# 得到提前训练的结果
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
if len(args.ckpt) == 0:
    args.ckpt = '{}_{}_{}'.format(args.dataset, args.pretrain_path, args.pooler)+"_simple_cat"  # 更改
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)
ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt)
print(os.path.abspath(ckpt))

# 获取得到mega数据集的路径
if args.dataset == 'mega':
    train_txt_file = os.path.join(root_path, 'benchmark', args.dataset, 'txt', '{}_train.txt'.format(args.dataset))
    val_txt_file = os.path.join(root_path, 'benchmark', args.dataset, 'txt', '{}_val.txt'.format(args.dataset))
    test_txt_file = os.path.join(root_path, 'benchmark', args.dataset, 'txt', '{}_test.txt'.format(args.dataset))
    train_img_dir = os.path.join(root_path, 'benchmark', args.dataset, 'img_org', 'train')
    val_img_dir = os.path.join(root_path, 'benchmark', args.dataset, 'img_org', 'val')
    test_img_dir = os.path.join(root_path, 'benchmark', args.dataset, 'img_org', 'test')
    rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))

# 加载得到实体关系 
rel2id = json.load(open(rel2id_file))
id2rel = {v:k for k,v in rel2id.items()}








class My_Model():
        def __init__(self,use_model=False):
                if use_model:
                        image_encoder = BEiTEncoder_MY(args.beit_path)
                        sentence_encoder = BERTEntityEncoder_MY(
                                max_length=args.max_length, 
                                pretrain_path=args.pretrain_path,
                                mask_entity=args.mask_entity
                                )
                        self.model = SoftmaxNN_MY(sentence_encoder, image_encoder, len(rel2id), rel2id)
                        framework = SentenceRE_MY(
                                                model=self.model,
                                                ckpt=ckpt,
                                                batch_size=args.batch_size,
                                                max_epoch=args.max_epoch,
                                                lr=args.lr,
                                                opt='adamw'
                                                )
                        self.final_model = framework
        def find_my_file(self,ob_id,get_head,get_tail):
                if 0<= ob_id < 12247:
                        ob_image_path = train_img_dir
                        ob_txt_path = train_txt_file
                        target = ob_id
                elif ob_id < 13871:
                        ob_image_path = val_img_dir
                        ob_txt_path = val_txt_file
                        target = ob_id - 12247
                elif ob_id < 15485:
                        ob_image_path = test_img_dir
                        ob_txt_path = test_txt_file
                        target = ob_id - 13871
                self.ob_txt_path = ob_txt_path
                self.ob_image_path = ob_image_path
                self.target = target 
                self.get_head = get_head
                self.get_tail = get_tail
        def get_item(self,target,get_head=[],get_tail=[],text_path=None):
                self.find_my_file(target,get_head,get_tail)
                # 首先得到文件组合
                f = open(self.ob_txt_path)
                data = []
                times = 0
                # 获取得到目标结果
                for line in f.readlines():
                        if len(line) > 0 and times == target:
                                line = line.rstrip()
                                data.append(eval(line))
                                if len(get_head) == 2:
                                        data[0]['h']['pos'] = get_head
                                if len(get_tail) == 2:
                                        data[0]['t']['pos'] = get_tail
                                break
                        times += 1
                f.close()
                return data[0]
        def get_my_data(self,text_path,pic_dir,rel2id,tokennizer,target,get_head,get_tail,**kwargs):
                # 首先得到文件组合
                f = open(text_path)
                data = []
                times = 0
                # 获取得到目标结果
                for line in f.readlines():
                        if len(line) > 0 and times == target:
                                line = line.rstrip()
                                data.append(eval(line))
                                if len(get_head) == 2:
                                        data[0]['h']['pos'] = get_head
                                if len(get_tail) == 2:
                                        data[0]['t']['pos'] = get_tail
                                break
                        times += 1
                f.close()
                
                # 找到所有图片的路径，并以图片的名字作为key存成dict
                pic_path_dict = {}
                pathList = os.listdir(pic_dir)
                for picName in pathList:
                        pic_path = os.path.join(pic_dir, picName)    #拼接文件路径
                        pic_path_dict[picName] = pic_path
                item = data[0]
                seq = list(tokennizer(item,**kwargs))
                # pos1 = seq[-1]
                # pos2 = seq[-2]
                image_path = pic_path_dict[item['img_id']]
                # image = cv2.imread(image_path)
                # jpeg4py速度比cv2快
                image = jpeg4py.JPEG(image_path).decode()
                # image = Image.open(image_path)
                img_tensor = torch.from_numpy(image).permute((2, 0, 1)) #.unsqueeze(0)
                res = [rel2id[item['relation']]] + seq + [img_tensor]
                return res  # label, seq1, seq2, ...,pic
        def eval(self):
                test_loader = self.get_my_data(self.ob_txt_path,self.ob_image_path,rel2id,self.model.sentence_encoder.tokenize,self.target,self.get_head,self.get_tail)
                result = self.final_model.eval_model(test_loader)
                return id2rel[result]









