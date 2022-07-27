import torch
from torch import nn, optim
from .base_model import SentenceRE
import torch.nn.functional as F

class SoftmaxNN(SentenceRE):
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
        self.select_device = "cuda:0"
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
        feat_img = self.image_encoder(args[-1])  # 64 197 768  args[-1] 3 600 800
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
