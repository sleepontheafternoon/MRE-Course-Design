# coding:utf-8
import torch
import numpy as np
import json
import os,sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path) 
import opennre
from opennre import encoder, model, framework
import argparse
import logging
import random
import wandb

#wandb.init(project="cyx_beit",name="cat1")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

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
# parser.add_argument('--train_file', default='', type=str,
#         help='Training data file')
# parser.add_argument('--val_file', default='', type=str,
#         help='Validation data file')
# parser.add_argument('--test_file', default='', type=str,
#         help='Test data file')
# parser.add_argument('--rel2id_file', default='', type=str,
#         help='Relation to ID file')

# Hyper-parameters
parser.add_argument('--batch_size', default=64, type=int,
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
    args.ckpt = '{}_{}_{}'.format(args.dataset, args.pretrain_path, args.pooler)+"myself"  # 更改
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

# 记录
logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(rel2id_file))

# Define the sentence encoder
if args.pooler == 'entity':
    sentence_encoder = opennre.encoder.BERTEntityEncoder(
        max_length=args.max_length, 
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
elif args.pooler == 'cls':
    sentence_encoder = opennre.encoder.BERTEncoder(
        max_length=args.max_length, 
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
else:
    raise NotImplementedError

# Define the image encoder
image_encoder = opennre.encoder.BEiTEncoder(args.beit_path)  # 即openre中的image_encoder

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, image_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_txt_file,
    val_txt_file,
    test_txt_file,   
    train_img_dir,
    val_img_dir,
    test_img_dir,
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    opt='adamw'
)

#
# Train the model
if not args.only_test:
    framework.train_model('micro_f1')

# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
logging.info('Test set results:')
logging.info('Accuracy: {}'.format(result['acc']))
logging.info('Micro precision: {}'.format(result['micro_p']))
logging.info('Micro recall: {}'.format(result['micro_r']))
logging.info('Micro F1: {}'.format(result['micro_f1']))
