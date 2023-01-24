import json
from pathlib import Path
from dataset import MarkdownDataset
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader , Dataset
from model import MarkdownModel
from tqdm import tqdm
import sys , os
import metrics
import argparse
import torch

# n_workers这个在我这个电脑上运不了，AutoDL上可以，这东西在dataLoader那里
# python train.py --md_max_len 64 --total_max_len 512 --batch_size 8 --accumulation_steps 4 --epochs 5 --n_workers 8
# python train.py --md_max_len 64 --total_max_len 512 --batch_size 8 --accumulation_steps 4 --epochs 5 --n_workers 4
parser = argparse.ArgumentParser(description='Process some arguments')

"""
    测试用的少数据
"""
parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
parser.add_argument('--train_mark_path', type=str, default='./data/train_markdown_demo.csv')
parser.add_argument('--train_features_path', type=str, default='./data/train_features_demo.json')
parser.add_argument('--val_mark_path', type=str, default='./data/vali_markdown_demo.csv')
parser.add_argument('--val_features_path', type=str, default='./data/vali_features_demo.json')
parser.add_argument('--val_path', type=str, default="./data/val_demo.csv")

"""
    使用全部数据
"""
# parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
# parser.add_argument('--train_mark_path', type=str, default='./data/train_markdown.csv')
# parser.add_argument('--train_features_path', type=str, default='./data/train_features.json')
# parser.add_argument('--val_mark_path', type=str, default='./data/vali_markdown.csv')
# parser.add_argument('--val_features_path', type=str, default='./data/vali_features.csv')
# parser.add_argument('--val_path', type=str, default="./data/val.csv")

parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=8)

args = parser.parse_args()

if not os.path.exists("./output"):
    os.mkdir("./output")

data_dir = Path("../")

train_markdown_df = pd.read_csv(args.train_mark_path).drop("parent_id" , axis=1).dropna().reset_index(drop=True)
train_features = json.load(open(args.train_features_path))
vali_markdown_df = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
vali_features = json.load(open(args.val_features_path))
vali_df = pd.read_csv(args.val_path)

df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

# dataset返回的：ids ， mask ， markdown_rate ， pct_rank
train_dataset = MarkdownDataset(train_markdown_df , model_name_or_path=args.model_name_or_path , markdown_max_len=args.md_max_len ,
                                total_max_len=args.total_max_len , features=train_features)
vali_dataset = MarkdownDataset(vali_markdown_df, model_name_or_path=args.model_name_or_path, markdown_max_len=args.md_max_len,
                                total_max_len=args.total_max_len, features=vali_features)
train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.n_workers,
    pin_memory=False,
    drop_last=True
)
vali_loader = DataLoader(
    vali_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.n_workers,
    pin_memory=False,
    drop_last=False
)

# 数据放cuda里面去
def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda() # 最后一位是target的

def validate(model , vali_loader):
    model.eval()
    tbar = tqdm(vali_loader , file=sys.stdout)
    preds = []
    labels = []
    with torch.no_grad():
        for idx , data in enumerate(tbar):
            inputs , target = read_data(data)
            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())
    return np.concatenate(labels) , np.concatenate(preds)


def train(model, train_loader, vali_loader, epochs):
    np.random.seed(416)
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs * len(train_loader) / args.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5 , correct_bias=False , no_deprecation_warning=True)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler() 

    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)
            """
                inputs: ids ， mask ， markdown_rate
                target: pct_rank
            """

            with torch.cuda.amp.autocast(): # 使用混合精度的方便方法，以加速训练
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward() # 先将梯度放大 防止梯度消失
            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item()) # detach()将tensor从计算图中抽离出来，不在需要计算grad
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")


        _, y_pred = validate(model, vali_loader)
        vali_df["pred"] = vali_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        vali_df.loc[vali_df["cell_type"] == "markdown", "pred"] = y_pred #赋值markdown的顺序
        y_dummy = vali_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
        print("Preds score: ", metrics.kendall_tau(df_orders.loc[y_dummy.index], y_dummy))
        """
            测试的少数据
        """
        torch.save(model.state_dict(), "./output/model_demo.bin")
        """
            用的全部数据
        """
        # torch.save(model.state_dict(), "./output/model.bin")

    return model, y_pred

if __name__ == "__main__":
    model = MarkdownModel(args.model_name_or_path)
    model = model.cuda()
    model, y_pred = train(model, train_loader, vali_loader, epochs=args.epochs)