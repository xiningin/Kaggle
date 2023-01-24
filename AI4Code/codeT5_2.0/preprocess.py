import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import os
import argparse
import re
import torch

"""
    正式训练： python preprocess.py --train_all_json True
    demo测试： python preprocess.py
"""

# 判断是否为正式训练
parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--train_all_json', type=bool, default=False)
parser.add_argument('--sample_num', type=int, default=40) # 画图看出来的集中在这个区域好些
args = parser.parse_args()

data_dir = Path('../')
if not os.path.exists("./data"):
    os.mkdir("./data")

def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type':'category' , 'source':'str'}
        ).assign(id=path.stem)
        .rename_axis('cell_id')
    )

if args.train_all_json :
    paths_train = list((data_dir / 'train').glob('*.json'))
else :
    paths_train = list((data_dir / 'train').glob('*.json'))[:20]

notebooks_train = [
    read_notebook(path) for path in tqdm(paths_train , desc='Train notebooks')
]

df = (
    pd.concat(notebooks_train)
        .set_index('id' , append=True)
        .swaplevel()
        .sort_index(level='id' , sort_remaining=False)
)
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True
).str.split()

def get_ranks(base , derived):
    return [base.index(d) for d in derived]

df_orders_ = df_orders.to_frame().join(
    df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
    how='right'
)

ranks = {}
for id_ , cell_order , cell_id in df_orders_.itertuples():
    ranks[id_] = {'cell_id':cell_id , 'rank':get_ranks(cell_order , cell_id)}
df_ranks = (
    pd.DataFrame.from_dict(ranks , orient='index')
        .rename_axis('id')
        .apply(pd.Series.explode)
        .set_index('cell_id' , append=True)
)


df_ancestors = pd.read_csv(data_dir / 'train_ancestors.csv', index_col='id')
df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])
df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

from sklearn.model_selection import GroupShuffleSplit

NVALID = 0.1  # size of validation set
splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))
train_df = df.loc[train_ind].reset_index(drop=True)
val_df = df.loc[val_ind].reset_index(drop=True)

# 代码清洗
def clean_code(cell):
    cleaned_code = re.sub(r"^#.*\n" , " " , str(cell) , flags=re.MULTILINE) #第一次去除 #类注释
    cleaned_code = re.sub(r'""".+"""', ' ', cleaned_code) # 去除 “”“”“”类型注释
    cleaned_code = re.sub(r' +', ' ', cleaned_code) # 去除多个空格那类型
    cleaned_code.replace("\\n" , " ")
    cleaned_code = cleaned_code.replace("\n" , " ") # 去除换行符
    return cleaned_code


from transformers import RobertaTokenizer , T5ForConditionalGeneration
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 使用模型生成总结
def summary_code(sent):
    inputs_ids = tokenizer(sent , return_tensors="pt")["input_ids"].to(device)
    generated_ids = model.generate(inputs_ids , max_length=50)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 浓缩一个cells中所有code cell
def codes2sentence(cells , n):
    """
        cells: 所有cell，一般是code cell
    """
    cells = [clean_code(cell) for cell in cells]
    # 对于cells特别多的，我们采取等距采样的方法
    if n < len(cells):
        sample_cells = []
        step = len(cells) / n # 步长
        idx = 0
        while int(np.round(idx)) < len(cells):
            sample_cells.append(cells[int(np.round(idx))])
            idx += step
        assert cells[0] in sample_cells
        if cells[-1] not in sample_cells:
            sample_cells[-1] = cells[-1]
        cells = sample_cells
    tmp_summary = []
    summary = []
    for cell in cells:
        tmp_summary.append(summary_code(str(cell)))
    summary = summary_code(" ".join(tmp_summary))
    
    return summary

# 获取一个notebook中的概括性特征
def get_features(df):
    features = dict()
    df = df.sort_values("rank").reset_index(drop=True)
    for idx , sub_df in tqdm(df.groupby("id")):
        features[idx] = dict()
        num_of_markdown = sub_df[sub_df["cell_type"] == "markdown"].shape[0]
        code_sub_df = sub_df[sub_df["cell_type"] == "code"]
        num_of_code = code_sub_df.shape[0]
        sum_codes = codes2sentence(code_sub_df["source"].values , args.sample_num) # 获取codes的总结
        features[idx]["num_of_code"] = num_of_code
        features[idx]["num_of_markdown"] = num_of_markdown
        features[idx]["sum_codes"] = sum_codes
    return features

vali_features = get_features(val_df)
train_features = get_features(train_df)

# 保存csv文件
train_markdown_df = train_df[train_df["cell_type"] == "markdown"].reset_index(drop=True)
vali_markdown_df = val_df[val_df["cell_type"] == "markdown"].reset_index(drop=True)
    
if args.train_all_json :
    train_markdown_df.to_csv("./data/train_markdown.csv" , index=False)
    vali_markdown_df.to_csv("./data/vali_markdown.csv" , index=False)
    val_df.to_csv("./data/val.csv", index=False)
    train_df.to_csv("./data/train.csv", index=False)
    json.dump(vali_features , open("./data/vali_features.json" , "wt"))
    json.dump(train_features , open("./data/train_features.json" , "wt"))
else :
    train_markdown_df.to_csv("./data/train_markdown_demo.csv" , index=False)
    vali_markdown_df.to_csv("./data/vali_markdown_demo.csv" , index=False)
    val_df.to_csv("./data/val_demo.csv", index=False)
    train_df.to_csv("./data/train_demo.csv", index=False)
    json.dump(vali_features , open("./data/vali_features_demo.json" , "wt"))
    json.dump(train_features , open("./data/train_features_demo.json" , "wt"))