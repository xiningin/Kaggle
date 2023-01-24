from torch.utils.data import DataLoader , Dataset
import torch
from transformers import AutoTokenizer

class MarkdownDataset(Dataset):
    def __init__(self , df , model_name_or_path , total_max_len , markdown_max_len , features):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.markdown_max_len = markdown_max_len
        self.total_max_len = total_max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.features = features
    def __getitem__(self, index):
        row = self.df.iloc[index]
        inputs = self.tokenizer.encode_plus(
            row["source"],
            add_special_tokens=True,
            max_length=self.markdown_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.features[row.id]["codes"]],
            add_special_tokens=True,
            max_length=23,
            padding="max_length",
            truncation=True
        )
        num_markdown = self.features[row.id]["num_of_markdown"]
        num_code = self.features[row.id]["num_of_code"]
        if num_markdown + num_code == 0:
            markdown_rate = torch.Tensor([0])
        else:
            markdown_rate = torch.FloatTensor([num_markdown / (num_code + num_markdown)])
        
        # 准备input_ids
        """
            末尾追加code cell翻译的inputs ， 由于encode的时候设置了“add_specical_tokens=True”，所以自带<s>
            最后形成<s> Markdown content <s> Code content 1 <s> Code content 2 <s> ... <s> Code content 20 <s>
        """
        ids = inputs['input_ids']
        for x in code_inputs['input_ids']:
            ids.extend(x[:-1]) 
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id ,] * (self.total_max_len - len(ids)) # 添加<padding>补齐到total_max_len
        ids = torch.LongTensor(ids)

        # 准备attention_mask,这个步骤需要与上面那个同步的
        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            mask.extend(x[:-1])
        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        assert len(ids) == self.total_max_len
        assert len(mask) == self.total_max_len

        return ids , mask , markdown_rate , torch.FloatTensor([row.pct_rank])
    
    def __len__(self):
        return self.df.shape[0]