import torch,json
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
class FCVR(Dataset):
    def __init__(self,Vedio_features,ann_data,text_encoder):
        super().__init__()
        self.Vedio_features=Vedio_features
        self.ann_data=ann_data
        self.text_encoder=text_encoder
    def __len__(self):
        return len(self.ann_data)
    def __getitem__(self, index) :
        sample=self.ann_data.loc[index].to_dict()
        for k,v in sample.items():
            if k in ['video']:
                v=v
            elif k in ['time_start','time_end','feature_start','feature_end','fps','number_frames']:
                v=torch.tensor(float(v))
            sample[k]=v
        cur_feature=self.Vedio_features[sample['video']]
  
        sample['video_feature_length']=len(cur_feature)
        pad_feature=torch.zeros((560,1024))
        pad_feature[:len(cur_feature),:]=cur_feature
        sample['all_video_feature']=pad_feature
        cur_feature=torch.sum(cur_feature,dim=0)

        tokens=eval(sample['tokens'])
        del sample['tokens']
        del sample['video']
        tokens_id=self.text_encoder(tokens,is_split_into_words=True,padding='max_length',max_length=20,truncation=True)#,return_tensors='pt'
        sample['tokens_id']={k:torch.tensor(v) for k,v in tokens_id.items()}
        sample['cur_feature']=cur_feature
    
        total_time=sample['number_frames']/sample['fps']
        sample['time_start']=(sample['time_start']/total_time).view(1)
        sample['time_end']=(sample['time_end']/total_time).view(1)
        return sample
def bulid_loader(Vedio_feature_path,train_ann_data_path,eval_ann_data_path,text_encoder_root,batch_size=16):
    
    Vedio_features=torch.load(Vedio_feature_path)
    
    train_ann_data=pd.read_csv(train_ann_data_path)
    
    eval_ann_data=pd.read_csv(eval_ann_data_path)
    print('End')
    text_encoder=AutoTokenizer.from_pretrained(text_encoder_root)

    train_dataset=RELO_DataSet(Vedio_features=Vedio_features,ann_data=train_ann_data,text_encoder=text_encoder)
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

    eval_dataset=RELO_DataSet(Vedio_features=Vedio_features,ann_data=eval_ann_data,text_encoder=text_encoder)
    eval_loader=DataLoader(dataset=eval_dataset,batch_size=batch_size,shuffle=False,drop_last=True)
    print('#'*100)
    print(f" train_dataset={len(train_dataset)}\n eval_dataset={len(eval_dataset)} \n batch_size={batch_size}")
    print('#'*100)
    return train_loader,eval_loader

if __name__=='__main__':
    Vedio_feature_path=
    train_ann_data_path=
    eval_ann_data_path=
    text_encoder_root=
    batch_size=
    train_loader,eval_loader=bulid_loader(Vedio_feature_path,train_ann_data_path,eval_ann_data_path,text_encoder_root,batch_size)
    for sample in train_loader:
        for k in sample:
            try:
                print(k,sample[k].shape)
            except:
                for kk in sample[k]:
                    print(k,kk,sample[k][kk].shape)
        break
