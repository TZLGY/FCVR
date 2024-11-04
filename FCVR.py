from FCVR_dataset import bulid_loader
from torch import nn
from transformers import AutoModel
import torch,random
import torch.nn.functional as F
class AdditiveAttention(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024):
        super(AdditiveAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.query_key_layer = nn.Linear(input_dim, hidden_dim)
        self.energy_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, input):
        query_key = self.query_key_layer(input)  
        
        energy = torch.tanh(query_key)             
        energy = self.energy_layer(energy).squeeze(-1)  
        
        attention_weights = torch.softmax(energy, dim=-1)  
        
        context = torch.bmm(attention_weights.unsqueeze(1), input)  
        
        return context.squeeze(1), attention_weights
class QueryEncoder(nn.Module):
    def __init__(self,text_encoder_path,text_feature_dim) -> None:
        super().__init__()
        self.encoder=AutoModel.from_pretrained(text_encoder_path)
        self.ffn=nn.Sequential(
            nn.Linear(768,512),
            nn.Mish(),
            nn.Linear(512,text_feature_dim),
            nn.Mish(),
            )
        # transformer_encoder
        transformer_layer=nn.TransformerEncoderLayer(d_model=text_feature_dim,nhead=4,batch_first=True)
        self.transformerBlock=nn.TransformerEncoder(transformer_layer,num_layers=2)
        # additive attention
        self.additive_attention=AdditiveAttention()
    def forward(self,input_ids):
        text_feature=self.encoder(**input_ids)['last_hidden_state']
        text_feature=self.ffn(text_feature)
        text_feature=self.transformerBlock(text_feature)
        Qv,_=self.additive_attention(text_feature)
        return Qv
class VideoEncoder(nn.Module):
    def __init__(self,video_feature_dim) -> None:
        super().__init__()
        self.ffn=nn.Sequential(
            nn.Linear(1024,512),
            nn.Mish(),
            nn.Linear(512,video_feature_dim),
            nn.Mish()
            
        )
        # transformer_encoder
        transformer_layer=nn.TransformerEncoderLayer(d_model=video_feature_dim,nhead=4,batch_first=True)
        self.transformerBlock=nn.TransformerEncoder(transformer_layer,num_layers=3)

    def forward(self,video_feature):
        video_feature=self.ffn(video_feature)
        Hv=self.transformerBlock(video_feature)
        return Hv
class MomentLocalization(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffn=nn.Sequential(
            nn.Linear(512,256),
            nn.Mish(),
            nn.Linear(256,512),
            nn.Mish()
        )
        self.start_head=nn.Sequential(
            nn.Conv1d(512,256,kernel_size=3,padding=1,stride=1),
            nn.Mish(),
            nn.Conv1d(256,128,kernel_size=3,padding=1,stride=1),
            nn.Mish(),
            nn.AdaptiveAvgPool2d(output_size=1)
        )
        self.end_head=nn.Sequential(
            nn.Conv1d(512,256,kernel_size=3,padding=1,stride=1),
            nn.Mish(),
            nn.Conv1d(256,128,kernel_size=3,padding=1,stride=1),
            nn.Mish(),
            nn.AdaptiveAvgPool2d(output_size=1)
        )
        self.Sig=nn.Sigmoid()
    def forward(self,Qv,Hv):
        bs=Qv.shape[0]
        Qv=self.ffn(Qv)
        QHv=(Qv*Hv).view(bs,-1,1)
        P_start=self.start_head(QHv).view(bs,-1)
        P_end=self.end_head(QHv).view(bs,-1)
        return P_start,P_end
class InterPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pred_layer=nn.Sequential(
            nn.Linear(512,256),
            nn.Mish(),
            nn.Linear(256,128),
            nn.Mish(),
            nn.Linear(128,2),
            nn.Softmax(dim=-1)
        )
    def forward(self,Qv,Hv):
        merge_feature=Qv*Hv
        bs,seq_len=merge_feature.shape
        out=self.pred_layer(merge_feature)
        return out
class Frames_Simlarity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_project=nn.Sequential(
            nn.Linear(1024,512),
            nn.Mish(),
            nn.Linear(512,512)
        )
    def forward(self,sample):
        all_feature=sample['all_video_feature']
        all_cos_sim=[]

        for now_video,roi_start,roi_end,length in zip(all_feature,sample['feature_start'],sample['feature_end'],sample['video_feature_length']):
            roi_start,roi_end,length=int(roi_start),int(roi_end),int(length)
            roi_feature=now_video[roi_start:roi_end,:]
            if length-roi_end>(roi_end-roi_start)+1:
                other_start=random.randint(roi_end+1,length-(roi_end-roi_start)-1)
                other_end=other_start+(roi_end-roi_start)
            elif roi_start-1>(roi_end-roi_start)+1:
                other_end=random.randint(roi_start-1-(roi_end-roi_start),roi_start-1)
                other_start=other_end-(roi_end-roi_start)
            else:
                other_start=roi_start+1
                other_end=roi_start+(roi_end-roi_start)
            other_feature=now_video[other_start:other_end,:]

            other_feature=self.feature_project(other_feature)
            other_feature=torch.sum(other_feature,dim=0)
            roi_feature=self.feature_project(roi_feature)
            roi_feature=torch.sum(roi_feature,dim=0)
            cos_sim = F.cosine_similarity(other_feature, roi_feature, dim=0)

            all_cos_sim.append(cos_sim)

        all_cos_sim=torch.stack(all_cos_sim,dim=0)
        return all_cos_sim

class FCVR(nn.Module):
    def __init__(self,args,text_feature_dim=512,video_feature_dim=512) -> None:
        super().__init__()
        self.args=args
        text_encoder_root=self.args['DataSet']['text_encoder_root']
        
        self.query_encoder=QueryEncoder(text_encoder_root,text_feature_dim)
        
        self.vedio_encoder=VideoEncoder(video_feature_dim)
        
        self.moment_localization=MomentLocalization()
        
        self.get_simlarity=Frames_Simlarity()
        
        if self.args['Model']['use_inter_pred']:
            self.inter_predictor=InterPredictor()


    def forward(self,sample):
       
        Qv=self.query_encoder(sample['tokens_id'])
        
        Hv=self.vedio_encoder(sample['cur_feature'])
         
        P_start,P_end=self.moment_localization(Qv,Hv)
        
        similarity_score=self.get_simlarity(sample)
        
        if self.args['Model']['use_inter_pred']:
            inter_pred_out=self.inter_predictor(Qv,Hv)
            return P_start,P_end,inter_pred_out,similarity_score
        else:
            return P_start,P_end,None,similarity_score





