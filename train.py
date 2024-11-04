from RELO_dataset import bulid_loader
from ReLoCLNet_model import ReLoCLNet
import yaml,torch,argparse
from torch import optim,nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train(model,epoches,train_loader,eval_loader,opt,loss_func,lr_drop,device_ids,logs_writer,save_dir,inter_pred_th,loss_gamma):
    if_multi=len(device_ids)>1
    if if_multi:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda(device=device_ids[0])
    else:
        model = model.cuda(device=device_ids[0])
    train_step=0
    eval_step=0
    best_iou=0
    for epoch in range(epoches):
        val_iou=[]
        train_iou=[]
        val_loss=[]
        train_loss=[]
        model.train()
        for sample in tqdm(train_loader,desc='training...',ncols=80):
            sample = sample2cuda(sample,device_ids)

            G_start,G_end=sample['time_start'],sample['time_end']
            P_start,P_end,inter_pred_out,similarity_score=model(sample)
            loss,iou=cal_loss(P_start,G_start,P_end,G_end,loss_func,inter_pred_out,device_ids,inter_pred_th,similarity_score,loss_gamma)
            logs_writer.add_scalar(tag='train_loss_step',scalar_value=loss.item(),global_step=train_step)
            logs_writer.add_scalar(tag='train_iou_step',scalar_value=iou.item(),global_step=train_step)
            train_step+=1
            train_iou.append(iou.item())
            train_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        lr_drop.step()
        train_iou=sum(train_iou)/len(train_iou)
        train_loss=sum(train_loss)/len(train_loss)
        model.eval()
        with torch.no_grad():
            for sample in tqdm(eval_loader,desc='eval...',ncols=80):
                sample = sample2cuda(sample,device_ids)
                G_start,G_end=sample['time_start'],sample['time_end']
                P_start,P_end,_,_=model(sample)
                loss,iou=cal_loss(P_start,G_start,P_end,G_end,loss_func)
                logs_writer.add_scalar(tag='eval_loss_step',scalar_value=loss.item(),global_step=eval_step)
                logs_writer.add_scalar(tag='eval_iou_step',scalar_value=iou.item(),global_step=eval_step)
                eval_step+=1
                val_iou.append(iou.item())
                val_loss.append(loss.item())

            val_iou=sum(val_iou)/len(val_iou)
            val_loss=sum(val_loss)/len(val_loss)
            if val_iou>best_iou:
                best_iou=val_iou
                torch.save(model.state_dict(),f"{save_dir}/best_charades_sts.pt") 
            torch.save(model.state_dict(),f"{save_dir}/last_charades_sts.pt") 
        logs_writer.add_scalar(tag='eval_iou_epoch',scalar_value=train_iou,global_step=epoch)
        logs_writer.add_scalar(tag='train_iou_epoch',scalar_value=val_iou,global_step=epoch)

        logs_writer.add_scalar(tag='eval_loss_epoch',scalar_value=train_loss,global_step=epoch)
        logs_writer.add_scalar(tag='train_loss_epoch',scalar_value=val_loss,global_step=epoch)
        text=f"{epoch}/{epoches} train_loss={train_loss} train_iou={train_iou} val_iou={val_iou} val_loss={val_loss}"
        print(text)

def batch_iou_loss(lines1, lines2):
    intersection_start = torch.max(lines1[:, 0], lines2[:, 0])
    intersection_end = torch.min(lines1[:, 1], lines2[:, 1])
    
    intersection_length = torch.clamp(intersection_end - intersection_start, min=0)
    
    union_length = (lines1[:, 1] - lines1[:, 0]) + (lines2[:, 1] - lines2[:, 0]) - intersection_length
    
    iou = intersection_length / union_length.clamp(min=1e-6)  
    
    return 1-iou.mean(),iou.mean(),iou
def cal_loss(P_start,G_start,P_end,G_end,loss_func,inter_pred_out=None,device_ids=None,inter_pred_th=0.5,similarity_score=None,loss_gamma=None):
    loss_start=loss_func(G_start,P_start)
    loss_end=loss_func(G_end,P_end)
    P_line=torch.concatenate((P_start.view(-1,1),P_end.view(-1,1)),dim=-1)
    G_line=torch.concatenate((G_start.view(-1,1),G_end.view(-1,1)),dim=-1)
    iou_loss,iou,iou_list=batch_iou_loss(P_line,G_line)
    if str(similarity_score)!='None':
        b=similarity_score.shape[0]
        similarity_label=torch.zeros((b)).to(similarity_score.device)
        similarity_loss=nn.MSELoss()(similarity_score,similarity_label)
    else:
        similarity_loss=0
    if str(inter_pred_out)!='None':
        labels=[]
        for now_iou in iou_list:
            if now_iou>inter_pred_th:
                label=1
            else:
                label=0
            labels.append(label)
        labels=torch.tensor(labels,dtype=torch.int64)
        labels=labels.cuda(device_ids[0])
        inter_loss=nn.CrossEntropyLoss()(inter_pred_out,labels)
        loss=loss_start*loss_gamma['loss_start']+loss_end*loss_gamma['loss_end']+iou_loss*loss_gamma['iou_loss']+inter_loss*loss_gamma['inter_loss']+similarity_loss*loss_gamma['similarity_loss']
    else:
        loss=loss_start+loss_end+iou_loss+similarity_loss
    return loss,iou
def sample2cuda(sample,device_ids):
    for k in sample:
        if k not in ['tokens_id']:
            sample[k]=sample[k].cuda(device_ids[0])
        else:
            for kk in sample[k]:
                sample[k][kk]=sample[k][kk].cuda(device_ids[0])
    return sample
def display_yaml_file(file_path):
    with open(file_path, 'r',encoding='utf-8') as file:#encoding='utf-8'为了在windows上运行
        data = yaml.safe_load(file)
        print(yaml.dump(data, default_flow_style=False, sort_keys=False))
def main(args):
    display_yaml_file(args.config_file)
    with open(args.config_file, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    logs_writer=SummaryWriter(args['Logs_dir'])
    epoches=args['Train']['epoches']
    Vedio_feature_path=args['DataSet']['Vedio_feature_path']
    train_ann_data_path=args['DataSet']['train_ann_data_path']
    eval_ann_data_path=args['DataSet']['eval_ann_data_path']
    text_encoder_root=args['DataSet']['text_encoder_root']
    batch_size=args['DataSet']['batch_size']
    base_lr=eval(args['Train']['base_lr'])
    save_dir=args['save_dir']
    inter_pred_th=args['Train']['inter_pred_th']
    model=ReLoCLNet(args)
    opt=eval(f"optim.{args['Train']['optim']}")(model.parameters(),lr=base_lr)
    loss_gamma=args['Train']['loss_gamma']
    lr_drop=optim.lr_scheduler.StepLR(opt,step_size=10,gamma=0.1)
    loss_func=nn.MSELoss()
    device_ids=args['Train']['device_ids']
    if args['Train']['weights']:
        print(f"load weights from {args['Train']['weights']}")
        print(model.load_state_dict(torch.load(args['Train']['weights']),strict=False))
           

    if args['Train']['if_train_query_encoder_textFeatureEncoder']==False:
        for na,par in model.named_parameters():
            if 'query_encoder.encoder' in na:
                par.requires_grad=False

    train_loader,eval_loader=bulid_loader(Vedio_feature_path,train_ann_data_path,eval_ann_data_path,text_encoder_root,batch_size)

    train(model,epoches,train_loader,
          eval_loader,opt,loss_func,lr_drop,device_ids,
          logs_writer,save_dir,inter_pred_th,loss_gamma)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--config_file", type=str, default='D:\\A_Paper_code\\idea_多任务\\train.yaml', help="配置文件")
    args = parser.parse_args()
    main(args)
    
