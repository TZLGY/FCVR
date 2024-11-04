from FCVR_dataset import bulid_loader

from FCVR_model import FCVR
import yaml,torch,argparse,time
from tqdm import tqdm
import numpy as np
from tabulate import tabulate
def eval_model(model,eval_loader,device_ids):
    model = model.cuda(device=device_ids[0])
    results_IOU=[]
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for sample in tqdm(eval_loader,desc='eval...',ncols=100):
            sample = sample2cuda(sample,device_ids)
            G_start,G_end=sample['time_start'],sample['time_end']
            P_start,P_end,_,_=model(sample)
            iou=cal_met(P_start,G_start,P_end,G_end)
            results_IOU+=iou.cpu().tolist()
    end_time = time.time()
    cost_time_ms=(float(end_time - start_time) * 1000.0)/len(results_IOU)
    results_IOU=np.array(results_IOU)
    met_table=get_finres(results_IOU)
    return met_table,cost_time_ms
def get_finres(results_IOU):
    results_table={'R@1 IOU=0.3':0,'R@1 IOU=0.5':0,'R@1 IOU=0.7':0}
    all_sample=len(results_IOU)
    results_table['R@1 IOU=0.3']=[(sum(results_IOU>=0.3))/all_sample]
    results_table['R@1 IOU=0.5']=[(sum(results_IOU>=0.5))/all_sample]
    results_table['R@1 IOU=0.7']=[(sum(results_IOU>=0.7))/all_sample]
    res=tabulate(results_table,headers='keys',tablefmt='github',numalign='center')
    return res
def batch_iou_loss(lines1, lines2):
    intersection_start = torch.max(lines1[:, 0], lines2[:, 0])
    intersection_end = torch.min(lines1[:, 1], lines2[:, 1])
    
    intersection_length = torch.clamp(intersection_end - intersection_start, min=0)
    
    union_length = (lines1[:, 1] - lines1[:, 0]) + (lines2[:, 1] - lines2[:, 0]) - intersection_length
    
    iou = intersection_length / union_length.clamp(min=1e-6)  
    
    return iou
def cal_met(P_start,G_start,P_end,G_end,):
    
    P_line=torch.concatenate((P_start.view(-1,1),P_end.view(-1,1)),dim=-1)
    G_line=torch.concatenate((G_start.view(-1,1),G_end.view(-1,1)),dim=-1)
    iou=batch_iou_loss(P_line,G_line)
    return iou
def sample2cuda(sample,device_ids):
    for k in sample:
        if k not in ['tokens_id']:
            sample[k]=sample[k].cuda(device_ids[0])
        else:
            for kk in sample[k]:
                sample[k][kk]=sample[k][kk].cuda(device_ids[0])
    return sample
def display_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        print(yaml.dump(data, default_flow_style=False, sort_keys=False))
def main(args):
    # display_yaml_file(args.config_file)
    with open(args.config_file, 'r' as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    Vedio_feature_path=args['DataSet']['Vedio_feature_path']
    train_ann_data_path=args['DataSet']['train_ann_data_path']
    eval_ann_data_path=args['DataSet']['eval_ann_data_path']
    text_encoder_root=args['DataSet']['text_encoder_root']
    batch_size=args['DataSet']['batch_size']
    device_ids=args['Train']['device_ids']
    weights=args['Train']['weights']

    model=ReLoCLNet(args)
    if weights:
        flag=model.load_state_dict(torch.load(args['Train']['weights']),strict=False)
        print(flag)
        print(f"load weights from {args['Train']['weights']}")

    


    train_loader,eval_loader=bulid_loader(Vedio_feature_path,train_ann_data_path,eval_ann_data_path,text_encoder_root,batch_size)



    
    start_time_train = time.time() 
    met_table, cost_time_ms = eval_model(model, train_loader, device_ids)
    total_time_train = time.time() - start_time_train  


    
    met_table,cost_time_ms=eval_model(model,train_loader,device_ids)
    print(f"{'*'*20}训练集指标{'*'*20}")
    print(met_table)
    print(f"训练集总用时：{round(total_time_train, 2)}秒")
    print(f"每次检索平均用时{round(cost_time_ms,4)}ms")



    start_time_eval = time.time()  
    met_table, cost_time_ms = eval_model(model, eval_loader, device_ids)
    total_time_eval = time.time() - start_time_eval  




    
    met_table,cost_time_ms=eval_model(model,eval_loader,device_ids)
    print(f"{'*'*20}测试集指标{'*'*20}")
    print(met_table)
    print(f"测试集总用时：{round(total_time_eval, 2)}秒")
    print(f"每次检索平均用时{round(cost_time_ms,4)}ms")
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="评估模型")
    parser.add_argument("--config_file", type=str, default='D:\\A_Paper_code\\idea_多任务\\eval.yaml', help="配置文件")
    args = parser.parse_args()
    main(args)
    
