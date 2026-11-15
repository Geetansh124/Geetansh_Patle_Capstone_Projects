"""Patient-grouped holdout evaluation with confusion matrix and per-label metrics."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from train_classifier import ChestXrayDataset, make_model


def main():
    p=argparse.ArgumentParser(description="For education only — not for clinical use.")
    p.add_argument('--csv',type=Path,required=True); p.add_argument('--images',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('artifacts/evaluation.json')); p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args(); frame=pd.read_csv(a.csv).rename(columns={'Image Index':'image'}).dropna(subset=['image','Patient ID'])
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False); labels=ckpt['labels']
    from sklearn.model_selection import GroupShuffleSplit
    _,test_idx=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(frame,groups=frame['Patient ID']))
    ds=ChestXrayDataset(frame.iloc[test_idx],a.images,labels); loader=DataLoader(ds,batch_size=a.batch_size)
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=make_model(len(labels)); model.load_state_dict(ckpt['state_dict']); model.to(device).eval()
    ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader: ys.append(y.numpy()); ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    y=np.concatenate(ys); prob=np.concatenate(ps); pred=prob>=.5
    per={}
    for i,label in enumerate(labels):
        pr,re,f1,_=precision_recall_fscore_support(y[:,i],pred[:,i],average='binary',zero_division=0)
        per[label]={'precision':float(pr),'recall':float(re),'f1':float(f1),'confusion_matrix':confusion_matrix(y[:,i],pred[:,i],labels=[0,1]).tolist(),'roc_auc':float(roc_auc_score(y[:,i],prob[:,i])) if len(np.unique(y[:,i]))==2 else None}
    result={'notice':'For education only — not for clinical use.','threshold':.5,'test_images':len(y),'labels':per,'model_checkpoint':str(a.checkpoint)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
