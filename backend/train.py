from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import CHECKPOINT_DIR, FINDINGS
from .data import CheXpertDataset, compute_pos_weight, find_dataset_root
from .metrics import calculate_metrics
from .models import build_model, forward_logits


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval(); losses=[]; probs=[]; truths=[]
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = forward_logits(model, images)
            losses.append(float(criterion(logits, labels).item()) * images.size(0))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            truths.append(labels.cpu().numpy())
    y_prob=np.concatenate(probs, axis=0); y_true=np.concatenate(truths, axis=0)
    summary, per_finding=calculate_metrics(y_true, y_prob, threshold=0.5)
    return float(sum(losses)/len(loader.dataset)), summary, per_finding, y_true, y_prob


def main():
    parser=argparse.ArgumentParser(description="Train a CheXpert multi-label classifier")
    parser.add_argument('--model', required=True, choices=['vit','densenet','capsnet','gat'])
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-train-samples', type=int, default=0)
    parser.add_argument('--max-valid-samples', type=int, default=0)
    parser.add_argument('--output-dir', default=str(CHECKPOINT_DIR))
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args=parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1: parser.error('epochs and batch-size must be positive')
    set_seed(args.seed)
    root=find_dataset_root(args.data_root)
    train_ds=CheXpertDataset(root/'train.csv', root, args.model, args.max_train_samples, train=True)
    valid_ds=CheXpertDataset(root/'valid.csv', root, args.model, args.max_valid_samples, train=False)
    if not len(train_ds) or not len(valid_ds): raise RuntimeError('Training and validation datasets must not be empty.')
    workers=max(0,args.num_workers)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=workers,pin_memory=torch.cuda.is_available())
    valid_loader=DataLoader(valid_ds,batch_size=args.batch_size,shuffle=False,num_workers=workers,pin_memory=torch.cuda.is_available())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(args.model, pretrained=not args.no_pretrained).to(device)
    lr=args.learning_rate or {'vit':2e-5,'densenet':1e-4,'capsnet':1e-4,'gat':1e-4}[args.model]
    pos_weight=compute_pos_weight(train_ds).to(device)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    output_dir=Path(args.output_dir).expanduser().resolve(); output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_path=output_dir/f'{args.model}.pt'
    history=[]; best_auc=-float('inf'); best_loss=float('inf')
    print(f'Model: {args.model} | device: {device} | train: {len(train_ds)} | validation: {len(valid_ds)}')
    print('Label policy: uncertain -1 -> 0; missing NaN -> 0 (baseline experiment assumption).')
    for epoch in range(1,args.epochs+1):
        started=time.perf_counter(); model.train(); train_total=0.0
        for images,labels in tqdm(train_loader,desc=f'Epoch {epoch}/{args.epochs}'):
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=forward_logits(model,images)
            loss=criterion(logits,labels)
            loss.backward(); optimizer.step()
            train_total += float(loss.item())*images.size(0)
        train_loss=train_total/len(train_ds)
        val_loss,summary,per_finding,_,_=evaluate(model,valid_loader,criterion,device)
        scheduler.step()
        auc=summary.get('macro_auroc')
        score=float(auc) if auc is not None else -val_loss
        improved=score>best_auc if auc is not None else val_loss<best_loss
        if improved:
            best_auc=score if auc is not None else best_auc
            best_loss=val_loss
            payload={
                'model_name':args.model,'model_state_dict':model.state_dict(),
                'labels':FINDINGS,'epoch':epoch,'validation_summary':summary,
                'input_size':224,'label_policy':{'uncertain':0,'missing':0},
                'pretrained_initialization':not args.no_pretrained,
            }
            torch.save(payload,checkpoint_path)
            per_finding.to_csv(output_dir/f'{args.model}_metrics.csv',index=False)
            (output_dir/f'{args.model}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        entry={'epoch':epoch,'train_loss':train_loss,'validation_loss':val_loss,
               'validation_macro_auroc':summary.get('macro_auroc'),
               'validation_macro_auprc':summary.get('macro_auprc'),
               'learning_rate':optimizer.param_groups[0]['lr'],
               'epoch_seconds':time.perf_counter()-started,'checkpoint_updated':bool(improved)}
        history.append(entry)
        print(json.dumps(entry,indent=2))
    (output_dir/f'{args.model}_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print(f'Best checkpoint: {checkpoint_path}')
    print('Copy the checkpoint and matching metrics files into the local project checkpoints/ folder.')

if __name__=='__main__':
    main()
