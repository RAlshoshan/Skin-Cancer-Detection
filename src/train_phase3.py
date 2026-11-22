# -*- coding: utf-8 -*-
"""Phase 3: ResNet50 training using Phase 2's FROZEN patient-level split.

Place this file inside PROJECT/src/ and run from the PROJECT folder:
    py src/train_phase3.py --pilot   # 1 train batch + 1 validation batch; NO reported score
    py src/train_phase3.py           # real training (20 epochs max; early stopping)

Does NOT regenerate/swap splits and does NOT open the test set for evaluation.
Main optimization settings match released reference train_cnn_final.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

import config as C
import splits as S
import torch_pipeline as TP


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(device, pilot):
    split_file = C.split_path()
    if not split_file.exists():
        raise FileNotFoundError(f'Frozen split missing: {split_file}')
    patient_groups = S.load_split(split_file)
    S.assert_disjoint(patient_groups)
    meta = S.load_metadata()
    per = S.labels_for_split(meta, patient_groups)
    for name in ('train', 'val', 'test'):
        if len(per[name]) == 0:
            raise ValueError(f'{name} is empty: check split and metadata')
    if sum(len(per[k]) for k in ('train', 'val', 'test')) != len(meta):
        raise ValueError('Frozen split does not cover every metadata row exactly once')

    img_dirs = C.image_dirs()
    missing = [str(p) for p in img_dirs if not p.is_dir()]
    if missing:
        raise FileNotFoundError('Image directories not found: ' + ', '.join(missing))
    train_ds = TP.PADUFESDataset(img_dirs, per['train'], transform=TP.get_resnet50_transforms(train=True))
    val_ds = TP.PADUFESDataset(img_dirs, per['val'], transform=TP.get_resnet50_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                            num_workers=C.NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    count_positive = float(per['train']['is_cancer'].sum())
    count_negative = float(len(per['train']) - count_positive)
    if count_positive <= 0 or count_negative <= 0:
        raise ValueError('Training data must have both classes')
    pos_weight = torch.tensor([count_negative / (count_positive + 1e-6)], device=device)
    print(f"Frozen split: {split_file}", flush=True)
    print(f"Patients disjoint: PASS | train={len(train_ds)} val={len(val_ds)} "
          f"test={len(per['test'])} (test held out)", flush=True)
    print(f"Train cancer={int(count_positive)} non-cancer={int(count_negative)}; "
          f"pos_weight={pos_weight.item():.4f}", flush=True)
    return train_loader, val_loader, pos_weight, digest(split_file)


def train_epoch(model, loader, criterion, optimizer, device, limit=None):
    model.train()
    loss_sum, seen, batches = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
        optimizer.step()
        n = len(images)
        loss_sum += loss.item() * n
        seen += n
        batches += 1
        if limit is not None and batches >= limit:
            break
    if seen == 0:
        raise RuntimeError('Empty training loader')
    return loss_sum / seen, batches


@torch.no_grad()
def validation(model, loader, criterion, device, limit=None):
    model.eval()
    total_loss, seen, batches = 0.0, 0, 0
    true_labels, probs = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        n = len(images)
        total_loss += loss.item() * n
        seen += n
        batches += 1
        true_labels.append(labels.cpu().numpy().reshape(-1))
        probs.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
        if limit is not None and batches >= limit:
            break
    if seen == 0:
        raise RuntimeError('Empty validation loader')
    y = np.concatenate(true_labels)
    p = np.concatenate(probs)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan')
    pred = (p > 0.5).astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {'val_loss': total_loss / seen, 'val_auc': auc,
            'val_bal_acc': (sens + spec) / 2,
            'val_sensitivity': sens, 'val_specificity': spec}


def main():
    ap = argparse.ArgumentParser(description='ResNet50 Phase 3, frozen patient split')
    ap.add_argument('--pilot', action='store_true', help='one train batch + one val batch, no reportable metric')
    args = ap.parse_args()

    TP.seed_everything(C.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Mode: {"PILOT" if args.pilot else "FULL TRAINING"} | device: {device}', flush=True)
    train_loader, val_loader, pos_weight, split_sha = prepare(device, args.pilot)

    model = TP.build_resnet50(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    backbone = [p for n, p in model.named_parameters() if not n.startswith('fc.')]
    head = list(model.fc.parameters())
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': C.LR * C.BACKBONE_LR_MULT, 'weight_decay': C.BACKBONE_WD},
        {'params': head, 'lr': C.LR * C.HEAD_LR_MULT, 'weight_decay': C.HEAD_WD},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=C.SCHED_PATIENCE, factor=C.SCHED_FACTOR)

    if args.pilot:
        trloss, n = train_epoch(model, train_loader, criterion, optimizer, device, limit=1)
        val = validation(model, val_loader, criterion, device, limit=1)
        print(f'PILOT PASS: 1 training batch (loss={trloss:.4f}), '
              f'1 validation batch (loss={val["val_loss"]:.4f})', flush=True)
        print('Pilot metrics are NOT valid experiment results. No model was saved.', flush=True)
        return

    outdir = C.project_root() / 'results' / 'phase3' / 'resnet50'
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'run_config.json').write_text(json.dumps({
        'reference': 'Released train_cnn_final.py, ResNet50 binary baseline',
        'seed': C.SEED, 'batch_size': C.BATCH_SIZE, 'max_epochs': C.EPOCHS,
        'early_stop_patience': C.PATIENCE, 'min_delta': C.MIN_DELTA,
        'base_lr': C.LR, 'backbone_lr': C.LR*C.BACKBONE_LR_MULT,
        'head_lr': C.LR*C.HEAD_LR_MULT, 'backbone_wd': C.BACKBONE_WD,
        'head_wd': C.HEAD_WD, 'scheduler': 'ReduceLROnPlateau(val_loss)',
        'frozen_split_sha256': split_sha, 'device': str(device),
        'torch': torch.__version__,
        'note': 'Training uses reference optimization & transforms. Epoch-0 '
                'untrained evaluation and the reference plots are not reproduced here. '
                'Test set is reserved for final evaluation only.'
    }, indent=2), encoding='utf-8')
    metrics_file = outdir / 'train_history.csv'
    best_auc = -1.0
    early_stop_best = -1.0
    stale = 0
    with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
        fields = ['epoch', 'train_loss', 'val_loss', 'val_auc', 'val_bal_acc',
                  'val_sensitivity', 'val_specificity']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, C.EPOCHS + 1):
            print(f'\nEpoch {epoch}/{C.EPOCHS}...', flush=True)
            train_loss, batches = train_epoch(model, train_loader, criterion, optimizer, device)
            stats = validation(model, val_loader, criterion, device)
            scheduler.step(stats['val_loss'])
            row = {'epoch': epoch, 'train_loss': train_loss, **stats}
            writer.writerow(row)
            f.flush()
            print(f'train_loss={train_loss:.4f} val_loss={stats["val_loss"]:.4f} '
                  f'val_AUC={stats["val_auc"]:.4f} val_bal_acc={stats["val_bal_acc"]:.4f}', flush=True)
            auc = stats['val_auc']
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                torch.save({'model_state_dict': model.state_dict(),
                            'epoch': epoch, 'val_auc': best_auc,
                            'split_sha256': split_sha}, outdir / 'best_model.pth')
                print('Best model saved.', flush=True)
            if not np.isnan(auc) and auc > early_stop_best + C.MIN_DELTA:
                early_stop_best = auc
                stale = 0
            else:
                stale += 1
            if stale >= C.PATIENCE:
                print(f'Early stopping: {stale} epochs without val-AUC improvement.', flush=True)
                break
    print(f'\nTRAINING FINISHED | best validation AUC={best_auc:.4f}', flush=True)
    print(f'Results: {outdir}', flush=True)
    print('Test set has NOT been evaluated.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
