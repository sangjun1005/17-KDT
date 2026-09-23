# -*- coding: utf-8 -*-
"""
wbc.py — 백혈구(WBC) 4클래스 분류 공통 모듈
2조 프로젝트 / 딥러닝 수업 4-5장 코드 스타일을 다중분류로 확장한 것

노트북 4개(00~03)가 이 파일을 import 해서 쓴다.
공통 코드를 한 곳에 두는 이유
  1) 같은 train_model 을 네 번 복사하면 실험끼리 조건이 어긋난다
  2) 윈도우에서 DataLoader(num_workers>0) 는 클래스가 .py 안에 있어야 안전하다
"""

import os, time, json, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, balanced_accuracy_score)

# ----------------------------------------------------------------------------
# 0. 기본 설정
# ----------------------------------------------------------------------------
DATA_DIR   = './dataset2-master/images'      # TRAIN / TEST 폴더의 부모
ORIG_DIR   = './dataset-master'              # 원본 366장 + labels.csv (외부 검증용)
RESULT_DIR = './results'
CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']   # ImageFolder 알파벳 순
NUM_CLASSES = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 윈도우 + 주피터에서 워커가 말썽이면 0 으로 내린다
NUM_WORKERS = 4
EPOCH_BUDGET_SEC = 180          # 과제 조건: 1 에폭 최대 3분

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, 'preds'), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, 'ckpt'), exist_ok=True)


def set_seed(seed=42):
    """재현성. 가설검정에서 시드를 바꿔 반복할 것이므로 시드 관리가 중요하다."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True     # 입력 크기가 고정이라 켜두면 빨라진다


# ----------------------------------------------------------------------------
# 1. 전처리 프리셋
# ----------------------------------------------------------------------------
# 이 데이터셋의 특수 사정
#   - 이미 회전/이동 증강이 적용된 상태로 배포됐다(모서리에 검은 패딩이 보인다)
#   - 그래서 회전을 또 크게 주면 패딩 위에 패딩이 쌓인다
#   - 세포는 방향성이 없으므로 좌우/상하 반전은 안전하다 (X-ray 와 반대)
#   - 색(염색 색조)은 클래스 정보를 담고 있다. 호산구의 분홍 과립이 대표적이라
#     hue 를 크게 흔들면 오히려 신호를 지운다 -> 색 증강은 약하게만
PRESETS = {
    'none'    : dict(flip=False, affine=0,  color=0.0, crop=1.00, erase=0.0),
    'flip'    : dict(flip=True,  affine=0,  color=0.0, crop=1.00, erase=0.0),
    'geo'     : dict(flip=True,  affine=15, color=0.0, crop=1.00, erase=0.0),
    'crop'    : dict(flip=True,  affine=0,  color=0.0, crop=0.80, erase=0.0),   # 회전 패딩 제거
    'color'   : dict(flip=True,  affine=0,  color=0.2, crop=1.00, erase=0.0),
    'strong'  : dict(flip=True,  affine=15, color=0.2, crop=0.80, erase=0.25),
    'crop_geo': dict(flip=True,  affine=10, color=0.1, crop=0.80, erase=0.0),
}


def build_transforms(preset='flip', image_size=224):
    """(train_tf, eval_tf) 를 돌려준다. 검증/테스트에는 증강을 절대 걸지 않는다."""
    p = PRESETS[preset]
    crop_px = int(240 * p['crop'])            # 원본이 320x240 이므로 짧은 변 기준

    pre = []
    if p['crop'] < 1.0:
        pre.append(transforms.CenterCrop((crop_px, int(320 * p['crop']))))

    train_ops = pre + [transforms.Resize((image_size, image_size))]
    if p['flip']:
        train_ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    if p['affine'] > 0:
        train_ops += [transforms.RandomAffine(degrees=p['affine'],
                                              translate=(0.05, 0.05),
                                              scale=(0.9, 1.1))]
    if p['color'] > 0:
        c = p['color']
        train_ops += [transforms.ColorJitter(brightness=c, contrast=c,
                                             saturation=c * 0.5, hue=c * 0.1)]
    train_ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if p['erase'] > 0:
        train_ops += [transforms.RandomErasing(p=p['erase'], scale=(0.02, 0.10))]

    eval_ops = pre + [transforms.Resize((image_size, image_size)),
                      transforms.ToTensor(),
                      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


# ----------------------------------------------------------------------------
# 2. 데이터로더
# ----------------------------------------------------------------------------
def make_loaders(preset='flip', image_size=224, batch_size=64,
                 val_ratio=0.2, seed=42, data_dir=DATA_DIR, subset=None):
    """
    TRAIN 폴더를 층화분할해 train/val 을 만들고, TEST 폴더는 최종 평가에만 쓴다.
    같은 폴더를 두 번 여는 요령(증강 있는 것 / 없는 것)은 5장과 동일하다.
    subset: 속도 실험용으로 train 을 n장만 쓰고 싶을 때
    """
    train_tf, eval_tf = build_transforms(preset, image_size)

    train_full = datasets.ImageFolder(os.path.join(data_dir, 'TRAIN'), transform=train_tf)
    eval_full  = datasets.ImageFolder(os.path.join(data_dir, 'TRAIN'), transform=eval_tf)
    test_set   = datasets.ImageFolder(os.path.join(data_dir, 'TEST'),  transform=eval_tf)
    assert train_full.classes == CLASS_NAMES, train_full.classes

    # 층화분할: 클래스 비율을 train/val 에서 동일하게 유지한다
    targets = np.array(train_full.targets)
    rng = np.random.RandomState(seed)
    val_idx, train_idx = [], []
    for c in range(NUM_CLASSES):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = int(len(idx) * val_ratio)
        val_idx += idx[:n_val].tolist()
        train_idx += idx[n_val:].tolist()
    rng.shuffle(train_idx)
    if subset:
        train_idx = train_idx[:subset]

    train_set = Subset(train_full, train_idx)     # 증강 있음
    val_set   = Subset(eval_full,  val_idx)       # 증강 없음 (5장에서 지적된 버그를 고친 형태)

    kw = dict(num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'),
              persistent_workers=(NUM_WORKERS > 0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_set,   batch_size=batch_size * 2, shuffle=False, **kw)
    test_loader  = DataLoader(test_set,  batch_size=batch_size * 2, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------------
# 3. 모델
# ----------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """베이스라인 A — 3장에서 만든 스크래치 CNN 을 4클래스로.
    GAP + Linear 한 층이라 CAM(Class Activation Map) 을 그대로 뽑을 수 있는 구조다."""
    def __init__(self, num_classes=NUM_CLASSES, width=32):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(3, width), block(width, width * 2),
            block(width * 2, width * 4), block(width * 4, width * 8))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width * 8, num_classes)
        self.num_features = width * 8

    def forward_features(self, x):
        return self.features(x)                       # (B, C, H, W)  <- CAM 용

    def forward(self, x):
        f = self.forward_features(x)
        return self.head(self.pool(f).flatten(1))


class TimmNet(nn.Module):
    """전이학습 모델 — 5장의 timm 사용법 그대로.
    head 를 Linear 한 층으로 두면 GAP+Linear 구조가 되어 CAM 을 쓸 수 있다."""
    def __init__(self, backbone='resnet18', num_classes=NUM_CLASSES,
                 pretrained=True, freeze=False, dropout=0.0, probe_size=224):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool='')   # 풀링도 끈다
        # 채널 수는 더미 입력을 흘려서 확인한다.
        # backbone.num_features 를 그냥 믿으면 안 되는 모델이 있다(예: mobilenetv3 는
        # conv-head 이후 차원을 보고하지만 global_pool='' 로는 그 앞 특징맵이 나온다).
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 3, probe_size, probe_size))
            f = self._to_map(f)
        self.num_features = f.shape[1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.num_features, num_classes)
        if freeze:                                     # 4장 방식: 백본 고정, 헤드만 학습
            for p in self.backbone.parameters():
                p.requires_grad = False

    @staticmethod
    def _to_map(f):
        if f.dim() == 3:            # ViT 계열은 (B, N, C) 로 나온다 -> 정사각 격자로 되돌린다
            n = f.shape[1]; s = int(round(math.sqrt(n)))
            f = f[:, n - s * s:, :].transpose(1, 2).reshape(f.shape[0], -1, s, s)
        return f

    def forward_features(self, x):
        return self._to_map(self.backbone(x))

    def forward(self, x):
        f = self.forward_features(x)
        return self.head(self.drop(self.pool(f).flatten(1)))


def build_model(name='resnet18', pretrained=True, freeze=False, dropout=0.0, image_size=224):
    if name == 'simplecnn':
        return SimpleCNN()
    return TimmNet(name, pretrained=pretrained, freeze=freeze, dropout=dropout,
                   probe_size=image_size)


def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, trn


# ----------------------------------------------------------------------------
# 4. 평가 — 다중분류용
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion=None):
    """4장/5장 evaluate 를 다중분류로 확장.
    이진분류의 sigmoid/BCE 자리에 softmax/CrossEntropy 가 들어간다."""
    model.eval()
    losses, probs, trues = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)                              # (B, 4)  squeeze 하지 않는다
        if criterion is not None:
            losses.append(criterion(logits, yb).item() * len(yb))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        trues.append(yb.cpu().numpy())

    probs = np.concatenate(probs); trues = np.concatenate(trues)
    preds = probs.argmax(1)
    try:
        auc = roc_auc_score(trues, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
    return {
        'loss'      : (sum(losses) / len(trues)) if criterion is not None else float('nan'),
        'accuracy'  : accuracy_score(trues, preds),
        'balanced'  : balanced_accuracy_score(trues, preds),
        'macro_f1'  : f1_score(trues, preds, average='macro'),
        'macro_prec': precision_score(trues, preds, average='macro', zero_division=0),
        'macro_rec' : recall_score(trues, preds, average='macro', zero_division=0),
        'auc'       : auc,
        'probs': probs, 'trues': trues, 'preds': preds,
    }


# ----------------------------------------------------------------------------
# 5. 학습
# ----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=10, lr=3e-4,
                weight_decay=1e-4, label_smoothing=0.0, class_weight=None,
                ckpt_path=None, label='model', patience=4, amp=True,
                scheduler='plateau', monitor='macro_f1', verbose=True,
                epoch_budget=EPOCH_BUDGET_SEC):
    """
    5장 train_model 을 다중분류 + 조기종료 + AMP + 시간예산 확인으로 확장.
    monitor: 'macro_f1'(클수록 좋음) 또는 'loss'(작을수록 좋음) 로 체크포인트를 잡는다.
    """
    model = model.to(device)
    ckpt_path = ckpt_path or os.path.join(RESULT_DIR, 'ckpt', f'{label}.pt')

    cw = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if scheduler == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.3, patience=2, min_lr=1e-7)
    use_amp = bool(amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = {k: [] for k in ['loss', 'accuracy', 'val_loss', 'val_accuracy',
                               'val_macro_f1', 'val_auc', 'lr', 'sec']}
    best = -float('inf'); best_epoch = 0; bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, run_correct, n = 0.0, 0, 0
        t0 = time.time()

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

            run_loss += loss.item() * len(yb)
            run_correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)

        tr_loss, tr_acc = run_loss / n, run_correct / n
        m = evaluate(model, val_loader, criterion)
        sec = time.time() - t0

        score = m['macro_f1'] if monitor == 'macro_f1' else -m['loss']
        if scheduler == 'cosine': sched.step()
        else:                     sched.step(m['loss'])

        for k, v in [('loss', tr_loss), ('accuracy', tr_acc), ('val_loss', m['loss']),
                     ('val_accuracy', m['accuracy']), ('val_macro_f1', m['macro_f1']),
                     ('val_auc', m['auc']), ('lr', optimizer.param_groups[0]['lr']), ('sec', sec)]:
            history[k].append(v)

        star = ''
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            torch.save(model.state_dict(), ckpt_path); star = '저장'
        else:
            bad += 1

        if verbose:
            over = ' ⚠3분초과' if sec > epoch_budget else ''
            print(f"[{label}] {epoch:2d}/{epochs} loss {tr_loss:.4f} val_loss {m['loss']:.4f} "
                  f"val_acc {m['accuracy']:.4f} val_f1 {m['macro_f1']:.4f} val_auc {m['auc']:.4f} "
                  f"lr {optimizer.param_groups[0]['lr']:.1e} ({sec:.0f}s){over} {star}")

        if bad >= patience:
            if verbose: print(f'  조기종료: {patience} 에폭 동안 개선 없음')
            break

    history['best_epoch'] = best_epoch
    history['best_score'] = best if monitor == 'macro_f1' else -best
    history['ckpt_path'] = ckpt_path
    history['epoch_sec_mean'] = float(np.mean(history['sec']))
    if verbose:
        print(f"best {monitor} = {history['best_score']:.4f} @epoch {best_epoch} -> {ckpt_path}")
        print(f"에폭당 평균 {history['epoch_sec_mean']:.0f}s (예산 {epoch_budget}s)")
    return history


def load_ckpt(model, ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)


# ----------------------------------------------------------------------------
# 6. 속도 프로브 — "1 에폭 3분" 조건을 맞추기 위한 사전 측정
# ----------------------------------------------------------------------------
def speed_probe(model_name, image_size=224, batch_size=64, n_batches=12,
                pretrained=False, freeze=False, n_train=7966):
    """실제 학습 없이 배치 몇 개만 돌려 초당 처리량을 재고, 1 에폭 예상시간을 계산한다."""
    set_seed(0)
    model = build_model(model_name, pretrained=pretrained, freeze=freeze,
                        image_size=image_size).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    crit = nn.CrossEntropyLoss()
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)

    model.train()
    for i in range(n_batches + 3):
        if i == 3:                                     # 워밍업 3 배치 제외
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if device.type == 'cuda': torch.cuda.synchronize()
    dt = time.time() - t0

    ips = n_batches * batch_size / dt                  # images per second
    est = n_train / ips * 1.25                         # 검증/데이터로딩 오버헤드 25% 가산
    tot, trn = count_params(model)
    del model, opt; torch.cuda.empty_cache() if device.type == 'cuda' else None
    return dict(model=model_name, image_size=image_size, batch_size=batch_size,
                params_M=round(tot / 1e6, 2), img_per_sec=round(ips, 1),
                est_epoch_sec=round(est, 1), fits=est <= EPOCH_BUDGET_SEC)


# ----------------------------------------------------------------------------
# 7. 실험 기록 — 나중에 가설검정을 하려면 예측을 남겨야 한다
# ----------------------------------------------------------------------------
def log_run(run_id, params, val_metrics=None, test_metrics=None, history=None, csv_path=None):
    """runs.csv 에 한 줄 추가하고, 예측 확률을 npz 로 저장한다.
    McNemar 검정은 '같은 테스트셋에 대한 두 모델의 예측'이 있어야 하므로 반드시 남긴다."""
    import csv
    csv_path = csv_path or os.path.join(RESULT_DIR, 'runs.csv')
    row = dict(run_id=run_id, **{k: v for k, v in params.items()})
    for tag, m in [('val', val_metrics), ('test', test_metrics)]:
        if m is None: continue
        for k in ['accuracy', 'balanced', 'macro_f1', 'macro_prec', 'macro_rec', 'auc', 'loss']:
            row[f'{tag}_{k}'] = round(float(m[k]), 5)
    if history is not None:
        row['best_epoch'] = history.get('best_epoch')
        row['epoch_sec'] = round(history.get('epoch_sec_mean', float('nan')), 1)

    save = {}
    if val_metrics is not None:
        save.update(val_probs=val_metrics['probs'], val_trues=val_metrics['trues'])
    if test_metrics is not None:
        save.update(test_probs=test_metrics['probs'], test_trues=test_metrics['trues'])
    if save:
        np.savez_compressed(os.path.join(RESULT_DIR, 'preds', f'{run_id}.npz'), **save)

    exists = os.path.exists(csv_path)
    old = []
    if exists:
        with open(csv_path, newline='', encoding='utf-8-sig') as f:
            old = list(csv.DictReader(f))
    keys = sorted(set().union(*[set(r) for r in old + [row]])) if old else list(row)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in old + [row]: w.writerow(r)
    return row


def load_preds(run_id, split='test'):
    d = np.load(os.path.join(RESULT_DIR, 'preds', f'{run_id}.npz'))
    return d[f'{split}_probs'], d[f'{split}_trues']


def runs_table(csv_path=None):
    import pandas as pd
    csv_path = csv_path or os.path.join(RESULT_DIR, 'runs.csv')
    return pd.read_csv(csv_path, encoding='utf-8-sig') if os.path.exists(csv_path) else None


# ----------------------------------------------------------------------------
# 8. CAM / Grad-CAM — "모델이 어디를 보고 판단했는가"
# ----------------------------------------------------------------------------
# 두 가지를 모두 제공한다.
#   CAM      : GAP + Linear 구조에서만 성립. 헤드 가중치를 그대로 쓰므로 근사가 없다.
#   Grad-CAM : 구조에 상관없이 성립. 마지막 특징맵에 대한 기울기를 가중치로 쓴다.
# 우리 모델(SimpleCNN, TimmNet)은 둘 다 GAP+Linear 이므로 CAM 이 정공법이고,
# Grad-CAM 은 검증용으로 함께 그려서 두 결과가 일치하는지 본다.

def _to_numpy_img(x):
    """정규화된 텐서를 다시 0~1 이미지로 되돌린다(시각화용)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (x.detach().cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


@torch.no_grad()
def cam(model, x, class_idx=None):
    """
    CAM: 마지막 특징맵 f_k(h,w) 에 헤드 가중치 w_{c,k} 를 곱해 더한다.
        M_c(h,w) = sum_k w_{c,k} * f_k(h,w)
    x: (3,H,W) 또는 (B,3,H,W)
    반환: (B,h,w) 0~1 정규화된 맵, 예측 클래스
    """
    model.eval()
    if x.dim() == 3: x = x.unsqueeze(0)
    x = x.to(device)
    feats = model.forward_features(x)                     # (B,C,h,w)
    logits = model.head(model.pool(feats).flatten(1))
    pred = logits.argmax(1)
    idx = pred if class_idx is None else torch.full_like(pred, class_idx)
    w = model.head.weight[idx]                            # (B,C)
    maps = (feats * w[:, :, None, None]).sum(1)           # (B,h,w)
    maps = F.relu(maps)
    maps = maps / (maps.flatten(1).max(1).values[:, None, None] + 1e-8)
    return maps.cpu().numpy(), pred.cpu().numpy(), torch.softmax(logits, 1).cpu().numpy()


def grad_cam(model, x, class_idx=None):
    """Grad-CAM: 특징맵에 대한 기울기를 공간평균해 채널 가중치로 쓴다. hook 으로 구현."""
    model.eval()
    if x.dim() == 3: x = x.unsqueeze(0)
    x = x.to(device).requires_grad_(False)

    store = {}
    feats = model.forward_features(x)
    feats.retain_grad()
    logits = model.head(model.pool(feats).flatten(1))
    pred = logits.argmax(1)
    idx = pred if class_idx is None else torch.full_like(pred, class_idx)
    score = logits.gather(1, idx[:, None]).sum()
    model.zero_grad(set_to_none=True)
    score.backward()

    w = feats.grad.mean(dim=(2, 3))                       # (B,C)
    maps = F.relu((feats * w[:, :, None, None]).sum(1))
    maps = maps / (maps.flatten(1).max(1).values[:, None, None] + 1e-8)
    return maps.detach().cpu().numpy(), pred.detach().cpu().numpy()


def show_cam(model, dataset, indices, method='cam', class_names=CLASS_NAMES,
             alpha=0.45, cols=4, title=None, save_path=None):
    """CAM 오버레이 그리드. dataset 은 (이미지텐서, 라벨) 을 주는 것이면 무엇이든 된다."""
    import matplotlib.pyplot as plt
    n = len(indices); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, indices):
        x, y = dataset[i]
        if method == 'cam':
            m, pred, prob = cam(model, x)
            conf = prob[0][pred[0]]
        else:
            m, pred = grad_cam(model, x); conf = float('nan')
        img = _to_numpy_img(x)
        heat = np.array(torch.nn.functional.interpolate(
            torch.tensor(m)[None], size=img.shape[:2], mode='bilinear',
            align_corners=False)[0, 0])
        ax.imshow(img); ax.imshow(heat, cmap='jet', alpha=alpha)
        ok = '○' if pred[0] == y else '✗'
        ax.set_title(f'{ok} 정답 {class_names[y][:4]} / 예측 {class_names[pred[0]][:4]}'
                     + (f' {conf:.2f}' if conf == conf else ''), fontsize=9)
        ax.axis('off')
    for ax in axes[n:]: ax.axis('off')
    if title: fig.suptitle(title)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def cam_on_cell_ratio(model, dataset, indices, method='cam'):
    """
    CAM 정량 검증: '모델이 본 곳'이 실제 백혈구(진한 보라 핵) 위인가?
    HSV 에서 보라색 핵을 잡아 마스크를 만들고, CAM 질량 중 마스크 안에 든 비율을 잰다.
    무작위로 보면 마스크 면적비와 비슷하고, 세포를 보고 있으면 그보다 훨씬 크다.
    """
    ratios, areas = [], []
    for i in indices:
        x, y = dataset[i]
        m, pred = (cam(model, x)[:2] if method == 'cam' else grad_cam(model, x))
        img = _to_numpy_img(x)
        hsv = _rgb_to_hsv(img)
        # 보라/파랑 계열(핵) : hue 0.6~0.85, 채도 0.15 이상
        mask = ((hsv[..., 0] > 0.60) & (hsv[..., 0] < 0.88) & (hsv[..., 1] > 0.15)).astype(np.float32)
        heat = np.array(torch.nn.functional.interpolate(
            torch.tensor(m)[None], size=img.shape[:2], mode='bilinear', align_corners=False)[0, 0])
        if heat.sum() <= 0 or mask.sum() <= 0: continue
        ratios.append(float((heat * mask).sum() / heat.sum()))
        areas.append(float(mask.mean()))
    return np.array(ratios), np.array(areas)


def _rgb_to_hsv(img):
    """opencv 없이 쓰는 RGB->HSV (0~1 스케일)."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx, mn = img.max(-1), img.min(-1)
    d = mx - mn + 1e-8
    h = np.zeros_like(mx)
    h = np.where(mx == r, ((g - b) / d) % 6, h)
    h = np.where(mx == g, (b - r) / d + 2, h)
    h = np.where(mx == b, (r - g) / d + 4, h)
    h = h / 6.0
    s = np.where(mx > 0, d / (mx + 1e-8), 0)
    return np.stack([h, s, mx], -1)


# ----------------------------------------------------------------------------
# 9. 가설검정
# ----------------------------------------------------------------------------
from scipy import stats


def mcnemar_test(y_true, pred_a, pred_b, exact=None, correction=True):
    """
    같은 테스트셋에서 두 모델을 비교하는 표준 검정.
      H0: 두 모델의 오분류율이 같다 (b == c)
      H1: 다르다 (양측)
    b = A만 맞힘, c = B만 맞힘. 두 모델이 같은 데이터를 봤으므로 독립표본 검정을 쓰면 안 된다.
    b+c 가 작으면(<25) 정확 이항검정, 크면 연속성 보정 카이제곱을 쓴다.
    """
    a_ok = (np.asarray(pred_a) == np.asarray(y_true))
    b_ok = (np.asarray(pred_b) == np.asarray(y_true))
    n01 = int(np.sum(a_ok & ~b_ok))      # A만 맞음
    n10 = int(np.sum(~a_ok & b_ok))      # B만 맞음
    n11 = int(np.sum(a_ok & b_ok)); n00 = int(np.sum(~a_ok & ~b_ok))
    n = n01 + n10
    use_exact = (n < 25) if exact is None else exact
    if n == 0:
        stat, p = 0.0, 1.0
    elif use_exact:
        stat = float(min(n01, n10))
        p = float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    else:
        num = (abs(n01 - n10) - (1 if correction else 0)) ** 2
        stat = num / n
        p = float(stats.chi2.sf(stat, df=1))
    return dict(table=[[n11, n01], [n10, n00]], b_only_A=n01, c_only_B=n10,
                n_discordant=n, statistic=float(stat), p_value=p,
                method='exact binomial' if use_exact else 'chi2 (continuity corrected)',
                acc_A=float(a_ok.mean()), acc_B=float(b_ok.mean()),
                acc_diff=float(a_ok.mean() - b_ok.mean()))


def bootstrap_ci(y_true, y_pred, metric='accuracy', n_boot=2000, alpha=0.05, seed=42):
    """테스트셋 재표본으로 성능 지표의 신뢰구간을 구한다. '99.1%' 같은 숫자에 오차범위를 붙이는 용도."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.RandomState(seed); n = len(y_true); vals = []
    fn = {'accuracy': accuracy_score,
          'macro_f1': lambda t, p: f1_score(t, p, average='macro'),
          'balanced': balanced_accuracy_score}[metric]
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2: continue
        vals.append(fn(y_true[idx], y_pred[idx]))
    vals = np.array(vals)
    return dict(point=float(fn(y_true, y_pred)),
                lo=float(np.percentile(vals, 100 * alpha / 2)),
                hi=float(np.percentile(vals, 100 * (1 - alpha / 2))), n_boot=len(vals))


def paired_test(scores_a, scores_b, name_a='A', name_b='B', alternative='two-sided'):
    """
    시드를 바꿔 여러 번 돌린 점수쌍(같은 시드끼리 짝지음)을 비교한다.
      H0: 두 조건의 평균 성능이 같다
    정규성을 가정한 대응표본 t검정과, 가정이 약한 윌콕슨 부호순위검정을 함께 보고한다.
    효과크기(Cohen's d) 도 같이 봐야 한다 — n 이 크면 사소한 차이도 유의해지기 때문이다.
    """
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    d = a - b
    t, p_t = stats.ttest_rel(a, b, alternative=alternative)
    try:
        w, p_w = stats.wilcoxon(a, b, alternative=alternative)
    except ValueError:
        w, p_w = float('nan'), float('nan')
    sd = d.std(ddof=1)
    return dict(n=len(a), mean_a=float(a.mean()), mean_b=float(b.mean()),
                mean_diff=float(d.mean()), sd_diff=float(sd),
                t=float(t), p_ttest=float(p_t), w=float(w), p_wilcoxon=float(p_w),
                cohens_d=float(d.mean() / sd) if sd > 0 else float('nan'),
                ci95=(float(d.mean() - 1.96 * sd / np.sqrt(len(d))),
                      float(d.mean() + 1.96 * sd / np.sqrt(len(d)))),
                name_a=name_a, name_b=name_b)


def two_proportion_test(k1, n1, k2, n2):
    """
    독립인 두 데이터셋에서의 정확도 비교 (예: 증강 TEST vs 원본 366장).
      H0: 두 모집단의 정확도가 같다
    표본이 다르므로 McNemar 가 아니라 두 비율 검정을 쓴다.
    """
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p_val = 2 * stats.norm.sf(abs(z))
    se_d = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return dict(p1=p1, p2=p2, diff=p1 - p2, z=z, p_value=float(p_val),
                ci95=(p1 - p2 - 1.96 * se_d, p1 - p2 + 1.96 * se_d))


def holm_correction(pvals, names=None, alpha=0.05):
    """여러 번 검정하면 우연히 유의해질 확률이 커진다. Holm-Bonferroni 로 보정한다."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        run = max(run, val)
        adj[i] = min(run, 1.0)
    names = names or [f'test{i}' for i in range(m)]
    return [dict(name=nm, p_raw=float(pv), p_holm=float(pa), reject=bool(pa < alpha))
            for nm, pv, pa in zip(names, p, adj)]


def permutation_test_acc(y_true, pred_a, pred_b, n_perm=10000, seed=0):
    """McNemar 의 비모수 대안. 짝지은 정오표를 무작위로 뒤집어 귀무분포를 만든다."""
    a_ok = (np.asarray(pred_a) == np.asarray(y_true)).astype(int)
    b_ok = (np.asarray(pred_b) == np.asarray(y_true)).astype(int)
    obs = a_ok.mean() - b_ok.mean()
    d = a_ok - b_ok; rng = np.random.RandomState(seed)
    null = np.array([(d * rng.choice([-1, 1], len(d))).mean() for _ in range(n_perm)])
    return dict(obs_diff=float(obs), p_value=float((np.abs(null) >= abs(obs) - 1e-12).mean()))


# ----------------------------------------------------------------------------
# 10. 시각화
# ----------------------------------------------------------------------------
def plot_history(history, title='', save_path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(history['loss'], label='train'); ax[0].plot(history['val_loss'], label='val')
    ax[0].set_title(f'{title} loss'); ax[0].set_xlabel('epoch'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(history['accuracy'], label='train'); ax[1].plot(history['val_accuracy'], label='val')
    ax[1].set_title('accuracy'); ax[1].set_xlabel('epoch'); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(history['val_macro_f1'], label='val macro-F1', color='tab:green')
    ax[2].set_title('macro F1'); ax[2].set_xlabel('epoch'); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def plot_confusion(y_true, y_pred, class_names=CLASS_NAMES, normalize=False,
                   title='혼동행렬', save_path=None):
    """5장 plot_confusion_matrix 의 지적사항(정규화를 imshow 앞으로)을 반영한 버전."""
    import matplotlib.pyplot as plt, itertools
    cm = confusion_matrix(y_true, y_pred)
    shown = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(shown, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title); fig.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(ticks); ax.set_yticklabels(class_names)
    thresh = shown.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        txt = f'{shown[i, j]:.2f}' if normalize else f'{cm[i, j]}'
        ax.text(j, i, txt, ha='center', color='white' if shown[i, j] > thresh else 'black')
    ax.set_ylabel('정답'); ax.set_xlabel('예측')
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def use_korean_font():
    """윈도우 맑은고딕. 그래프 한글이 깨지면 이걸 부른다."""
    import matplotlib.pyplot as plt, matplotlib
    for f in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']:
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams['font.family'] = f; break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    return plt.rcParams['font.family']


# ----------------------------------------------------------------------------
# 11. 편의 함수
# ----------------------------------------------------------------------------
def already_run(run_id, csv_path=None):
    """이미 돌린 실험이면 True. 노트북을 다시 돌릴 때 학습을 건너뛰기 위한 것."""
    import csv as _csv
    csv_path = csv_path or os.path.join(RESULT_DIR, 'runs.csv')
    if not os.path.exists(csv_path): return False
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        return any(r.get('run_id') == run_id for r in _csv.DictReader(f))


def make_eval_loader(root, image_size=224, batch_size=64, crop=1.0, center_crop_ratio=None):
    """
    평가 전용 로더(증강 없음). 외부 검증셋(원본 366장)처럼 폴더만 있는 경우에 쓴다.
    center_crop_ratio: 원본 640x480 은 dataset2 보다 시야가 넓다. 시야를 맞추려면 0.7 정도로 자른다.
    """
    ops = []
    if center_crop_ratio:
        ops.append(transforms.Lambda(lambda im: transforms.functional.center_crop(
            im, [int(im.size[1] * center_crop_ratio), int(im.size[0] * center_crop_ratio)])))
    ops += [transforms.Resize((image_size, image_size)), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    ds = datasets.ImageFolder(root, transform=transforms.Compose(ops))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), ds


def summarize(m, name='', digits=4):
    """평가 결과를 한 줄로."""
    return (f"{name:22s} acc {m['accuracy']:.{digits}f}  macroF1 {m['macro_f1']:.{digits}f}  "
            f"balanced {m['balanced']:.{digits}f}  AUC {m['auc']:.{digits}f}")


def class_report(m, class_names=CLASS_NAMES):
    import pandas as pd
    from sklearn.metrics import classification_report
    rep = classification_report(m['trues'], m['preds'], target_names=class_names,
                                output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.round(4)


def run_experiment(run_id, model_name='resnet18', preset='flip', image_size=224,
                   batch_size=64, lr=3e-4, epochs=10, freeze=False, pretrained=True,
                   dropout=0.0, weight_decay=1e-4, label_smoothing=0.0, seed=42,
                   subset=None, patience=4, monitor='macro_f1', scheduler='plateau',
                   eval_test=False, force=False, verbose=True, data_dir=None):
    """
    실험 한 건을 처음부터 끝까지 돌리고 결과를 남긴다.
    - 이미 runs.csv 에 있는 run_id 는 건너뛴다(force=True 로 강제 재실행).
    - eval_test 는 최종 모델에만 True 로 준다. 테스트셋을 보면서 튜닝하면 그 점수는 의미를 잃는다.
    """
    if already_run(run_id) and not force:
        if verbose: print(f'[skip] {run_id} 는 이미 runs.csv 에 있음')
        return None
    set_seed(seed)
    tr, va, te = make_loaders(preset=preset, image_size=image_size, batch_size=batch_size,
                              seed=42, subset=subset, data_dir=data_dir or DATA_DIR)
    model = build_model(model_name, pretrained=pretrained, freeze=freeze,
                        dropout=dropout, image_size=image_size)
    tot, trn = count_params(model)
    if verbose:
        print(f'=== {run_id} | {model_name} | preset={preset} size={image_size} bs={batch_size} '
              f'lr={lr:g} seed={seed} | 학습 파라미터 {trn:,}/{tot:,}')
    hist = train_model(model, tr, va, epochs=epochs, lr=lr, weight_decay=weight_decay,
                       label_smoothing=label_smoothing, patience=patience, monitor=monitor,
                       scheduler=scheduler, label=run_id, verbose=verbose)
    load_ckpt(model, hist['ckpt_path'])
    crit = nn.CrossEntropyLoss()
    mv = evaluate(model, va, crit)
    mt = evaluate(model, te, crit) if eval_test else None
    params = dict(model=model_name, preset=preset, image_size=image_size, batch_size=batch_size,
                  lr=lr, epochs=epochs, freeze=int(freeze), pretrained=int(pretrained),
                  dropout=dropout, weight_decay=weight_decay, label_smoothing=label_smoothing,
                  seed=seed, subset=subset or 0, params_M=round(tot / 1e6, 2))
    log_run(run_id, params, mv, mt, hist)
    if verbose:
        print(summarize(mv, '  VAL'))
        if mt: print(summarize(mt, '  TEST'))
    return dict(model=model, history=hist, val=mv, test=mt, params=params)


def load_run_model(run_id, csv_path=None):
    """runs.csv 에 기록된 설정으로 모델을 다시 만들고 저장된 가중치를 불러온다.
    (사전학습 가중치는 어차피 덮어쓰므로 pretrained=False 로 받는다 — 5장과 같은 요령)"""
    t = runs_table(csv_path)
    row = t[t.run_id == run_id]
    if len(row) == 0:
        raise KeyError(f'{run_id} 가 runs.csv 에 없다')
    row = row.iloc[0]
    model = build_model(str(row['model']), pretrained=False,
                        dropout=float(row.get('dropout', 0) or 0),
                        image_size=int(row['image_size']))
    ckpt = os.path.join(RESULT_DIR, 'ckpt', f'{run_id}.pt')
    load_ckpt(model, ckpt)
    return model, row


def eval_on_test(run_id, batch_size=None, save_as=None):
    """저장된 모델을 테스트셋에서 평가하고 예측을 npz 로 남긴다(McNemar 검정 재료)."""
    model, row = load_run_model(run_id)
    size = int(row['image_size'])
    bs = batch_size or int(row['batch_size'])
    _, _, te = make_loaders(preset='none', image_size=size, batch_size=bs)
    m = evaluate(model, te, nn.CrossEntropyLoss())
    np.savez_compressed(os.path.join(RESULT_DIR, 'preds', f'{save_as or run_id}_TEST.npz'),
                        test_probs=m['probs'], test_trues=m['trues'])
    return model, m


# ----------------------------------------------------------------------------
# 12. 노트북 사이에서 설정을 넘기는 통로
# ----------------------------------------------------------------------------
CFG_PATH = './config.json'
DEFAULT_CFG = dict(model_name='resnet18', preset='flip', image_size=224, batch_size=64,
                   lr=3e-4, weight_decay=1e-4, label_smoothing=0.0, dropout=0.0,
                   full_epochs=25)


def load_cfg(path=CFG_PATH):
    """노트북 02~05 가 하나씩 채워 넣은 최종 설정을 읽는다. 없으면 기본값."""
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(path):
        cfg.update(json.load(open(path, encoding='utf-8')))
    return cfg


def save_cfg(path=CFG_PATH, **kw):
    """바꾼 항목만 갱신해서 저장한다."""
    cfg = load_cfg(path); cfg.update(kw)
    json.dump(cfg, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('config.json 갱신:', json.dumps(kw, ensure_ascii=False))
    return cfg
