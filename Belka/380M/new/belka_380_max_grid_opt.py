import os
import glob
import warnings
warnings.filterwarnings("ignore")
import time
import random
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import dask.dataframe as dd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import multiprocessing
from rdkit import Chem
from torch.utils.data import DataLoader, IterableDataset, Dataset
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelAUROC
from torchmetrics import MeanMetric
import hashlib

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.pop("WANDB_REQUIRE_SERVICE", None)
WANDB_BASE = "/mnt/weka/asafrastyan/wandb"
WANDB_CACHE = os.path.join(WANDB_BASE, "cache")
WANDB_LOGS = os.path.join(WANDB_BASE, "logs")
os.makedirs(WANDB_CACHE, exist_ok=True)
os.makedirs(WANDB_LOGS, exist_ok=True)
os.environ["WANDB_CACHE_DIR"] = WANDB_CACHE
os.environ["WANDB_DIR"] = WANDB_LOGS
os.environ["HF_TOKEN"] = "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"
HF_TOKEN       = os.environ["HF_TOKEN"]
MODEL_NAME     = "yerevann/ChemLlama-380M"
TOKENIZER_NAME = "yerevann/ChemLlama-380M"
SEED           = 42
RAW_PARQUET    = "/mnt/weka/asafrastyan/belka/data/train.parquet"
DATA_DIR       = "/mnt/weka/asafrastyan/belka/data/processed_full"
CKPT_DIR       = "/mnt/weka/asafrastyan/refined/belka/380/weights" 
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR,  exist_ok=True)

# Global tracker for best validation score across trials
GLOBAL_BEST_VAL_AP = -1.0

CFG = {
    "batch_size":   128,
    "dropout":      0.25,
    "head_lr":      1e-4,
    "backbone_lr":  1e-5,  
    "weight_decay": 0.05,
    "warmup_ratio": 0.1,
    "focal_gamma":  2.0,
    "mlp_hidden_size": 512,
    "pooling":      "last_token",
    "epochs":       1,
    "patience":     35,
    "unfreeze_last_n": 6, 
}

SENTINEL       = os.path.join(DATA_DIR, ".ready")
MODEL_SENTINEL = os.path.join(CKPT_DIR, ".models_ready_max_grid")

class AttnPool(nn.Module):
    def __init__(self, in_dim: int, dropout: float):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(x).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

def _mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(x.dtype)
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom

def canon_single(s):
    try:
        return Chem.CanonSmiles(s.replace("[Dy]", "[H]"))
    except:
        return s

def canon_chunk(chunk):
    return [canon_single(x) for x in chunk]

def parallel_canon_map(smiles_list, num_workers=32):
    chunks = np.array_split(smiles_list, num_workers)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(num_workers) as pool:
        results = pool.map(canon_chunk, chunks)
    flat_results = np.concatenate(results)
    return dict(zip(smiles_list, flat_results))

def prepare_data():
    if not os.path.exists(SENTINEL):
        raise FileNotFoundError(
            f"Prepared training data sentinel {SENTINEL} not found! To avoid collision, "
            "belka_380_max_grid.py runs in read-only mode and will not regenerate data. "
            "Please ensure the data is prepared first using the original script."
        )
    print("Training data verified (SENTINEL found). Ready to proceed.")

class TrainDataset(IterableDataset):
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.epoch     = 0
        self.rank      = int(os.environ.get("RANK", 0))
        self.world     = int(os.environ.get("WORLD_SIZE", 1))

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
        files = files[self.rank :: self.world]

        wi        = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi else 0
        n_workers = wi.num_workers if wi else 1
        files     = files[worker_id :: n_workers]

        if not files:
            return
        rng = random.Random((SEED + self.epoch * 10_000 + worker_id) % 2**32)

        while True:
            rng.shuffle(files)
            for f in files:
                pf     = pq.ParquetFile(f)
                groups = list(range(pf.num_row_groups))
                rng.shuffle(groups)
                for g in groups:
                    t   = pf.read_row_group(g)
                    sm  = t["smiles"].to_numpy()
                    sub = t["subset"].to_numpy()
                    # Filter indices: train set (subset == 0) and 10% sample (hash % 10 == 0)
                    idx = [
                        i for i, s in enumerate(sm)
                        if sub[i] == 0 and int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % 10 == 0
                    ]
                    if len(idx) == 0:
                        continue
                    b4  = t["BRD4"].to_numpy()
                    hs  = t["HSA"].to_numpy()
                    se  = t["sEH"].to_numpy()
                    rng.shuffle(idx)
                    for i in idx:
                        yield self._enc(sm[i], b4[i], hs[i], se[i])

    def _enc(self, s, a, b, c):
        prompt = f"[START_SMILES]{s}[END_SMILES]"
        e = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids":      e["input_ids"].squeeze(0),
            "attention_mask": e["attention_mask"].squeeze(0),
            "labels":         torch.tensor([a, b, c], dtype=torch.float),
        }

class ValDataset(Dataset):
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.rows      = []
        print("Loading validation set into memory …")
        for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet"))):
            t   = pq.read_table(f)
            sm_list = t["smiles"].to_pylist()
            b4_list = t["BRD4"].to_pylist()
            hs_list = t["HSA"].to_pylist()
            se_list = t["sEH"].to_pylist()
            sub_list = t["subset"].to_pylist()
            
            # Filter rows: val set (subset == 1) and downsample to ~10% of val (~95k samples)
            for i, s_str in enumerate(sm_list):
                if sub_list[i] == 1 and int(hashlib.md5(s_str.encode('utf-8')).hexdigest(), 16) % 10 == 0:
                    self.rows.append((
                        s_str,
                        int(b4_list[i]),
                        int(hs_list[i]),
                        int(se_list[i]),
                    ))
        print(f"Validation set: {len(self.rows):,} molecules")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        s, a, b, c = self.rows[idx]
        prompt = f"[START_SMILES]{s}[END_SMILES]"
        e = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids":      e["input_ids"].squeeze(0),
            "attention_mask": e["attention_mask"].squeeze(0),
            "labels":         torch.tensor([a, b, c], dtype=torch.float),
        }

class ReseedCallback(pl.Callback):
    def __init__(self, dataset):
        self.ds = dataset

    def on_train_epoch_start(self, trainer, pl_module):
        self.ds.set_epoch(trainer.current_epoch)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        pos = torch.tensor([0.005, 0.005, 0.005])
        self.register_buffer("wp", 1.0 / pos.sqrt())
        self.register_buffer("wn", 1.0 / (1 - pos).sqrt())
        self.gamma = gamma

    def forward(self, x, y):
        y_smooth = y * 0.9 + 0.05
        bce   = F.binary_cross_entropy_with_logits(x, y_smooth, reduction="none")
        pt    = torch.exp(-bce)
        alpha = torch.where(y == 1, self.wp, self.wn)
        alpha = alpha / alpha.mean()
        return (alpha * (1 - pt) ** self.gamma * bce).mean()

class BelkaModel(pl.LightningModule):
    def __init__(self, vocab: int, num_train_samples: int):
        super().__init__()
        self.save_hyperparameters()
        self.num_train_samples = num_train_samples

        self.backbone = AutoModel.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        self.backbone.resize_token_embeddings(vocab)
        for p in self.backbone.parameters():
            p.requires_grad = False

        n = CFG.get("n_unfreeze", CFG.get("unfreeze_last_n", 4))
        if n > 0:
            for layer in self.backbone.layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        dim = self.backbone.config.hidden_size
        self.pooling = CFG["pooling"]
        if self.pooling == "attn":
            self.pool = AttnPool(dim, dropout=CFG["dropout"])
        elif self.pooling == "mean":
            self.pool = None

        mlp_hidden_size = CFG["mlp_hidden_size"]
        self.head = nn.Sequential(
            nn.Dropout(CFG["dropout"]),
            nn.Linear(dim, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(mlp_hidden_size, 3),
        )
        self.loss    = FocalLoss(CFG["focal_gamma"])
        self.tr_ap   = MultilabelAveragePrecision(num_labels=3, average="macro")
        self.tr_auc  = MultilabelAUROC(num_labels=3, average="macro")
        self.val_ap  = MultilabelAveragePrecision(num_labels=3, average="macro")
        self.val_auc = MultilabelAUROC(num_labels=3, average="macro")
        self.train_loss_avg = MeanMetric()

    def forward(self, ids, mask):
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.pooling == "attn":
            pooled = self.pool(h, attention_mask=mask)
        elif self.pooling == "last_token":
            batch_idx = torch.arange(ids.shape[0], device=ids.device)
            last_token_idx = mask.sum(1).long() - 1
            pooled = h[batch_idx, last_token_idx, :]
        else:
            pooled = _mean_pool(h, attention_mask=mask)

        return self.head(pooled)

    def training_step(self, batch, _):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss   = self.loss(logits, batch["labels"])
        preds  = logits.sigmoid()
        self.tr_ap.update(preds,  batch["labels"].long())
        self.tr_auc.update(preds, batch["labels"].long())
        self.train_loss_avg.update(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=False,
                 prog_bar=True, sync_dist=False)
        self.log("train_AP",  self.tr_ap,  on_step=False, on_epoch=True)
        self.log("train_AUC", self.tr_auc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss   = self.loss(logits, batch["labels"])
        preds  = logits.sigmoid()
        self.val_ap.update(preds,  batch["labels"].long())
        self.val_auc.update(preds, batch["labels"].long())
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("val_AP",  self.val_ap,  on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_AUC", self.val_auc, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            train_loss_val = self.train_loss_avg.compute()
            self.log("train_loss_val", train_loss_val, sync_dist=True)
            self.train_loss_avg.reset()

    def configure_optimizers(self):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params     = list(self.head.parameters())

        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": CFG["backbone_lr"],
             "weight_decay": CFG["weight_decay"]},
            {"params": head_params,     "lr": CFG["head_lr"],
             "weight_decay": CFG["weight_decay"]},
        ])

        world = max(self.trainer.world_size, 1)
        total = (self.num_train_samples // (CFG["batch_size"] * world * 4)) * CFG["epochs"]
        sch   = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=int(total * CFG["warmup_ratio"]),
            num_training_steps=total,
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "step"}}

def train_trial(tokenizer):
    pl.seed_everything(SEED)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    num_train_samples = 0
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet"))):
        t = pq.read_table(f, columns=["smiles", "subset"])
        sm_list = t["smiles"].to_pylist()
        sub_list = t["subset"].to_pylist()
        num_train_samples += sum(1 for i, s in enumerate(sm_list) if sub_list[i] == 0 and int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % 10 == 0)
    print(f"Number of train samples (subsampled): {num_train_samples:,}")

    train_batches = num_train_samples // (CFG["batch_size"] * world_size)

    def custom_collate(batch):
        from torch.nn.utils.rnn import pad_sequence
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    train_ds = TrainDataset(tokenizer, max_len=128)
    train_dl = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        collate_fn=custom_collate
    )
    val_dl = DataLoader(
        ValDataset(tokenizer, max_len=128),
        batch_size=CFG["batch_size"],
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=custom_collate
    )
    
    logger = WandbLogger(
        project="chemlactica-belka-thesis",
        save_dir=WANDB_LOGS,
        log_model=False,
    )

    global GLOBAL_BEST_VAL_AP

    model = BelkaModel(vocab=len(tokenizer), num_train_samples=num_train_samples)
    
    reseed_cb = ReseedCallback(train_ds)
    
    ckpt_cb = ModelCheckpoint(
        monitor="val_AP", 
        dirpath=CKPT_DIR,
        filename="380_max_grid_trial", 
        save_top_k=1, 
        mode="max"
    )

    strategy = DDPStrategy(
        find_unused_parameters=True,
    ) if world_size > 1 else "auto"

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=CFG["epochs"],
        logger=logger,
        callbacks=[reseed_cb, ckpt_cb],
        limit_train_batches=train_batches,
        val_check_interval=5000,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        accumulate_grad_batches=4,
    )
    trainer.fit(
        model, 
        train_dataloaders=train_dl, 
        val_dataloaders=val_dl
    )

    if trainer.global_rank == 0 and ckpt_cb.best_model_path:
        current_score = ckpt_cb.best_model_score.item()
        if current_score > GLOBAL_BEST_VAL_AP:
            GLOBAL_BEST_VAL_AP = current_score
            sd = torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False).get("state_dict", {})
            out = os.path.join(CKPT_DIR, "380_max_grid.pt")
            torch.save(sd, out)
            print(f"New global best weights saved to {out} (Score: {current_score:.5f})")

def main():
    pl.seed_everything(SEED)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://"
            )

    if local_rank == 0:
        print("Downloading tokenizer and backbone to HF cache …")
        AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=HF_TOKEN)
        AutoModel.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        open(MODEL_SENTINEL, "w").close()
        print("Models cached.")
    else:
        print(f"[rank {local_rank}] waiting for model cache …")
        while not os.path.exists(MODEL_SENTINEL):
            time.sleep(2)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]}
    )
    
    if local_rank == 0:
        prepare_data()
    else:
        print(f"[rank {local_rank}] waiting for data …")
        while not os.path.exists(SENTINEL):
            time.sleep(2)

    if local_rank == 0:
        sweep_config = {
            'method': 'bayes',
            'name': 'belka_380_max_grid_opt',
            'metric': {
                'goal': 'minimize', 
                'name': 'val_loss'
            },
            'parameters': {
                'head_lr': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-4,
                    'max': 1e-2
                },
                'backbone_lr': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-3
                },
                'weight_decay': {
                    'values': [0.001, 0.05, 0.1, 0.2]
                },
                'mlp_hidden_size': {
                    'values': [512, 1024]
                },
                'pooling': {
                    'values': ["last_token", "attn"]
                },
                'n_unfreeze': {
                    'values': [4, 8, 16]
                },
                'focal_gamma': {
                    'values': [0, 2]
                }
            }
        }

        sweep_id_file = os.path.join(CKPT_DIR, ".sweep_id_380_opt")
        if os.path.exists(sweep_id_file):
            with open(sweep_id_file, "r") as f:
                sweep_id = f.read().strip()
            print(f"Resuming existing sweep with ID: {sweep_id}")
        else:
            sweep_id = wandb.sweep(sweep_config, project="chemlactica-belka-thesis")
            with open(sweep_id_file, "w") as f:
                f.write(sweep_id)
            print(f"Created new sweep with ID: {sweep_id}")

        def sweep_train():
            run = wandb.init()
            config = {
                "head_lr": wandb.config.head_lr,
                "backbone_lr": wandb.config.backbone_lr,
                "weight_decay": wandb.config.weight_decay,
                "mlp_hidden_size": wandb.config.mlp_hidden_size,
                "pooling": wandb.config.pooling,
                "n_unfreeze": wandb.config.n_unfreeze,
                "focal_gamma": wandb.config.focal_gamma,
            }
            
            signal = [True, config]
            if world_size > 1:
                torch.distributed.broadcast_object_list(signal, src=0)
                
            CFG.update(config)
            train_trial(tokenizer)

        wandb.agent(sweep_id, project="chemlactica-belka-thesis", entity="artur_safrastyan-american-university-of-armenia", function=sweep_train, count=20)

        signal = [False, None]
        if world_size > 1:
            torch.distributed.broadcast_object_list(signal, src=0)
            
        print("Sweep sequence completed.")
    else:
        while True:
            signal = [None, None]
            torch.distributed.broadcast_object_list(signal, src=0)
            if not signal[0]: 
                break
                
            config = signal[1]
            CFG.update(config)
            
            train_trial(tokenizer)


if __name__ == "__main__":
    main()
