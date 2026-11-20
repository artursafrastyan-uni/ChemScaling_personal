import os
import glob
import warnings
warnings.filterwarnings("ignore")
import time
import random
import gc
from datetime import timedelta
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import multiprocessing
from rdkit import Chem
from tqdm import tqdm
from torch.utils.data import DataLoader, IterableDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torchmetrics import MeanMetric

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.pop("WANDB_REQUIRE_SERVICE", None)
WANDB_BASE = "/path/to/wandb"
WANDB_CACHE = os.path.join(WANDB_BASE, "cache")
WANDB_LOGS = "/path/to/logs"
os.makedirs(WANDB_CACHE, exist_ok=True)
os.makedirs(WANDB_LOGS, exist_ok=True)
os.environ["WANDB_CACHE_DIR"] = WANDB_CACHE
os.environ["WANDB_DIR"] = WANDB_LOGS
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "<YOUR_HF_TOKEN>")
HF_TOKEN       = os.environ["HF_TOKEN"]
MODEL_NAME     = "<MODEL_NAME_OR_PATH>"
TOKENIZER_NAME = "<TOKENIZER_NAME_OR_PATH>"
SEED           = 75
RAW_PARQUET    = "/path/to/data/train.parquet"
DATA_DIR       = "/path/to/data/processed"
CKPT_DIR       = "/path/to/checkpoints"
TEST_PATH      = "/path/to/data/test.parquet"
SUB_PATH       = f"/path/to/outputs/submission_{SEED}.csv"
FINAL_WEIGHTS  = os.path.join(CKPT_DIR, f"170_full_{SEED}.pt")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Optimal Hyperparameters derived from the best sweep run (radiant-sweep-17)
CFG = {
    "batch_size":        128,
    "dropout":           0.25,
    "head_lr":           0.00017086901249025303,
    "backbone_lr":       0.0004278901069818482,
    "weight_decay":      0.2,
    "warmup_ratio":      0.1,
    "focal_gamma":       2.0,
    "mlp_hidden_size":   1024,
    "pooling":           "last_token",
    "epochs":            1,
    "unfreeze_last_n":   8,
    "n_unfreeze":        8,
    "vocab":             128258,
    "num_train_samples": 8968083,
}

SENTINEL       = os.path.join(DATA_DIR, ".ready")
MODEL_SENTINEL = os.path.join(CKPT_DIR, f".models_ready_170_full_{SEED}")

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

def prepare_data():
    if not os.path.exists(SENTINEL):
        raise FileNotFoundError(
            f"Prepared training data sentinel {SENTINEL} not found! "
            "Please ensure data is prepared in DATA_DIR first."
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
                    b4  = t["BRD4"].to_numpy()
                    hs  = t["HSA"].to_numpy()
                    se  = t["sEH"].to_numpy()
                    idx = list(range(len(sm)))
                    rng.shuffle(idx)
                    for i in idx:
                        yield self._enc(sm[i], b4[i], hs[i], se[i])

    def _enc(self, s, a, b, c):
        prompt = f"[START_SMILES]{s}[END_SMILES]"
        e = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len, 
            return_tensors="pt",
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

class SaveWeightsCallback(pl.Callback):
    def __init__(self, save_path):
        self.save_path = save_path

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Save an intermediate checkpoint every 5,000 steps on rank 0
        if trainer.global_rank == 0 and trainer.global_step > 0 and trainer.global_step % 5000 == 0:
            step_path = self.save_path.replace(".pt", f"_step_{trainer.global_step}.pt")
            sd = pl_module.state_dict()
            keys_to_remove = [k for k in sd.keys() if 'loss' in k or 'train_' in k]
            for k in keys_to_remove:
                if k in sd:
                    del sd[k]
            torch.save(sd, step_path)
            print(f"[Step {trainer.global_step}] Intermediate checkpoint saved to: {step_path}")

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            sd = pl_module.state_dict()
            keys_to_remove = [k for k in sd.keys() if 'loss' in k or 'train_' in k]
            for k in keys_to_remove:
                if k in sd:
                    del sd[k]
            torch.save(sd, self.save_path)
            print(f"[on_train_end] Final model weights saved successfully to: {self.save_path}")

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
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.backbone.resize_token_embeddings(vocab)
        for p in self.backbone.parameters():
            p.requires_grad = False

        n = CFG.get("unfreeze_last_n", 16)
        if n > 0:
            for layer in self.backbone.layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        dim = self.backbone.config.hidden_size
        self.pooling = CFG["pooling"]

        mlp_hidden_size = CFG["mlp_hidden_size"]
        self.head = nn.Sequential(
            nn.Dropout(CFG["dropout"]),
            nn.Linear(dim, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(mlp_hidden_size, 3),
        )
        self.loss = FocalLoss(CFG["focal_gamma"])
        self.train_loss_avg = MeanMetric()

    def forward(self, ids, mask):
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.pooling == "last_token":
            batch_idx = torch.arange(ids.shape[0], device=ids.device)
            last_token_idx = mask.sum(1).long() - 1
            pooled = h[batch_idx, last_token_idx, :]
        else:
            pooled = _mean_pool(h, attention_mask=mask)
        return self.head(pooled)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss(logits, batch["labels"])
        self.train_loss_avg.update(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        return loss

    def on_train_epoch_end(self):
        train_loss_epoch = self.train_loss_avg.compute()
        self.log("train_loss_epoch", train_loss_epoch, sync_dist=True)
        self.train_loss_avg.reset()

    def configure_optimizers(self):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())

        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": CFG["backbone_lr"], "weight_decay": CFG["weight_decay"]},
            {"params": head_params, "lr": CFG["head_lr"], "weight_decay": CFG["weight_decay"]},
        ])

        world = max(self.trainer.world_size, 1)
        total = (self.num_train_samples // (CFG["batch_size"] * world * 4)) * CFG["epochs"]
        sch = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=int(total * CFG["warmup_ratio"]),
            num_training_steps=total,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

class TestDataset(Dataset):
    def __init__(self, sm_list, tokenizer, max_len=128):
        self.sm_list   = sm_list
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.sm_list)

    def __getitem__(self, idx):
        s = self.sm_list[idx]
        prompt = f"[START_SMILES]{s}[END_SMILES]"
        e = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len, 
            return_tensors="pt",
        )
        return {
            "input_ids":      e["input_ids"].squeeze(0),
            "attention_mask": e["attention_mask"].squeeze(0),
        }

def test_collate(batch, pad_token_id):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def train_full(tokenizer):
    pl.seed_everything(SEED)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    num_train_samples = 0
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet"))):
        pf = pq.ParquetFile(f)
        num_train_samples += pf.metadata.num_rows
    print(f"Total training samples (100% Full Dataset): {num_train_samples:,}")

    train_batches = num_train_samples // (CFG["batch_size"] * world_size)

    def custom_collate(batch):
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
    
    logger = WandbLogger(
        project="chemlactica-belka-thesis",
        name=f"belka_170_full_training_{SEED}",
        save_dir=WANDB_LOGS,
        log_model=False,
    )

    model = BelkaModel(vocab=len(tokenizer), num_train_samples=num_train_samples)
    reseed_cb = ReseedCallback(train_ds)
    save_cb = SaveWeightsCallback(FINAL_WEIGHTS)

    strategy = DDPStrategy(
        find_unused_parameters=False,
        timeout=timedelta(minutes=60),
    ) if world_size > 1 else "auto"

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=CFG["epochs"],
        logger=logger,
        callbacks=[reseed_cb, save_cb],
        limit_train_batches=train_batches,
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        accumulate_grad_batches=4,
    )
    
    print("Starting Full BELKA training (100% dataset, no validation)...")
    trainer.fit(
        model, 
        train_dataloaders=train_dl
    )

    # Cleanly teardown DDP across all ranks
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    # Rank 0 executes test set inference
    if local_rank == 0:
        print(f"Full dataset training complete! Verified final weights at: {FINAL_WEIGHTS}")

        del model, trainer, train_dl, train_ds
        gc.collect()
        torch.cuda.empty_cache()

        # Run test inference on the test set
        if os.path.exists(TEST_PATH):
            print("Running test set inference with full-dataset trained model...")
            test_df = pd.read_parquet(TEST_PATH)
            unique_smiles = test_df['molecule_smiles'].unique()
            print(f"Total rows: {len(test_df):,}, Unique smiles: {len(unique_smiles):,}")

            print("Canonicalising smiles in parallel...")
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(16) as pool:
                canon_smiles = pool.map(canon_single, unique_smiles)

            inf_model = BelkaModel(vocab=len(tokenizer), num_train_samples=num_train_samples)
            inf_sd = torch.load(FINAL_WEIGHTS, map_location="cpu", weights_only=False)
            inf_model.load_state_dict(inf_sd, strict=False)
            inf_model.eval()
            if torch.cuda.is_available():
                inf_model.cuda()
                inf_model.to(torch.bfloat16)

            ds = TestDataset(canon_smiles, tokenizer, max_len=128)
            dl = DataLoader(
                ds,
                batch_size=512,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True,
                shuffle=False,
                collate_fn=lambda b: test_collate(b, tokenizer.pad_token_id)
            )

            all_preds = []
            with torch.inference_mode():
                for batch in tqdm(dl, desc="Predicting test set"):
                    if torch.cuda.is_available():
                        ids = batch["input_ids"].cuda()
                        mask = batch["attention_mask"].cuda()
                    else:
                        ids = batch["input_ids"]
                        mask = batch["attention_mask"]

                    logits = inf_model(ids, mask)
                    preds = torch.sigmoid(logits).float().cpu().numpy()
                    all_preds.append(preds)

            all_preds = np.concatenate(all_preds, axis=0)

            print("Merging predictions into submission format...")
            pred_df = pd.DataFrame({
                'molecule_smiles': unique_smiles,
                'BRD4': all_preds[:, 0],
                'HSA':  all_preds[:, 1],
                'sEH':  all_preds[:, 2],
            })

            test_df = test_df.merge(pred_df, on='molecule_smiles', how='left')
            test_df['binds'] = 0.0

            brd4_mask = test_df['protein_name'] == 'BRD4'
            hsa_mask  = test_df['protein_name'] == 'HSA'
            seh_mask  = test_df['protein_name'] == 'sEH'

            test_df.loc[brd4_mask, 'binds'] = test_df.loc[brd4_mask, 'BRD4']
            test_df.loc[hsa_mask,  'binds'] = test_df.loc[hsa_mask,  'HSA']
            test_df.loc[seh_mask,  'binds'] = test_df.loc[seh_mask,  'sEH']

            os.makedirs(os.path.dirname(SUB_PATH), exist_ok=True)
            sub = test_df[['id', 'binds']]
            sub.to_csv(SUB_PATH, index=False)
            print(f"Submission successfully saved to {SUB_PATH}")

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
                init_method="env://",
                timeout=timedelta(minutes=60),
            )

    if local_rank == 0:
        print("Downloading tokenizer and backbone to HF cache …")
        AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=HF_TOKEN)
        AutoModel.from_pretrained(MODEL_NAME, token=HF_TOKEN, dtype=torch.bfloat16)
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

    train_full(tokenizer)

if __name__ == "__main__":
    main()
