import os
import gc
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from tqdm import tqdm
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"
HF_TOKEN = os.environ["HF_TOKEN"]

# Model and Tokenizer aligned with belka_380_max_grid_opt.py
MODEL_NAME     = "yerevann/ChemLlama-380M"
TOKENIZER_NAME = "yerevann/ChemLlama-380M"

# Weight checkpoint and target paths (380_max_grid.pt contains super-sweep-6 weights with val_AP 0.36203)
CKPT_PATH = "/mnt/weka/asafrastyan/refined/belka/380/weights/380_max_grid.pt"
TEST_PATH = "/mnt/weka/asafrastyan/belka/data/test.parquet"
SUB_PATH  = "/mnt/weka/asafrastyan/refined/belka/380/submission_380_sub_10_max_grid_opt.csv"

# Hyperparameters
CFG = {
    "backbone_lr":       0.00013587292773745965,
    "batch_size":        64,
    "focal_gamma":       0,
    "head_lr":           0.0004523907726246295,
    "mlp_hidden_size":   1024,
    "n_unfreeze":         16,
    "num_train_samples": 8968083,
    "pooling":           "attn",
    "vocab":             128258,
    "weight_decay":      0.05,
    "dropout":           0.25,
}

class AttnPool(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.25):
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

class BelkaModel(pl.LightningModule):
    def __init__(self, vocab: int = 128258, mlp_hidden_size: int = 1024, pooling: str = "attn"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.backbone.resize_token_embeddings(vocab)
        
        dim = self.backbone.config.hidden_size
        self.pooling = pooling
        if self.pooling == "attn":
            self.pool = AttnPool(dim, dropout=CFG.get("dropout", 0.25))
        else:
            self.pool = None

        self.head = nn.Sequential(
            nn.Dropout(CFG.get("dropout", 0.25)),
            nn.Linear(dim, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(CFG.get("dropout", 0.25)),
            nn.Linear(mlp_hidden_size, 3),
        )

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

def canon_single(s):
    try:
        return Chem.CanonSmiles(s.replace("[Dy]", "[H]"))
    except:
        return s

class TestDataset(Dataset):
    def __init__(self, sm_list, tokenizer, max_len=256):
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

def main():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    print(f"Loading tokenizer from {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]}
    )

    print(f"Loading weights from {CKPT_PATH}...")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    # Filter out training/validation metrics
    keys_to_remove = [k for k in sd.keys() if 'loss' in k or 'tr_' in k or 'val_' in k]
    for k in keys_to_remove:
        del sd[k]

    # Dynamically verify head & pool architecture to guarantee shape compatibility
    ckpt_mlp = sd["head.1.weight"].shape[0] if "head.1.weight" in sd else CFG["mlp_hidden_size"]
    has_pool = any("pool" in k for k in sd)
    ckpt_pooling = "attn" if has_pool else "last_token"

    if ckpt_mlp != CFG["mlp_hidden_size"] or (ckpt_pooling == "attn") != (CFG["pooling"] == "attn"):
        print(f"Notice: Checkpoint layer dimensions ({ckpt_mlp} MLP, {ckpt_pooling} pooling) differ from default CFG ({CFG['mlp_hidden_size']} MLP, {CFG['pooling']} pooling).")
        print(f"Auto-aligning model architecture to match checkpoint weights ({ckpt_mlp} MLP, {ckpt_pooling}) for exact loading.")
        model_mlp = ckpt_mlp
        model_pooling = ckpt_pooling
    else:
        model_mlp = CFG["mlp_hidden_size"]
        model_pooling = CFG["pooling"]

    print("Initializing model architecture...")
    model = BelkaModel(vocab=CFG["vocab"], mlp_hidden_size=model_mlp, pooling=model_pooling)
    model.load_state_dict(sd, strict=False)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()
        model.to(torch.bfloat16)

    print("Loading test data...")
    test_df = pd.read_parquet(TEST_PATH)
    unique_smiles = test_df['molecule_smiles'].unique()
    print(f"Total rows: {len(test_df):,}, Unique smiles: {len(unique_smiles):,}")

    print("Canonicalising smiles in parallel...")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(16) as pool:
        canon_smiles = pool.map(canon_single, unique_smiles)

    ds = TestDataset(canon_smiles, tokenizer, max_len=256)
    dl = DataLoader(
        ds, 
        batch_size=CFG["batch_size"], 
        num_workers=8, 
        pin_memory=True, 
        prefetch_factor=4,
        persistent_workers=True,
        shuffle=False, 
        collate_fn=lambda b: test_collate(b, tokenizer.pad_token_id)
    )

    print("Running evaluation / inference (pure test evaluation)...")
    all_preds = []
    with torch.inference_mode():
        for batch in tqdm(dl, desc="Predicting test set"):
            if torch.cuda.is_available():
                ids = batch["input_ids"].cuda()
                mask = batch["attention_mask"].cuda()
            else:
                ids = batch["input_ids"]
                mask = batch["attention_mask"]

            logits = model(ids, mask)
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
    print(f"Evaluation complete! Submission successfully saved to {SUB_PATH}")

if __name__ == "__main__":
    main()
