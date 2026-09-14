import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import polaris as po
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import random

os.environ["HF_TOKEN"] = "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"
if os.path.exists("/mnt/weka/asafrastyan/polaris/data/"):
    os.environ["POLARIS_CACHE_DIR"] = "/mnt/weka/asafrastyan/polaris/data/"

polaris_to_internal_mapping = {
    "LOG_HLM_CLint": "HLM",
    "LOG_SOLUBILITY": "Sol",
    "LOG_MDR1-MDCK_ER": "MDR1",
    "LOG_RLM_CLint": "RLM",
    "LOG_HPPB": "hPPB",
    "LOG_RPPB": "rPPB"
}
internal_tasks = list(polaris_to_internal_mapping.values())

configs = {
    "170M": {
        "checkpoint": "yerevann/ChemLlama-170M",
        "weight_path": "/mnt/weka/asafrastyan/refined/polaris/multi-seed/weight_170/polaris_max_grid_170_seed_{seed}_best.pt",
        "n_unfreeze": 8, "mlp_hidden_size": 256, "mlp_layers": 2, "dropout": 0.1, "pooling": "last_token"
    },
    "380M": {
        "checkpoint": "yerevann/ChemLlama-380M",
        "weight_path": "/mnt/weka/asafrastyan/refined/polaris/multi-seed/weight_380/polaris_max_grid_380_seed_{seed}_best.pt",
        "n_unfreeze": 16, "mlp_hidden_size": 512, "mlp_layers": 2, "dropout": 0.1, "pooling": "last_token"
    },
    "1.3B": {
        "checkpoint": "yerevann/ChemLlama-1B",
        "weight_path": "/mnt/weka/asafrastyan/refined/polaris/multi-seed/weight_13/polaris_max_grid_13_seed_{seed}_best.pt",
        "n_unfreeze": 16, "mlp_hidden_size": 512, "mlp_layers": 2, "dropout": 0.1, "pooling": "last_token"
    },
    "3B": {
        "checkpoint": "yerevann/ChemLlama-3B",
        "weight_path": "/mnt/weka/asafrastyan/refined/polaris/multi-seed/weight_3/polaris_max_grid_3_seed_{seed}_best.pt",
        "n_unfreeze": 24, "mlp_hidden_size": 512, "mlp_layers": 2, "dropout": 0.1, "pooling": "last_token"
    }
}

seeds = [42, 81, 75, 95, 114]
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

def _mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(x.dtype)
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom

def _find_transformer_blocks(model: nn.Module):
    if hasattr(model, "layers"):
        return model.layers
    return None

def _freeze_all_but_last_n(model: nn.Module, unfreeze_last_n: int):
    for p in model.parameters():
        p.requires_grad = False

    blocks = _find_transformer_blocks(model)
    if blocks is None:
        print("WARNING: Could not locate transformer blocks to partially unfreeze; leaving base model fully frozen.")
        return

    if unfreeze_last_n <= 0:
        return

    for block in list(blocks)[-unfreeze_last_n:]:
        for p in block.parameters():
            p.requires_grad = True

    if hasattr(model, "norm"):
        for p in model.norm.parameters():
            p.requires_grad = True

class ChemlacticaMultiTaskRegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        tasks: list[str],
        pooling: str,
        mlp_hidden_size: int,
        mlp_layers: int,
        dropout: float,
        unfreeze_last_n: int,
        tokenizer_len: int
    ):
        super().__init__()
        self.tasks = tasks
        self.pooling = pooling

        self.backbone = AutoModel.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
        self.backbone.resize_token_embeddings(tokenizer_len)
        _freeze_all_but_last_n(self.backbone, unfreeze_last_n=unfreeze_last_n)

        base_hidden = int(self.backbone.config.hidden_size)
        if pooling == "attn":
            self.pool = AttnPool(base_hidden, dropout=dropout)
        elif pooling in ("mean", "last_token"):
            self.pool = None
        else:
            raise ValueError(f"Unknown pooling='{pooling}'")

        mlp = []
        in_dim = base_hidden
        for _ in range(int(mlp_layers)):
            mlp.append(nn.Linear(in_dim, mlp_hidden_size))
            mlp.append(nn.GELU())
            mlp.append(nn.Dropout(dropout))
            in_dim = mlp_hidden_size
        self.shared_mlp = nn.Sequential(*mlp) if len(mlp) > 0 else nn.Identity()

        self.heads = nn.ModuleDict({t: nn.Linear(in_dim, 1) for t in tasks})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state
        if self.pooling == "attn":
            pooled = self.pool(x, attention_mask=attention_mask)
        elif self.pooling == "last_token":
            batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
            last_token_idx = attention_mask.sum(1).long() - 1
            pooled = x[batch_idx, last_token_idx, :]
        else:
            pooled = _mean_pool(x, attention_mask=attention_mask)

        z = self.shared_mlp(pooled)
        preds = {t: self.heads[t](z).squeeze(-1) for t in self.tasks}
        return preds

def run_inference():
    print("Initializing Polaris Benchmark environment...")
    benchmark = po.load_benchmark("polaris/adme-fang-r-1")
    _, test_split = benchmark.get_train_test_split()
    test_smiles = test_split.inputs
    num_samples = len(test_smiles)
    
    results = []

    for size_name, config in configs.items():
        print(f"\n========== Processing Size: {size_name} ==========")
        
        tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"], token=os.environ.get("HF_TOKEN"))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
        tokenizer.add_special_tokens({"additional_special_tokens": ["[START_SMILES]", "[END_SMILES]"]})
        
        for seed in seeds:
            print(f"--- Loading seed {seed} ---")
            weight_file = config["weight_path"].format(seed=seed)
            if not os.path.exists(weight_file):
                print(f"Weight file not found: {weight_file}")
                continue
                
            model = ChemlacticaMultiTaskRegressor(
                model_name=config["checkpoint"],
                tasks=internal_tasks,
                pooling=config["pooling"],
                mlp_hidden_size=config["mlp_hidden_size"],
                mlp_layers=config["mlp_layers"],
                dropout=config["dropout"],
                unfreeze_last_n=config["n_unfreeze"],
                tokenizer_len=len(tokenizer)
            ).to(device)
            
            model.load_state_dict(torch.load(weight_file, map_location=device)["state_dict"])
            model.eval()
            
            with torch.no_grad():
                for i in tqdm(range(0, num_samples, batch_size), desc=f"Inference {size_name} Seed {seed}"):
                    batch_smiles = test_smiles[i:i + batch_size]
                    formatted_texts = [f"[START_SMILES] {smi} [END_SMILES]" for smi in batch_smiles]
                    encodings = tokenizer(
                        formatted_texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
                    )
                    input_ids = encodings["input_ids"].to(device)
                    attention_mask = encodings["attention_mask"].to(device)

                    with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        preds_dict = model(input_ids, attention_mask)
                        
                        for polaris_label in benchmark.target_cols:
                            internal_task_name = polaris_to_internal_mapping.get(polaris_label)
                            preds = preds_dict[internal_task_name].detach().cpu().to(torch.float32).numpy().tolist()
                            
                            for smi, p in zip(batch_smiles, preds):
                                results.append({
                                    "Size": size_name,
                                    "Seed": seed,
                                    "SMILES": smi,
                                    "Endpoint": polaris_label,
                                    "Predicted_Value": p
                                })
                                
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    df_results = pd.DataFrame(results)
    out_path = "polaris_test_predictions_all_sizes.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nAll predictions saved to {out_path}")

if __name__ == "__main__":
    run_inference()
