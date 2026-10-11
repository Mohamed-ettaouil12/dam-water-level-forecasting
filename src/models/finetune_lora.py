"""Fine-tuning Chronos avec LoRA (entraînement local GPU)."""

import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import DATA_PROCESSED, MLRUNS_DIR, MODELS_DIR, DATA_SOURCE, SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR = MODELS_DIR / "chronos-lora"
LORA_DIR.mkdir(parents=True, exist_ok=True)

MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6]
CONTEXT_LENGTH = 240  # Réduit pour s'adapter aux données (21 ans = 252 mois avant 2021)
MAX_EPOCHS = 3
LR = 1e-4
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q", "v"]


def get_gpu_info() -> dict:
    """Retourne les infos GPU."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "total_vram_gb": 0, "free_vram_gb": 0}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / 1e9
    allocated = torch.cuda.memory_allocated() / 1e9
    free = (props.total_memory - torch.cuda.memory_reserved()) / 1e9
    return {
        "name": props.name,
        "total_vram_gb": round(total, 1),
        "free_vram_gb": round(free, 1),
    }


def get_batch_size(free_vram_gb: float) -> int:
    """Déduit le batch size selon la VRAM libre."""
    if free_vram_gb >= 20:
        return 32
    elif free_vram_gb >= 10:
        return 16
    elif free_vram_gb >= 6:
        return 8
    elif free_vram_gb >= 3:
        return 4
    else:
        return 2


class ChronosDataset(Dataset):
    """Dataset pour l'entraînement Chronos (finetuning)."""

    def __init__(
        self,
        data: np.ndarray,
        context_length: int,
        prediction_length: int,
        stride: int = 1,
    ):
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride

        self.indices = []
        for i in range(0, len(data) - context_length - prediction_length + 1, stride):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        context = self.data[start:start + self.context_length]
        target = self.data[
            start + self.context_length:start + self.context_length + self.prediction_length
        ]
        return {
            "context": torch.tensor(context, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }


def prepare_training_data(
    df: pd.DataFrame,
    horizon: int,
    context_length: int = CONTEXT_LENGTH,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Prépare les données d'entraînement par série (avant 2021 uniquement)."""
    df = df.sort_values(["barrage", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year

    train_data = []

    for barrage in sorted(df["barrage"].unique()):
        bdf = df[df["barrage"] == barrage].copy()
        bdf = bdf[bdf["year"] < 2021].sort_values("date").reset_index(drop=True)

        if len(bdf) < CONTEXT_LENGTH + HORIZONS[-1]:
            continue

        series = bdf["niveau_pct"].values
        train_data.append((series, barrage))

    return train_data


def train_lora(
    pipeline,
    train_series_list: List[Tuple[np.ndarray, str]],
    horizon: int,
    epochs: int = MAX_EPOCHS,
    lr: float = LR,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict:
    """Fine-tune le modèle avec LoRA."""
    # Accéder au modèle T5 sous-jacent
    base_model = pipeline.model.model
    tokenizer = pipeline.tokenizer

    # Config LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    model.to(device)
    model.train()

    # Optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Scheduler
    num_training_steps = len(train_series_list) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    total_loss = 0
    step = 0

    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_series_list, desc=f"Epoch {epoch+1}/{epochs}")

        for series, barrage in pbar:
            if len(series) < CONTEXT_LENGTH + horizon:
                continue

            # Fenêtres glissantes
            for i in range(0, len(series) - CONTEXT_LENGTH - horizon + 1, 16):
                context = series[i:i + CONTEXT_LENGTH]
                target = series[i + CONTEXT_LENGTH:i + CONTEXT_LENGTH + horizon]

                # Tokenisation via le tokenizer Chronos
                context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
                target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

                token_ids, attention_mask, scale = pipeline.tokenizer.context_input_transform(context_tensor)
                target_ids, _, _ = pipeline.tokenizer.context_input_transform(target_tensor)

                token_ids = token_ids.to(device)
                attention_mask = attention_mask.to(device)
                target_ids = target_ids.to(device)

                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    # Forward
                    outputs = model(
                        input_ids=token_ids,
                        attention_mask=attention_mask,
                        labels=target_ids,
                    )
                    loss = outputs.loss

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

                total_loss += loss.item()
                epoch_loss += loss.item()
                step += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(1, len(train_series_list))
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Remplacer le modèle dans le pipeline
    pipeline.model.model = model
    return {"loss": total_loss / max(1, step), "steps": step}


def save_adapter(pipeline, path: Path):
    """Sauvegarde uniquement l'adaptateur LoRA."""
    # Le modèle PEFT est dans pipeline.model.model
    pipeline.model.model.save_pretrained(path)
    logger.info(f"Adaptateur LoRA sauvegardé : {path}")


def main(
    horizon: int = 1,
    epochs: int = MAX_EPOCHS,
    lr: float = LR,
    batch_size: int = None,
    device: str = "auto",
    seed: int = SEED,
) -> int:
    """Point d'entrée principal."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # GPU info
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gpu_info = get_gpu_info()
    logger.info(f"Device: {device}")
    logger.info(f"GPU: {gpu_info['name']}")
    logger.info(f"VRAM totale: {gpu_info['total_vram_gb']} GB")
    logger.info(f"VRAM libre: {gpu_info['free_vram_gb']} GB")

    # Batch size auto
    if batch_size is None:
        batch_size = get_batch_size(gpu_info["free_vram_gb"])
        logger.info(f"Batch size auto: {batch_size}")

    # Chargement données
    from src.config import DATA_PROCESSED
    from src.data.clean import load_raw_data

    df = load_raw_data(DATA_SOURCE)
    logger.info(f"Données chargées: {len(df)} lignes, {df['barrage'].nunique()} barrages")

    # Préparation données entraînement (avant 2021)
    train_data = prepare_training_data(df, horizon)
    logger.info(f"Séries d'entraînement: {len(train_data)}")

    if not train_data:
        logger.error("Aucune donnée d'entraînement disponible")
        return 1

    # Chargement pipeline
    from chronos import BaseChronosPipeline

    logger.info("Chargement modèle Chronos...")
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )

    # Entraînement
    logger.info(f"Début fine-tuning LoRA (horizon={horizon}, epochs={epochs})...")
    metrics = train_lora(
        pipeline,
        train_data,
        horizon=horizon,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )

    # Sauvegarde adaptateur
    adapter_path = LORA_DIR / f"horizon_{horizon}"
    adapter_path.mkdir(parents=True, exist_ok=True)
    save_adapter(pipeline, adapter_path)

    logger.info(f"Fine-tuning terminé. Loss finale: {metrics['loss']:.4f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Chronos avec LoRA")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 3, 6])
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    exit(main(
        horizon=args.horizon,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    ))