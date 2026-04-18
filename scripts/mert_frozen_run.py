from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, Wav2Vec2FeatureExtractor

PROJECT_ROOT = Path(
    os.environ.get("MMR_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_sis.config import (  # noqa: E402
    DEFAULT_MERT_NAME,
    MERT_FROZEN_CHECKPOINT_FILE,
    MERT_FROZEN_RANKINGS_FILE,
    MERT_FROZEN_RESULTS_FILE,
    REQUIRED_K_VALUES,
)
from music_sis.data.loaders import build_split_dataloaders  # noqa: E402
from music_sis.data.mert_collate import MertBatch  # noqa: E402
from music_sis.data.split_manifests import load_split_dataframes  # noqa: E402
from music_sis.eval.io import finalize_metrics_table, write_retrieval_artifacts  # noqa: E402
from music_sis.eval.retrieval import (  # noqa: E402
    build_protocol_eval_context,
    build_rankings_cosine,
    build_rankings_with_scores_df,
    evaluate_recall_at_k_from_rankings,
)


def load_frozen_mert(model_name: str = DEFAULT_MERT_NAME) -> AutoModel:
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _pool_hidden_states(outputs: object) -> torch.Tensor:
    """
    Convert model output to utterance-level embeddings.
    Uses time-average on last_hidden_state: [B, T, D] -> [B, D]
    """
    hidden = outputs.last_hidden_state
    return hidden.mean(dim=1)


def cache_split_embeddings(
    model: AutoModel,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader, start=1):
            batch = batch  # type: MertBatch
            inputs = batch.input_values.to(device)
            mask = batch.attention_mask.to(device)
            outputs = model(input_values=inputs, attention_mask=mask, output_hidden_states=False)
            emb = _pool_hidden_states(outputs).detach().cpu().numpy().astype(np.float32)

            for j in range(len(batch.clip_ids)):
                rows.append(
                    {
                        "clip_id": batch.clip_ids[j],
                        "track_id": batch.track_ids[j],
                        "label": batch.labels[j],
                        "embedding": emb[j],
                    }
                )
            if i % 50 == 0:
                print(f"[cache:{split_name}] processed {i * dataloader.batch_size} clips...")
    out = pd.DataFrame(rows)
    print(f"[cache:{split_name}] total cached clips: {len(out)}")
    return out


class CachedEmbeddingDataset(Dataset):
    def __init__(self, emb_df: pd.DataFrame):
        required = {"clip_id", "track_id", "label", "embedding"}
        missing = required.difference(emb_df.columns)
        if missing:
            raise ValueError(f"cached embedding dataframe missing columns: {sorted(missing)}")
        self.df = emb_df.reset_index(drop=True).copy()
        tracks = sorted(self.df["track_id"].unique().tolist())
        self.track_to_idx = {t: i for i, t in enumerate(tracks)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = torch.from_numpy(row["embedding"]).float()
        y = self.track_to_idx[row["track_id"]]
        return x, y


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss with multi-positive pairs by track_id label.
    """
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature

    device = z.device
    bsz = z.size(0)
    eye = torch.eye(bsz, dtype=torch.bool, device=device)
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = labels_eq & (~eye)
    logits_mask = ~eye

    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(sim) * logits_mask.float()
    log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / (positive_count.float() + 1e-12)
    loss = -mean_log_prob_pos[valid].mean()
    return loss


def project_embeddings_df(head: nn.Module, emb_df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    head.eval()
    out_rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for row in emb_df.itertuples(index=False):
            x = torch.from_numpy(row.embedding).float().to(device).unsqueeze(0)
            z = head(x).squeeze(0)
            z = F.normalize(z, dim=0).cpu().numpy().astype(np.float32)
            out_rows.append(
                {
                    "clip_id": row.clip_id,
                    "track_id": row.track_id,
                    "label": row.label,
                    "embedding": z,
                }
            )
    return pd.DataFrame(out_rows)


def evaluate_projected_split(
    head: nn.Module,
    emb_df: pd.DataFrame,
    k_values: Tuple[int, ...],
    device: torch.device,
) -> pd.DataFrame:
    projected_df = project_embeddings_df(head=head, emb_df=emb_df, device=device)
    split_meta = projected_df[["clip_id", "track_id", "label"]].copy()
    eval_context = build_protocol_eval_context(split_meta, k_values=k_values)
    rankings = build_rankings_cosine(
        query_embeddings_df=projected_df,
        candidate_embeddings_df=projected_df,
        exclude_self_clip=eval_context["exclude_self_clip"],
    )
    metrics = evaluate_recall_at_k_from_rankings(rankings_by_query=rankings, eval_context=eval_context)
    return metrics


@dataclass
class BestCheckpoint:
    step: int
    val_r1: float
    val_r10: float
    state_dict: Dict[str, torch.Tensor]


def _select_metric(metrics_df: pd.DataFrame, task: str, k: int) -> float:
    row = metrics_df[metrics_df["task"] == task]
    if row.empty:
        raise ValueError(f"task {task} not found in metrics table")
    return float(row.iloc[0][f"R@{k}"])


def train_retrieval_head(
    train_emb_df: pd.DataFrame,
    val_emb_df: pd.DataFrame,
    device: torch.device,
    out_dim: int = 256,
    batch_size: int = 256,
    epochs: int = 15,
    lr: float = 1e-3,
    temperature: float = 0.07,
    eval_every_steps: int = 250,
) -> BestCheckpoint:
    train_ds = CachedEmbeddingDataset(train_emb_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    in_dim = int(train_emb_df.iloc[0]["embedding"].shape[0])
    head = nn.Linear(in_dim, out_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    best = BestCheckpoint(
        step=0,
        val_r1=-1.0,
        val_r10=-1.0,
        state_dict=copy.deepcopy(head.state_dict()),
    )

    global_step = 0
    for epoch in range(1, epochs + 1):
        head.train()
        losses: List[float] = []
        for xb, yb in train_loader:
            global_step += 1
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            z = head(xb)
            loss = supervised_contrastive_loss(z, yb, temperature=temperature)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            if global_step % eval_every_steps == 0:
                val_metrics = evaluate_projected_split(
                    head=head,
                    emb_df=val_emb_df,
                    k_values=REQUIRED_K_VALUES,
                    device=device,
                )
                val_r10 = _select_metric(val_metrics, task="same_track", k=10)
                val_r1 = _select_metric(val_metrics, task="same_track", k=1)
                print(
                    f"[val@step {global_step}] same_track R@1={val_r1:.4f} R@10={val_r10:.4f}"
                )
                is_better = (
                    (val_r10 > best.val_r10)
                    or (val_r10 == best.val_r10 and val_r1 > best.val_r1)
                    or (val_r10 == best.val_r10 and val_r1 == best.val_r1 and global_step < best.step)
                )
                if is_better:
                    best = BestCheckpoint(
                        step=global_step,
                        val_r1=val_r1,
                        val_r10=val_r10,
                        state_dict=copy.deepcopy(head.state_dict()),
                    )

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[epoch {epoch}] train loss={epoch_loss:.6f}")

        val_metrics = evaluate_projected_split(
            head=head,
            emb_df=val_emb_df,
            k_values=REQUIRED_K_VALUES,
            device=device,
        )
        val_r10 = _select_metric(val_metrics, task="same_track", k=10)
        val_r1 = _select_metric(val_metrics, task="same_track", k=1)
        print(f"[val@end epoch {epoch}] same_track R@1={val_r1:.4f} R@10={val_r10:.4f}")
        is_better = (
            (val_r10 > best.val_r10)
            or (val_r10 == best.val_r10 and val_r1 > best.val_r1)
            or (val_r10 == best.val_r10 and val_r1 == best.val_r1 and global_step < best.step)
        )
        if is_better:
            best = BestCheckpoint(
                step=global_step,
                val_r1=val_r1,
                val_r10=val_r10,
                state_dict=copy.deepcopy(head.state_dict()),
            )

    print(
        f"[best] step={best.step} same_track val R@1={best.val_r1:.4f} R@10={best.val_r10:.4f}"
    )
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 Block 3: Frozen MERT retrieval run.")
    p.add_argument("--batch-size-audio", type=int, default=8)
    p.add_argument("--batch-size-head", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--eval-every-steps", type=int, default=250)
    p.add_argument("--max-seconds", type=float, default=5.0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    split_frames = load_split_dataframes()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        DEFAULT_MERT_NAME, trust_remote_code=True
    )
    dataloaders = build_split_dataloaders(
        split_frames=split_frames,
        processor=processor,
        batch_size=args.batch_size_audio,
        max_seconds=args.max_seconds,
    )

    model = load_frozen_mert(DEFAULT_MERT_NAME).to(device)

    print("Caching frozen MERT embeddings...")
    train_emb_df = cache_split_embeddings(model, dataloaders["train"], device=device, split_name="train")
    val_emb_df = cache_split_embeddings(model, dataloaders["val"], device=device, split_name="val")
    test_emb_df = cache_split_embeddings(model, dataloaders["test"], device=device, split_name="test")

    print("Training retrieval head on cached embeddings...")
    best = train_retrieval_head(
        train_emb_df=train_emb_df,
        val_emb_df=val_emb_df,
        device=device,
        out_dim=args.head_dim,
        batch_size=args.batch_size_head,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        eval_every_steps=args.eval_every_steps,
    )

    head = nn.Linear(int(train_emb_df.iloc[0]["embedding"].shape[0]), args.head_dim).to(device)
    head.load_state_dict(best.state_dict)
    head.eval()

    torch.save(
        {
            "step": best.step,
            "val_r1": best.val_r1,
            "val_r10": best.val_r10,
            "head_state_dict": best.state_dict,
            "head_dim": args.head_dim,
        },
        MERT_FROZEN_CHECKPOINT_FILE,
    )
    print(f"Saved best frozen head checkpoint: {MERT_FROZEN_CHECKPOINT_FILE}")

    print("Evaluating best checkpoint on test split...")
    projected_test_df = project_embeddings_df(head=head, emb_df=test_emb_df, device=device)
    test_meta = projected_test_df[["clip_id", "track_id", "label"]].copy()
    eval_context = build_protocol_eval_context(test_meta, k_values=REQUIRED_K_VALUES)
    rankings = build_rankings_cosine(
        query_embeddings_df=projected_test_df,
        candidate_embeddings_df=projected_test_df,
        exclude_self_clip=eval_context["exclude_self_clip"],
    )
    rankings_df = build_rankings_with_scores_df(
        query_embeddings_df=projected_test_df,
        candidate_embeddings_df=projected_test_df,
        exclude_self_clip=eval_context["exclude_self_clip"],
    )
    raw_metrics_df = evaluate_recall_at_k_from_rankings(rankings_by_query=rankings, eval_context=eval_context)
    metrics_df = finalize_metrics_table(raw_metrics_df, run_name="mert_frozen")
    write_retrieval_artifacts(
        metrics_df=metrics_df,
        rankings_df=rankings_df,
        metrics_path=MERT_FROZEN_RESULTS_FILE,
        rankings_path=MERT_FROZEN_RANKINGS_FILE,
    )
    print(f"Saved frozen metrics to: {MERT_FROZEN_RESULTS_FILE}")
    print(f"Saved frozen rankings to: {MERT_FROZEN_RANKINGS_FILE}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
