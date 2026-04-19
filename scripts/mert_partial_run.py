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
from transformers import AutoModel, Wav2Vec2FeatureExtractor

PROJECT_ROOT = Path(
    os.environ.get("MMR_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_sis.config import (  # noqa: E402
    DEFAULT_MERT_NAME,
    MERT_PARTIAL_CHECKPOINT_FILE,
    MERT_PARTIAL_RANKINGS_FILE,
    MERT_PARTIAL_RESULTS_FILE,
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


def load_mert(model_name: str = DEFAULT_MERT_NAME) -> AutoModel:
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return model


def _pool_hidden_states(outputs: object) -> torch.Tensor:
    hidden = outputs.last_hidden_state
    return hidden.mean(dim=1)


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
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


def _track_index_from_df(train_df: pd.DataFrame) -> Dict[str, int]:
    tracks = sorted(train_df["track_id"].unique().tolist())
    return {t: i for i, t in enumerate(tracks)}


def _labels_to_track_idx(track_ids: List[str], track_to_idx: Dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([track_to_idx[t] for t in track_ids], dtype=torch.long, device=device)


def freeze_all_then_unfreeze_top_k(model: AutoModel, unfreeze_top_k: int) -> int:
    for p in model.parameters():
        p.requires_grad = False

    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise RuntimeError("Unexpected MERT model shape: missing encoder.layers")

    num_layers = len(model.encoder.layers)
    if unfreeze_top_k < 1 or unfreeze_top_k > num_layers:
        raise ValueError(
            f"unfreeze_top_k must be in [1, {num_layers}], got {unfreeze_top_k}"
        )

    start = num_layers - unfreeze_top_k
    for idx in range(start, num_layers):
        for p in model.encoder.layers[idx].parameters():
            p.requires_grad = True
    return start


def _set_train_eval_modes_for_partial(model: AutoModel, unfreeze_start: int) -> None:
    model.train()
    # Keep fully frozen front-end/projection deterministic.
    model.feature_extractor.eval()
    model.feature_projection.eval()
    # Keep frozen lower encoder layers in eval mode; top-k in train mode.
    for i, layer in enumerate(model.encoder.layers):
        if i < unfreeze_start:
            layer.eval()
        else:
            layer.train()


@dataclass
class BestCheckpoint:
    step: int
    val_r1: float
    val_r10: float
    model_state_dict: Dict[str, torch.Tensor]
    head_state_dict: Dict[str, torch.Tensor]


def _select_metric(metrics_df: pd.DataFrame, task: str, k: int) -> float:
    row = metrics_df[metrics_df["task"] == task]
    if row.empty:
        raise ValueError(f"task {task} not found in metrics table")
    return float(row.iloc[0][f"R@{k}"])


def encode_split_with_model_head(
    model: AutoModel,
    head: nn.Module,
    dataloader,
    device: torch.device,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    model.eval()
    head.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch  # type: MertBatch
            inputs = batch.input_values.to(device)
            mask = batch.attention_mask.to(device)
            outputs = model(input_values=inputs, attention_mask=mask, output_hidden_states=False)
            emb = _pool_hidden_states(outputs)
            z = head(emb)
            z = F.normalize(z, dim=1).detach().cpu().numpy().astype(np.float32)
            for i in range(len(batch.clip_ids)):
                rows.append(
                    {
                        "clip_id": batch.clip_ids[i],
                        "track_id": batch.track_ids[i],
                        "label": batch.labels[i],
                        "embedding": z[i],
                    }
                )
    return pd.DataFrame(rows)


def evaluate_split(
    model: AutoModel,
    head: nn.Module,
    dataloader,
    device: torch.device,
    k_values: Tuple[int, ...],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    projected_df = encode_split_with_model_head(model=model, head=head, dataloader=dataloader, device=device)
    split_meta = projected_df[["clip_id", "track_id", "label"]].copy()
    eval_context = build_protocol_eval_context(split_meta, k_values=k_values)

    rankings = build_rankings_cosine(
        query_embeddings_df=projected_df,
        candidate_embeddings_df=projected_df,
        exclude_self_clip=eval_context["exclude_self_clip"],
    )
    rankings_df = build_rankings_with_scores_df(
        query_embeddings_df=projected_df,
        candidate_embeddings_df=projected_df,
        exclude_self_clip=eval_context["exclude_self_clip"],
    )
    metrics_df = evaluate_recall_at_k_from_rankings(rankings_by_query=rankings, eval_context=eval_context)
    return metrics_df, rankings_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 Block 4: Partially-unfrozen MERT retrieval run.")
    p.add_argument("--batch-size-audio", type=int, default=8)
    p.add_argument("--batch-size-head", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--eval-every-steps", type=int, default=250)
    p.add_argument("--max-seconds", type=float, default=5.0)
    p.add_argument("--unfreeze-top-k", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
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
    if args.batch_size_head != 256:
        print(
            f"Note: --batch-size-head={args.batch_size_head} is accepted for CLI parity; "
            "partial run uses on-the-fly audio batches for encoder/head updates."
        )

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

    model = load_mert(DEFAULT_MERT_NAME).to(device)
    unfreeze_start = freeze_all_then_unfreeze_top_k(model, args.unfreeze_top_k)
    print(
        f"Unfreezing top {args.unfreeze_top_k} encoder layers: "
        f"{list(range(unfreeze_start, len(model.encoder.layers)))}"
    )

    train_df = split_frames["train"]
    track_to_idx = _track_index_from_df(train_df)
    in_dim = int(model.config.hidden_size)
    head = nn.Linear(in_dim, args.head_dim).to(device)

    encoder_trainable = [p for p in model.parameters() if p.requires_grad]
    if not encoder_trainable:
        raise RuntimeError("No trainable encoder parameters found after unfreeze.")
    optimizer = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": args.head_lr},
            {"params": encoder_trainable, "lr": args.encoder_lr},
        ]
    )

    print(
        f"Trainable params: encoder={sum(p.numel() for p in encoder_trainable):,}, "
        f"head={sum(p.numel() for p in head.parameters()):,}"
    )

    best = BestCheckpoint(
        step=0,
        val_r1=-1.0,
        val_r10=-1.0,
        model_state_dict=copy.deepcopy(model.state_dict()),
        head_state_dict=copy.deepcopy(head.state_dict()),
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        _set_train_eval_modes_for_partial(model=model, unfreeze_start=unfreeze_start)
        head.train()
        epoch_losses: List[float] = []

        for batch in dataloaders["train"]:
            batch = batch  # type: MertBatch
            global_step += 1

            inputs = batch.input_values.to(device)
            mask = batch.attention_mask.to(device)
            labels = _labels_to_track_idx(batch.track_ids, track_to_idx=track_to_idx, device=device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_values=inputs, attention_mask=mask, output_hidden_states=False)
            emb = _pool_hidden_states(outputs)
            z = head(emb)
            loss = supervised_contrastive_loss(z=z, labels=labels, temperature=args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(head.parameters()) + encoder_trainable, args.grad_clip
            )
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

            if global_step % args.eval_every_steps == 0:
                val_metrics, _ = evaluate_split(
                    model=model,
                    head=head,
                    dataloader=dataloaders["val"],
                    device=device,
                    k_values=REQUIRED_K_VALUES,
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
                        model_state_dict=copy.deepcopy(model.state_dict()),
                        head_state_dict=copy.deepcopy(head.state_dict()),
                    )

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"[epoch {epoch}] train loss={epoch_loss:.6f}")

        val_metrics, _ = evaluate_split(
            model=model,
            head=head,
            dataloader=dataloaders["val"],
            device=device,
            k_values=REQUIRED_K_VALUES,
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
                model_state_dict=copy.deepcopy(model.state_dict()),
                head_state_dict=copy.deepcopy(head.state_dict()),
            )

    print(
        f"[best] step={best.step} same_track val R@1={best.val_r1:.4f} R@10={best.val_r10:.4f}"
    )

    model.load_state_dict(best.model_state_dict)
    head.load_state_dict(best.head_state_dict)
    model.eval()
    head.eval()

    torch.save(
        {
            "step": best.step,
            "val_r1": best.val_r1,
            "val_r10": best.val_r10,
            "model_state_dict": best.model_state_dict,
            "head_state_dict": best.head_state_dict,
            "head_dim": args.head_dim,
            "unfreeze_top_k": args.unfreeze_top_k,
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "grad_clip": args.grad_clip,
        },
        MERT_PARTIAL_CHECKPOINT_FILE,
    )
    print(f"Saved best partial checkpoint: {MERT_PARTIAL_CHECKPOINT_FILE}")

    test_raw_metrics_df, test_rankings_df = evaluate_split(
        model=model,
        head=head,
        dataloader=dataloaders["test"],
        device=device,
        k_values=REQUIRED_K_VALUES,
    )
    test_metrics_df = finalize_metrics_table(test_raw_metrics_df, run_name="mert_partial")
    write_retrieval_artifacts(
        metrics_df=test_metrics_df,
        rankings_df=test_rankings_df,
        metrics_path=MERT_PARTIAL_RESULTS_FILE,
        rankings_path=MERT_PARTIAL_RANKINGS_FILE,
    )
    print(f"Saved partial metrics to: {MERT_PARTIAL_RESULTS_FILE}")
    print(f"Saved partial rankings to: {MERT_PARTIAL_RANKINGS_FILE}")
    print(test_metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
