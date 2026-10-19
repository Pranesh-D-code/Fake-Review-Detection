"""Fine-tune a small transformer encoder as an optional second fake-review classifier.

It uses the same labelled reviews, the same stratified 70/15/15 split (seed 42,
one row per ``review_hash``) and the same threshold rules as
``scripts/train_review_classifier.py`` - the loader, split and metric helpers are
imported from that script - so the two models are scored on identical rows.

Outputs (default ``models/``):
    review_transformer/                model, tokenizer, ``meta.json`` and ``test_predictions.npz``
                                       (loaded by ``app/ml_detector.py`` when REVIEW_MODEL_BACKEND selects it)
    review_transformer.metrics.json    held-out metrics, same shape as review_detector.metrics.json

Stages (run the main stage first; the leave-one-category-out probes retrain the
whole model once per category, so they are a separate, much slower step):
    python scripts/train_transformer_classifier.py                  # --stage main
    python scripts/train_transformer_classifier.py --stage loco     # adds leave-one-category-out numbers
    python scripts/train_transformer_classifier.py --smoke          # ~10 minute pipeline check, writes to a scratch dir

Needs the optional packages in requirements-transformer.txt and downloads the base
model weights (about 330 MB for DistilRoBERTa) on first use. Training is CPU-friendly
but slow (hours); long runs write a checkpoint to --work-dir and continue from it
when the same command is started again.

What the metrics mean: how well the model separates *this dataset's* original reviews
from its computer-generated reviews on unseen rows. They are not an accuracy
guarantee for live marketplace reviews.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Cached downloads work without symlinks (Windows) and without the xet backend.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.model_selection import train_test_split

try:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Install the optional packages first: pip install -r requirements-transformer.txt") from exc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_review_classifier as tfidf  # noqa: E402  same loader, split helpers, metrics and threshold rules

DEFAULT_MODEL = "distilbert/distilroberta-base"
CAVEAT = (
    "Trained to separate original from computer-generated reviews in one labelled English "
    "dataset. Output is a text-style likelihood, not proof that a review is fake, and it "
    "has not been validated on live marketplace reviews."
)


# --------------------------------------------------------------------- data helpers

def make_split(labels: np.ndarray, seed: int):
    """Identical to the split in train_review_classifier.main()."""
    indices = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(indices, test_size=0.30, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.50, stratify=labels[rest_idx], random_state=seed)
    return train_idx, val_idx, test_idx


def fingerprint(texts: list[str], idx: np.ndarray) -> str:
    digest = hashlib.sha256()
    for i in idx:
        digest.update(int(i).to_bytes(4, "little"))
        digest.update(texts[i].encode("utf-8"))
    return digest.hexdigest()[:16]


def collate(sequences: list[list[int]], pad_id: int):
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, : len(seq)] = 1
    return ids, mask


def epoch_batches(lengths: np.ndarray, batch_size: int, seed: int, epoch: int) -> list[np.ndarray]:
    """Shuffled batches whose members have similar length (less padding, same data every epoch)."""
    rng = np.random.default_rng(seed * 1000 + epoch)
    order = rng.permutation(len(lengths))
    chunk = batch_size * 50
    batches: list[np.ndarray] = []
    for start in range(0, len(order), chunk):
        part = order[start:start + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches.extend(part[i:i + batch_size] for i in range(0, len(part), batch_size))
    return [batches[i] for i in rng.permutation(len(batches))]


@torch.inference_mode()
def predict_logits(model, sequences: list[list[int]], pad_id: int, batch_size: int) -> np.ndarray:
    """``logit(fake) - logit(original)`` for every sequence; sigmoid(z) is P(computer-generated)."""
    model.eval()
    lengths = np.array([len(s) for s in sequences])
    order = np.argsort(lengths, kind="stable")
    out = np.zeros(len(sequences), dtype=np.float64)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        ids, mask = collate([sequences[i] for i in idx], pad_id)
        logits = model(input_ids=ids, attention_mask=mask).logits
        out[idx] = (logits[:, 1] - logits[:, 0]).double().numpy()
    return out


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """One-parameter calibration fitted on validation logits only (minimises log loss)."""
    def nll(log_t: float) -> float:
        p = np.clip(expit(z / math.exp(log_t)), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(math.exp(minimize_scalar(nll, bounds=(-2.0, 3.0), method="bounded").x))


# ---------------------------------------------------------------------- fine-tuning

def fine_tune(label, train_seq, y_train, val_seq, y_val, args, work_dir: Path, epochs: int, pad_id: int):
    """Fine-tune from the base model; returns (model, history). Resumes from a checkpoint if one matches."""
    signature = {
        "label": label, "model": args.model, "seed": args.seed, "max_length": args.max_length,
        "epochs": epochs, "lr": args.lr, "batch_size": args.batch_size, "rows": len(train_seq),
        "first_row_hash": hashlib.sha256(repr(train_seq[0]).encode()).hexdigest()[:12],
    }
    ckpt_path = work_dir / f"checkpoint_{label}.pt"
    best_path = work_dir / f"best_{label}.pt"

    torch.manual_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    no_decay = ("bias", "LayerNorm.weight")
    parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    steps_per_epoch = math.ceil(len(train_seq) / args.batch_size)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(args.warmup * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)),
    )

    lengths = np.array([len(s) for s in train_seq])
    targets = torch.tensor(y_train, dtype=torch.long)
    history: list[dict] = []
    best_auc, start_epoch, start_batch, step, spent = -1.0, 0, 0, 0, 0.0

    if ckpt_path.is_file() and not args.fresh:
        state = torch.load(ckpt_path, map_location="cpu")
        if state["signature"] != signature:
            raise SystemExit(f"{ckpt_path} was written with different settings; rerun with --fresh to discard it.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best_auc, spent = state["history"], state["best_auc"], state["spent"]
        start_epoch, start_batch, step = state["epoch"], state["batch"], state["step"]
        print(f"[{label}] resumed at epoch {start_epoch + 1}, batch {start_batch}/{steps_per_epoch} (step {step}/{total_steps})", flush=True)

    def save_checkpoint(epoch: int, batch: int) -> None:
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history, "best_auc": best_auc,
                    "spent": spent + time.time() - started, "epoch": epoch, "batch": batch, "step": step}, tmp)
        os.replace(tmp, ckpt_path)

    started = time.time()
    last_checkpoint = time.time()
    step_at_start = step
    for epoch in range(start_epoch, epochs):
        batches = epoch_batches(lengths, args.batch_size, args.seed, epoch)
        model.train()
        running, counted = 0.0, 0
        for batch_no in range(start_batch if epoch == start_epoch else 0, len(batches)):
            idx = batches[batch_no]
            ids, mask = collate([train_seq[i] for i in idx], pad_id)
            loss = torch.nn.functional.cross_entropy(model(input_ids=ids, attention_mask=mask).logits, targets[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            running += float(loss.detach())
            counted += 1
            if step % args.log_every == 0:
                elapsed = spent + time.time() - started
                rate = (time.time() - started) / max(1, step - step_at_start)
                eta = rate * (total_steps - step)
                print(f"[{label}] epoch {epoch + 1}/{epochs} step {step}/{total_steps} loss {running / counted:.4f} "
                      f"{rate:.1f}s/step elapsed {elapsed / 60:.1f}m eta {eta / 60:.1f}m", flush=True)
                running, counted = 0.0, 0
            stop_here = args.stop_after_steps and step >= args.stop_after_steps
            if stop_here or time.time() - last_checkpoint > args.checkpoint_minutes * 60:
                save_checkpoint(epoch, batch_no + 1)
                last_checkpoint = time.time()
            if stop_here:
                raise SystemExit(f"stopped after {step} steps (--stop-after-steps); checkpoint saved, rerun to continue")

        record = {"epoch": epoch + 1, "step": step}
        if val_seq is not None:
            z_val = predict_logits(model, val_seq, pad_id, args.eval_batch_size)
            val_metrics = tfidf.full_metrics(y_val, expit(z_val))
            record["validation"] = val_metrics
            print(f"[{label}] epoch {epoch + 1} validation: accuracy {val_metrics['accuracy']} F1 {val_metrics['f1_fake']} "
                  f"ROC-AUC {val_metrics['roc_auc']} Brier {val_metrics['brier']}", flush=True)
            improved = val_metrics["roc_auc"] > best_auc
            best_auc = max(best_auc, val_metrics["roc_auc"])
        else:
            improved = True
        history.append(record)
        if improved:
            torch.save(model.state_dict(), best_path)
        save_checkpoint(epoch + 1, 0)
        last_checkpoint = time.time()

    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    ckpt_path.unlink(missing_ok=True)
    return model, history


# ------------------------------------------------------------------------ reporting

def build_results(y_val, z_val, y_test, z_test, temperature, thresholds):
    val_prob, test_prob = expit(z_val / temperature), expit(z_test / temperature)
    results = {
        "test_at_0.5": tfidf.full_metrics(y_test, test_prob),
        "test_at_suspicious_threshold": tfidf.metrics_at(y_test, test_prob, thresholds["suspicious_at"]),
        "validation_at_0.5": tfidf.full_metrics(y_val, val_prob),
    }
    genuine = test_prob <= thresholds["genuine_below"]
    results["test_genuine_band"] = {
        "share_of_rows": round(float(genuine.mean()), 4),
        "original_share_within_band": round(float(1 - y_test[genuine].mean()), 4) if genuine.any() else None,
    }
    uncertain = ~genuine & (test_prob < thresholds["suspicious_at"])
    results["test_uncertain_band_share"] = round(float(uncertain.mean()), 4)
    results["test_at_0.5"]["brier_uncalibrated"] = round(float(
        np.mean((expit(z_test) - y_test) ** 2)), 4)
    return results


def loco_probe(category, texts, labels, categories, sequences, args, work_dir, pad_id):
    mask = categories == category
    train_rows = np.where(~mask)[0]
    if args.loco_train_fraction < 1.0:
        rng = np.random.default_rng(args.seed)
        train_rows = np.sort(rng.choice(train_rows, int(len(train_rows) * args.loco_train_fraction), replace=False))
    test_rows = np.where(mask)[0]
    model, history = fine_tune(f"loco_{category}", [sequences[i] for i in train_rows], labels[train_rows], None, None,
                               args, work_dir, args.loco_epochs, pad_id)
    z = predict_logits(model, [sequences[i] for i in test_rows], pad_id, args.eval_batch_size)
    return {"rows": int(mask.sum()), "train_rows": int(len(train_rows)), **tfidf.full_metrics(labels[test_rows], expit(z))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["main", "loco"], default="main")
    parser.add_argument("--db", type=Path, default=ROOT / "trustengine.db")
    parser.add_argument("--dataset", default=None, help="external_dataset name (default: the largest one)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="checkpoints (about 1 GB); default: outside the project so OneDrive does not sync them")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face id or local path of the base encoder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128, help="tokens kept per review")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=float, default=0.06, help="share of steps spent warming the learning rate up")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4),
                        help="CPU threads; more is not faster here (10 threads measured ~50%% slower than 8 on a 6P+4E laptop)")
    parser.add_argument("--suspicious-precision", type=float, default=0.98)
    parser.add_argument("--genuine-purity", type=float, default=0.98)
    parser.add_argument("--holdout-categories", type=int, default=3)
    parser.add_argument("--loco-epochs", type=int, default=1)
    parser.add_argument("--loco-train-fraction", type=float, default=1.0,
                        help="share of the other categories' rows used per probe (1.0 = all, like the TF-IDF probe)")
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline; never writes into models/")
    parser.add_argument("--fresh", action="store_true", help="ignore any checkpoint in --work-dir")
    parser.add_argument("--checkpoint-minutes", type=float, default=20.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--stop-after-steps", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    base_work = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / "TrustEngineTransformer"
    args.work_dir = args.work_dir or (base_work / ("smoke" if args.smoke else "run"))
    if args.smoke:
        args.out_dir = args.work_dir / "out"
        args.epochs = args.loco_epochs = 1
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    started = time.time()

    dataset, mapping, texts, labels, categories = tfidf.load_dataset(args.db, args.dataset)
    print(f"Dataset '{dataset}': {len(texts)} unique reviews, {int(labels.sum())} fake, {int((1 - labels).sum())} original. "
          f"Label mapping: {mapping}")
    train_idx, val_idx, test_idx = make_split(labels, args.seed)
    if args.smoke:
        train_idx, val_idx, test_idx = train_idx[:1600], val_idx[:400], test_idx[:400]
    print(f"Split: train {len(train_idx)}, validation {len(val_idx)}, test {len(test_idx)}  "
          f"(test fingerprint {fingerprint(texts, test_idx)})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sequences = tokenizer(texts, truncation=True, max_length=args.max_length)["input_ids"]
    pad_id = tokenizer.pad_token_id
    truncated = float(np.mean([len(s) >= args.max_length for s in sequences]))
    print(f"Tokenised with max_length={args.max_length}; {truncated:.1%} of reviews are truncated. Threads: {args.threads}")

    metrics_path = args.out_dir / "review_transformer.metrics.json"
    model_dir = args.out_dir / "review_transformer"

    if args.stage == "main":
        pick = lambda idx: [sequences[i] for i in idx]
        model, history = fine_tune("main", pick(train_idx), labels[train_idx], pick(val_idx), labels[val_idx],
                                   args, args.work_dir, args.epochs, pad_id)
        z_val = predict_logits(model, pick(val_idx), pad_id, args.eval_batch_size)
        z_test = predict_logits(model, pick(test_idx), pad_id, args.eval_batch_size)
        y_val, y_test = labels[val_idx], labels[test_idx]

        # Calibration and decision thresholds come from validation data only (never the test set).
        temperature = fit_temperature(z_val, y_val)
        val_prob = expit(z_val / temperature)
        thresholds = {
            "suspicious_at": tfidf.threshold_for_precision(y_val, val_prob, args.suspicious_precision),
            "genuine_below": tfidf.threshold_for_genuine(y_val, val_prob, args.genuine_purity),
        }
        print(f"Temperature {temperature:.3f}; thresholds from validation: {thresholds}")
        results = build_results(y_val, z_val, y_test, z_test, temperature, thresholds)
        print("Scoring the existing rule-based baseline on the same test rows...")
        results["rule_based_baseline_on_test"] = tfidf.heuristic_baseline([texts[i] for i in test_idx], y_test)
        results["leave_one_category_out"] = {}

        words = np.array([len(t.split()) for t in texts])
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": "transformer",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "dataset": dataset,
            "label_mapping": mapping,
            "positive_class": tfidf.FAKE_NORMALIZED_LABEL,
            "rows": {"total": len(texts), "train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
            "model": f"{args.model} fine-tuned (max_length={args.max_length}, {args.epochs} epochs, temperature-calibrated)",
            "base_model": args.model,
            "max_length": args.max_length,
            "temperature": round(temperature, 4),
            "min_words_for_ml": int(max(6, np.percentile(words, 2))),
            "thresholds": thresholds,
            "seed": args.seed,
            "split_fingerprint": fingerprint(texts, test_idx),
            "caveat": CAVEAT,
        }
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        np.savez_compressed(model_dir / "test_predictions.npz", test_idx=test_idx, labels=y_test, z=z_test,
                            split_fingerprint=np.array(meta["split_fingerprint"]))
        report = {**meta, "metrics": results,
                  "training": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
                               "warmup_share": args.warmup, "weight_decay": args.weight_decay,
                               "truncated_share": round(truncated, 4), "history": history,
                               "threads": args.threads, "device": "cpu"},
                  "training_seconds": round(time.time() - started, 1)}
        metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        headline = results["test_at_0.5"]
        print("\n=== Held-out test (never used for fitting, calibration or threshold choice) ===")
        print(f"accuracy {headline['accuracy']}  precision {headline['precision_fake']}  recall {headline['recall_fake']}  "
              f"F1 {headline['f1_fake']}  ROC-AUC {headline['roc_auc']}  Brier {headline['brier']}")
        flagged = results["test_at_suspicious_threshold"]
        print(f"at flag threshold {thresholds['suspicious_at']}: precision {flagged['precision_fake']}  recall {flagged['recall_fake']}")
        print(f"Saved {model_dir} and {metrics_path.name}")
        return

    # ---- stage: loco -------------------------------------------------------------
    if not metrics_path.is_file():
        raise SystemExit("Run the main stage first (no review_transformer.metrics.json found).")
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    sizes = sorted(set(categories), key=lambda c: -int((categories == c).sum()))
    held_out = [c for c in sizes if (categories == c).sum() >= 1000][: args.holdout_categories]
    shift = dict(report["metrics"].get("leave_one_category_out") or {})
    for category in held_out:
        if category in shift:
            print(f"hold out {category}: already done")
            continue
        shift[category] = loco_probe(category, texts, labels, categories, sequences, args, args.work_dir, pad_id)
        print(f"  hold out {category:<30} accuracy={shift[category]['accuracy']:.4f} f1={shift[category]['f1_fake']:.4f} "
              f"roc_auc={shift[category]['roc_auc']:.4f}", flush=True)
        report["metrics"]["leave_one_category_out"] = shift
        report["leave_one_category_out_protocol"] = {
            "epochs": args.loco_epochs, "train_fraction": args.loco_train_fraction,
            "note": "Retrained from the base model on all other categories; probabilities are uncalibrated "
                    "(no validation split is held back), so Brier here is not comparable to the main-run Brier.",
        }
        metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved leave-one-category-out results into {metrics_path.name}")


if __name__ == "__main__":
    main()
