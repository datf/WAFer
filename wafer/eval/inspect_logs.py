"""CLI utility to parse, sanitize, train/calibrate, and evaluate access logs with Gemma-Clef."""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from wafer.data.log_parser import parse_and_sanitize_log_line, parse_and_classify_file
from wafer.data.synthetic import STANDARD_QUESTIONS, generate_synthetic_waf_samples
from wafer.model.backbone import MockTokenizer, SurrogateDecisionModel
from wafer.model.encoder import encode_record
from wafer.training.train import train_gemma_clef


def evaluate_sanitized_request(
    model: torch.nn.Module,
    tokenizer: Any,
    state_dict: Dict[str, Any],
    questions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates a single sanitized HTTPRequestState against WAF questions in a single forward pass.
    Returns calibrated probability distributions and decision recommendations.
    """
    q_specs = questions or STANDARD_QUESTIONS
    record = {
        "state": state_dict,
        "questions": q_specs,
    }

    enc = encode_record(tokenizer, record)

    model.eval()
    with torch.no_grad():
        logits_list = model.forward_record(enc)

    # Convert logits to calibrated softmax probabilities per question
    results: Dict[str, Any] = {}
    for q_id, logits, q_type in zip(q_specs.keys(), logits_list, enc.question_types):
        probs = F.softmax(logits.float(), dim=-1).cpu().tolist()
        spec = q_specs[q_id]
        criteria = spec.get("criteria")

        if q_type == "noul":
            opt_names = ["true", "false"]
        elif q_type == "choice" and isinstance(criteria, dict):
            opt_names = list(criteria.keys())
        elif q_type == "choice" and isinstance(criteria, list):
            opt_names = [str(i) for i in range(len(criteria))]
        elif q_type == "score" and isinstance(criteria, list):
            opt_names = [f"score_{i}" for i in range(len(criteria))]
        else:
            opt_names = [str(i) for i in range(len(probs))]

        prob_map = dict(zip(opt_names, [round(p, 4) for p in probs]))
        top_opt = max(prob_map, key=prob_map.get)
        results[q_id] = {
            "prediction": top_opt,
            "probabilities": prob_map,
        }

    # Derive gateway decision with simplified schema: action [allow|deny], alert [true|false]
    # Invariant: blocking strictly implies alerting
    action_probs = results.get("action", {}).get("probabilities", {})
    p_deny = action_probs.get("deny", results.get("gateway_action", {}).get("probabilities", {}).get("block", 0.0))
    p_allow = action_probs.get("allow", 1.0 - p_deny)

    alert_probs = results.get("alert", {}).get("probabilities", {})
    p_alert = alert_probs.get("true", results.get("is_malicious", {}).get("probabilities", {}).get("true", 0.0))

    if p_deny >= 0.50:
        decision = "DENY (403)"
        alert_flag = True
    elif p_alert >= 0.50:
        decision = "ALERT (200)"
        alert_flag = True
    else:
        decision = "ALLOW (200)"
        alert_flag = False

    return {
        "state": state_dict,
        "decision": decision,
        "action": "deny" if p_deny >= 0.50 else "allow",
        "alert": alert_flag,
        "p_deny": round(p_deny, 4),
        "p_allow": round(p_allow, 4),
        "p_alert": round(p_alert, 4),
        # Legacy backwards compatibility keys
        "p_malicious": round(p_alert, 4),
        "p_block": round(p_deny, 4),
        "attack_vector": "anomaly" if alert_flag else "benign",
        "severity": "2" if p_deny >= 0.50 else ("1" if alert_flag else "0"),
        "full_eval": results,
    }


def load_waf_training_records(
    log_source: Optional[Union[str, List[str], Path]] = None,
    include_synthetic: bool = True,
    max_records: Optional[int] = 1000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Assembles a balanced WAF training dataset combining real parsed access logs
    (with classified ground truth labels, including 301/302/308 tarpit redirects)
    and synthetic exploit vectors.
    """
    real_records: List[Dict[str, Any]] = []

    sources: List[Path] = []
    if log_source:
        if isinstance(log_source, (str, Path)):
            sources.append(Path(log_source))
        else:
            sources.extend(Path(s) for s in log_source)
    else:
        for candidate in ["data/dnlogs.log", "data/sample_traffic.log"]:
            p = Path(candidate)
            if p.exists():
                sources.append(p)
        caddy_dir = Path("backuplogs/caddy")
        if caddy_dir.exists():
            sources.append(caddy_dir)

    for src in sources:
        if src.is_file():
            for _, _, rec in parse_and_classify_file(src):
                if rec.get("targets"):
                    real_records.append(rec)
        elif src.is_dir():
            for root, _, files in os.walk(src):
                for f in sorted(files):
                    fp = Path(root) / f
                    if fp.suffix in (".log", ".gz", ".txt") or "access" in f:
                        for _, _, rec in parse_and_classify_file(fp, limit=300):
                            if rec.get("targets"):
                                real_records.append(rec)

    synthetic_records = generate_synthetic_waf_samples() if include_synthetic else []
    combined = real_records + synthetic_records

    rng = random.Random(seed)
    allows = [r for r in combined if r.get("targets", {}).get("action") == "allow"]
    denies = [r for r in combined if r.get("targets", {}).get("action") == "deny"]
    rng.shuffle(allows)
    rng.shuffle(denies)

    if max_records is not None and max_records > 0:
        half = max_records // 2
        selected = allows[:half] + denies[:half]
    else:
        selected = allows + denies

    rng.shuffle(selected)
    return selected


def get_trained_surrogate(
    checkpoint_path: str = "checkpoints/calibrated_surrogate.pt",
    epochs: Optional[int] = None,
    seed: int = 42,
    force_retrain: bool = False,
    train_records: Optional[int] = None,
    train_logs: Optional[str] = None,
) -> Tuple[torch.nn.Module, Any]:
    """
    Returns a calibrated surrogate model.
    Loads from checkpoint_path if available; otherwise trains deterministically
    with fixed seed and saves checkpoint to disk.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = MockTokenizer()
    model = SurrogateDecisionModel(hidden_size=128, d_head=64, n_heads=4, vocab_size=1000)

    if os.path.exists(checkpoint_path) and not force_retrain:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            return model, tokenizer
        except Exception:
            pass  # Fallback to retrain if file corrupt or incompatible

    eff_epochs = epochs if epochs is not None else 20
    eff_records = train_records if train_records is not None else 500

    # Train deterministically on combined real logs and synthetic samples
    records = load_waf_training_records(
        log_source=train_logs,
        include_synthetic=True,
        max_records=eff_records,
        seed=seed,
    )
    encoded = [encode_record(tokenizer, r, targets=r.get("targets")) for r in records]

    train_gemma_clef(
        model=model,
        train_data=encoded,
        epochs=eff_epochs,
        head_lr=1e-3,
        lora_lr=1e-3,
        gradient_accumulation_steps=4,
    )

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    model.eval()
    return model, tokenizer


def get_trained_pretrained_backbone(
    model_id: str = "google/embeddinggemma-2",
    checkpoint_path: Optional[str] = None,
    epochs: Optional[int] = None,
    seed: int = 42,
    force_retrain: bool = False,
    device_pref: str = "auto",
    load_in_4bit: bool = False,
    train_records: Optional[int] = None,
    train_logs: Optional[str] = None,
) -> Tuple[torch.nn.Module, Any]:
    """
    Initializes a real pretrained backbone (EmbeddingGemma 2, Gemma 4, SmolLM2, etc.),
    attaches Rank-32/64 LoRA + JointSchemaHead, and trains on synthetic WAF data.
    """
    from transformers import AutoTokenizer, AutoConfig
    from wafer.model.backbone import (
        EmbeddingGemmaDecisionModel,
        GemmaClefDecisionModel,
        resolve_model_id,
        get_backbone_output_dim,
    )

    real_model_id = resolve_model_id(model_id)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if checkpoint_path is None:
        safe_name = real_model_id.replace("/", "_")
        checkpoint_path = f"checkpoints/{safe_name}_clef.pt"

    tokenizer = AutoTokenizer.from_pretrained(real_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if device_pref == "cpu":
        device = "cpu"
    elif device_pref == "cuda":
        device = "cuda"
    else:
        has_compatible_cuda = False
        if torch.cuda.is_available():
            try:
                major, minor = torch.cuda.get_device_capability()
                dev_name = torch.cuda.get_device_name(0)
                if major >= 7:  # Volta (7.0), Turing (7.5), Ampere (8.0), Ada (8.9), Hopper (9.0)
                    has_compatible_cuda = True
                else:
                    print(
                        f"Notice: Detected GPU '{dev_name}' (Compute Capability {major}.{minor}), which is older than "
                        f"the PyTorch wheel's sm_70+ build. Falling back to CPU to prevent CUBLAS_STATUS_ARCH_MISMATCH.",
                        flush=True,
                    )
            except Exception:
                has_compatible_cuda = False
        device = "cuda" if has_compatible_cuda else "cpu"

    dtype = torch.float32 if device == "cpu" else torch.float16

    # Determine head dimensions based on model size
    cfg = AutoConfig.from_pretrained(real_model_id)
    hidden_size = get_backbone_output_dim(cfg, default=768)
    d_head = 256 if hidden_size <= 768 else 512
    lora_r = 32 if hidden_size <= 768 else 64

    print(f"Loading pretrained backbone '{real_model_id}' (hidden_size={hidden_size}, d_head={d_head}) on device={device} (4-bit={load_in_4bit})...", flush=True)
    if "embeddinggemma" in real_model_id.lower():
        model = EmbeddingGemmaDecisionModel(
            model_id=real_model_id,
            lora_rank=lora_r,
            lora_alpha=lora_r,
            d_head=d_head,
            torch_dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
    else:
        model = GemmaClefDecisionModel(
            model_id=real_model_id,
            lora_rank=lora_r,
            lora_alpha=lora_r,
            d_head=d_head,
            torch_dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
    model.to(device)

    if os.path.exists(checkpoint_path) and not force_retrain:
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Loaded existing checkpoint from {checkpoint_path}", flush=True)
            return model, tokenizer
        except Exception as e:
            print(f"Notice: Could not load existing checkpoint ({e}), retraining...", flush=True)

    eff_epochs = epochs if epochs is not None else (2 if device == "cpu" else 3)
    eff_records = train_records if train_records is not None else (200 if device == "cpu" else 1000)

    records = load_waf_training_records(
        log_source=train_logs,
        include_synthetic=True,
        max_records=eff_records,
        seed=seed,
    )
    encoded = [encode_record(tokenizer, r, targets=r.get("targets")) for r in records]

    print(f"Fine-tuning {model_id} JointSchemaHead + LoRA on {len(records)} records (real traffic logs + synthetic vectors, {eff_epochs} epochs)...", flush=True)
    train_gemma_clef(
        model=model,
        train_data=encoded,
        epochs=eff_epochs,
        head_lr=2e-4,
        lora_lr=5e-5,
        gradient_accumulation_steps=4,
    )

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}", flush=True)
    model.eval()
    return model, tokenizer


def inspect_log_stream(
    lines: List[str],
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[Any] = None,
    train_first: bool = False,
    model_id: str = "surrogate",
    checkpoint_path: Optional[str] = None,
    force_retrain: bool = False,
    epochs: int = 5,
    device: str = "auto",
    load_in_4bit: bool = False,
    train_records: Optional[int] = 1000,
    train_logs: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Parses, sanitizes, and evaluates an array of raw access log lines."""
    if model is None:
        if model_id != "surrogate":
            model, tokenizer = get_trained_pretrained_backbone(
                model_id=model_id,
                checkpoint_path=checkpoint_path,
                epochs=epochs,
                force_retrain=force_retrain,
                device_pref=device,
                load_in_4bit=load_in_4bit,
                train_records=train_records,
                train_logs=train_logs,
            )
        elif train_first:
            ckpt = checkpoint_path or "checkpoints/calibrated_surrogate.pt"
            model, tokenizer = get_trained_surrogate(
                checkpoint_path=ckpt,
                epochs=epochs if epochs != 5 else 20,
                force_retrain=force_retrain,
                train_records=train_records,
                train_logs=train_logs,
            )
        else:
            torch.manual_seed(42)
            model = SurrogateDecisionModel(hidden_size=64, d_head=32, n_heads=4, vocab_size=1000)
            tokenizer = MockTokenizer()

    evaluations: List[Dict[str, Any]] = []
    for line in lines:
        state = parse_and_sanitize_log_line(line)
        if state is None:
            continue
        eval_res = evaluate_sanitized_request(model, tokenizer, state.to_model_state())
        evaluations.append(eval_res)

    return evaluations


def print_evaluation_summary(evaluations: List[Dict[str, Any]]) -> None:
    """Prints a clean ASCII summary table of evaluated requests."""
    print("=" * 135)
    print(f"{'METHOD':<6} | {'URI':<48} | {'PROTO':<8} | {'USER-AGENT':<28} | {'DECISION':<12} | {'CONF(DENY)':<10} | {'CONF(ALERT)'}")
    print("=" * 135)

    for item in evaluations:
        st = item["state"]
        method = st.get("method", "GET")[:6]
        uri = st.get("uri", "/")
        if len(uri) > 46:
            uri = uri[:43] + "..."
        proto = (st.get("protocol") or "HTTP/1.1")[:8]
        ua = (st.get("user_agent") or "")[:26] or "-"
        decision = item["decision"]
        p_deny = f"{item['p_deny']:.2f}"
        p_alert = f"{item['p_alert']:.2f}"

        print(f"{method:<6} | {uri:<48} | {proto:<8} | {ua:<28} | {decision:<12} | {p_deny:<10} | {p_alert}")

    print("=" * 135)


def main():
    parser = argparse.ArgumentParser(description="Evaluate sanitized HTTP access logs with Gemma-Clef")
    parser.add_argument("logfile", nargs="?", help="Path to log file (or reads stdin if omitted)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--train", action="store_true", default=True, help="Train surrogate on synthetic data first (default: True)")
    parser.add_argument("--model-id", default="surrogate", help="Model backbone: 'surrogate' or HuggingFace ID (e.g. Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: auto-selected by device)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to execute on (auto, cpu, cuda)")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    parser.add_argument("--4bit", "--load-in-4bit", dest="load_in_4bit", action="store_true", help="Load backbone in 4-bit NF4 precision via BitsAndBytes")
    parser.add_argument("--train-records", type=int, default=None, help="Maximum number of balanced records for fine-tuning (default: auto-selected by device)")
    parser.add_argument("--train-logs", default=None, help="Path to access log file or directory to use for fine-tuning")
    args = parser.parse_args()

    if args.logfile:
        with open(args.logfile, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    evals = inspect_log_stream(
        lines,
        train_first=args.train,
        model_id=args.model_id,
        checkpoint_path=args.checkpoint,
        force_retrain=args.force_retrain,
        epochs=args.epochs,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        train_records=args.train_records,
        train_logs=args.train_logs,
    )

    if args.json:
        print(json.dumps(evals, indent=2))
    else:
        print_evaluation_summary(evals)


if __name__ == "__main__":
    main()

