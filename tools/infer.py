"""Run reproducible SFT or TimeLens inference on an explicit query manifest."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import parse_prediction
from vtg.data import ABS_PROMPT, REL_PROMPT, first_span, read_records, validate_sample, write_records


def sft_inputs(row, processor, args):
    from qwen_vl_utils import process_vision_info
    prompt = ABS_PROMPT if args.representation == "absolute" else REL_PROMPT
    messages = [{"role": "user", "content": [
        {"type": "video", "video": row["video_path"], "fps": args.fps,
         "max_frames": args.max_frames, "min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        {"type": "text", "text": prompt.format(query=row["query"])}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True, image_patch_size=14)
    inputs = processor(text=[text], images=images, videos=videos, do_resize=False,
                       return_tensors="pt", **kwargs)
    inputs.pop("mm_token_type_ids", None)
    return inputs


def timelens_inputs(row, processor, args):
    sys.path.insert(0, str(ROOT / "timelens"))
    from training.data.inference_collator import GroundingDatasetInference, collate_fn
    settings = SimpleNamespace(format_model_path="qwen3-vl-8b", min_tokens=args.min_tokens,
                               total_tokens=args.total_tokens, fps=args.fps, fps_max_frames=args.max_frames)
    sample = GroundingDatasetInference([row], settings)[0]
    return collate_fn([sample], processor, model_name="qwen3-vl-8b")["inputs"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=["sft", "timelens"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--adapter-only", action="store_true")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--representation", choices=["absolute", "relative"], default="absolute")
    p.add_argument("--attention", choices=["sdpa", "flash_attention_2"], default="sdpa")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--min-pixels", type=int, default=100352)
    p.add_argument("--max-pixels", type=int, default=109760)
    p.add_argument("--min-tokens", type=int, default=64)
    p.add_argument("--total-tokens", type=int, default=14336)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--round-unit", type=float, help="TimeLens historical protocol: 1 second; SFT: 0 (no rounding)")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        p.error("Invalid shard selection")
    if args.workload == "timelens" and args.representation != "absolute":
        p.error("TimeLens Qwen3 experiment uses absolute seconds")
    args.max_frames = args.max_frames or (700 if args.workload == "sft" else 448)
    if args.round_unit is None:
        args.round_unit = 0.0 if args.workload == "sft" else 1.0
    if args.round_unit < 0 or args.fps <= 0 or args.max_frames < 2:
        p.error("Invalid video/rounding settings")
    rows = read_records(args.manifest)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        p.error("Manifest contains duplicate ids")
    rows = rows[args.shard_index::args.num_shards]
    for row in rows:
        validate_sample(row, check_video=False)
    if args.dry_run:
        print(json.dumps({"queries": len(rows), "settings": vars(args)}, indent=2))
        return
    if not rows:
        p.error("Selected shard contains no queries")
    output = Path(args.output)
    if output.exists():
        p.error("Prediction output already exists; choose another path")
    import torch
    from vtg.checkpoints import load_model
    model, processor = load_model(args.model, args.adapter, args.attention, adapter_only=args.adapter_only)
    expected_type = "qwen2_5_vl" if args.workload == "sft" else "qwen3_vl"
    if model.config.model_type != expected_type:
        p.error(f"Expected {expected_type}, got {model.config.model_type}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_records(str(output) + ".manifest.json", rows)
    write_records(str(output) + ".config.json", vars(args))
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows, 1):
            result = {"id": row["id"], "span": first_span(row["span"]), "query": row["query"],
                      "video_path": row["video_path"], "duration": row["duration"]}
            try:
                validate_sample(row, check_video=True)
                make_inputs = sft_inputs if args.workload == "sft" else timelens_inputs
                inputs = make_inputs(row, processor, args).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
                trimmed = generated[:, inputs["input_ids"].shape[1]:]
                answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0]
                prediction = parse_prediction(answer, args.representation, row["duration"])
                if prediction is not None and args.round_unit:
                    prediction = [round(t / args.round_unit) * args.round_unit for t in prediction]
                result.update(answer=answer, prediction=prediction)
            except Exception as error:
                result.update(prediction=None, error_type=type(error).__name__, error=str(error))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(rows)} {result['id']} {'error' if result.get('error') else 'done'}")


if __name__ == "__main__":
    main()
