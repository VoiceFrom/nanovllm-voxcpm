"""Generate a deterministic waveform and optionally compare it with a reference."""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np


def compare_waveforms(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    common_samples = min(actual.size, reference.size)
    actual_common = actual[:common_samples].astype(np.float64)
    reference_common = reference[:common_samples].astype(np.float64)
    error = actual_common - reference_common
    centered_actual = actual_common - float(actual_common.mean())
    centered_reference = reference_common - float(reference_common.mean())
    denominator = np.linalg.norm(centered_actual) * np.linalg.norm(centered_reference)
    correlation = float(np.dot(centered_actual, centered_reference) / denominator) if denominator else 1.0
    return {
        "actual_samples": int(actual.size),
        "reference_samples": int(reference.size),
        "common_samples": common_samples,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "abs_error_p99_9": float(np.percentile(np.abs(error), 99.9)),
        "max_abs_error": float(np.max(np.abs(error))),
        "correlation": correlation,
    }


async def async_main(args: argparse.Namespace) -> None:
    from nanovllm_voxcpm import VoxCPM

    target_text = (
        Path(args.target_text_file).read_text(encoding="utf-8").strip()
        if args.target_text_file
        else args.target_text
    )
    server_pool = VoxCPM.from_pretrained(
        model=args.model,
        inference_timesteps=args.inference_timesteps,
        max_num_batched_tokens=16384,
        max_num_seqs=512,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        devices=[args.device],
        enforce_eager=args.enforce_eager,
    )
    async def generate_one(seed: int) -> np.ndarray:
        chunks = []
        async for chunk in server_pool.generate(
            target_text=target_text,
            max_generate_length=args.max_generate_length,
            temperature=args.temperature,
            cfg_value=args.cfg_value,
            seed=seed,
        ):
            chunks.append(np.asarray(chunk, dtype=np.float32))
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    try:
        await server_pool.wait_for_ready()
        waveforms = await asyncio.gather(*(generate_one(args.seed + index) for index in range(args.concurrency)))
    finally:
        await server_pool.stop()

    if args.concurrency == 1:
        np.save(args.output, waveforms[0])
    else:
        np.savez(args.output, **{f"request_{index}": waveform for index, waveform in enumerate(waveforms)})
    result: dict[str, object] = {
        "output": str(Path(args.output).resolve()),
        "num_samples": [int(waveform.size) for waveform in waveforms],
    }
    if args.reference:
        reference = np.load(args.reference)
        if args.concurrency == 1:
            result["comparison"] = compare_waveforms(waveforms[0], reference)
        else:
            result["comparison"] = [
                compare_waveforms(waveform, reference[f"request_{index}"])
                for index, waveform in enumerate(waveforms)
            ]
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--target-text", default="Hello world.")
    parser.add_argument("--target-text-file")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--max-generate-length", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
