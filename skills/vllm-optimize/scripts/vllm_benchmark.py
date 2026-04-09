#!/usr/bin/env python3
"""vLLM concurrent benchmark with configurable concurrency sweep.

Usage: python3 vllm_benchmark.py [--model MODEL] [--isl ISL] [--osl OSL]
    [--concurrency 4,8,16,32,64,128] [--output PATH]
    [--base-url URL] [--api-key KEY]
"""

import argparse
import concurrent.futures
import json
import os
import time


def make_request(base_url, headers, model_name, osl):
    start = time.time()
    try:
        import requests
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Write a short story."}],
                "max_tokens": osl,
                "temperature": 0,
            },
            timeout=300,
        )
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            return {
                "success": True,
                "time": elapsed,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", len(output.split()) if output else 0),
            }
        return {"success": False, "error": resp.status_code, "time": elapsed}
    except Exception as e:
        return {"success": False, "error": str(e), "time": 0}


def main():
    parser = argparse.ArgumentParser(description="vLLM concurrent benchmark")
    parser.add_argument("--model", default=os.environ.get("MODEL", "default"))
    parser.add_argument("--isl", type=int, default=int(os.environ.get("ISL", 1024)))
    parser.add_argument("--osl", type=int, default=int(os.environ.get("OSL", 1024)))
    parser.add_argument("--concurrency", default="4,8,16,32,64,128")
    parser.add_argument("--output", default=os.environ.get("OUTPUT_PATH", "./vllm_results/benchmark_report.json"))
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="dummy")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    levels = [int(x) for x in args.concurrency.split(",")]

    results = []
    for conc in levels:
        print(f"\n=== Testing concurrency {conc} ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as executor:
            start = time.time()
            futures = [
                executor.submit(make_request, args.base_url, headers, args.model, args.osl)
                for _ in range(conc)
            ]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
            total_time = time.time() - start

        successes = [r for r in responses if r.get("success")]
        failures = [r for r in responses if not r.get("success")]

        if successes:
            rps = len(successes) / total_time
            total_input = sum(r.get("prompt_tokens", 0) for r in successes)
            total_output = sum(r.get("output_tokens", 0) for r in successes)
            input_tps = total_input / total_time
            output_tps = total_output / total_time
            latencies = sorted([r["time"] for r in successes])
            n = len(latencies)

            result = {
                "conc": conc,
                "success": len(successes),
                "fail": len(failures),
                "rps": round(rps, 3),
                "input_tps": round(input_tps, 2),
                "output_tps": round(output_tps, 2),
                "total_tps": round(input_tps + output_tps, 2),
                "lat_avg": round(sum(latencies) / n, 2),
                "lat_p50": round(latencies[int(n * 0.50)], 2),
                "lat_p90": round(latencies[int(n * 0.90)], 2),
                "lat_p99": round(latencies[int(n * 0.99)], 2),
            }
            results.append(result)
            print(f"  RPS: {rps:.3f}, Total TPS: {input_tps + output_tps:.2f}, Avg Lat: {sum(latencies)/n:.2f}s")
        else:
            results.append({"conc": conc, "success": 0, "fail": conc, "rps": 0, "total_tps": 0})
            print("  FAILED!")

    output = {
        "meta": {
            "model": args.model,
            "input_tokens": args.isl,
            "output_tokens": args.osl,
            "framework": "vLLM",
            "precision": os.environ.get("DTYPE", "half"),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
