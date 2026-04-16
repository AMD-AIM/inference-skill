# Inference Benchmark Report

## Configuration
- **Config Key**: {{CONFIG_KEY}}
- **Date**: <current date>
- **GPU**: <detected GPU>
- **Framework**: <framework from config>
- **Model**: <model name>
- **Precision**: <precision>
- **Docker Image**: <image from sweep config>

## Benchmark Results

### Throughput Summary
| Metric                  | Value    |
|-------------------------|----------|
| Request Throughput      | <req/s>  |
| Input Token Throughput  | <tok/s>  |
| Output Token Throughput | <tok/s>  |
| Total Token Throughput  | <tok/s>  |

Note: If the raw benchmark data does not include `input_throughput`, compute it as `total_token_throughput - output_token_throughput`.

### Latency Summary

| Concurrency | ISL×OSL | TTFT Mean (ms) | TTFT P99 (ms) | ITL Mean (ms) | ITL P99 (ms) | End-to-end Latency (s) |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

### Throughput per GPU, Interactivity & Latency

**IMPORTANT**: This MUST be exactly ONE table. Do NOT split into separate tables.

- `Token Throughput per GPU = total_token_throughput / tp`
- `Input Token Throughput per GPU = input_token_throughput / tp`
- `Output Token Throughput per GPU = output_token_throughput / tp`
- `Interactivity (tok/s/user) = 1 / TPOT` where `TPOT = mean_itl_ms / 1000`
- `End-to-end Latency (s) = mean_e2el_ms / 1000`

Conc | ISLxOSL   | TP | Token Thpt/GPU | In Thpt/GPU | Out Thpt/GPU | Interactivity(tok/s/user) | End-to-end Latency (s)
---- | --------- | -- | -------------- | ----------- | ------------ | ------------------------- | ----------------------
...  | ...       | .. | ...            | ...         | ...          | ...                       | ...

### Scaling Analysis
- How throughput scales with concurrency
- Optimal concurrency point
- Memory utilization patterns

## Raw Data
- Location of result files
