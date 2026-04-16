# Analyzer Manifest Schema

Defines the file manifest format used when spawning analyzer subagents. The manifest ensures the analyzer receives an explicit, bounded list of data files to read.

## Format

The phase agent passes this manifest as part of the analyzer subagent's prompt:

```yaml
analyzer_manifest:
  task: "{description of analysis task}"
  output_path: "{path for analysis output}"
  files:
    - path: "{relative path from OUTPUT_DIR}"
      description: "{what this file contains}"
      format: "json | csv | log | markdown"
      required: true | false
    - path: "..."
      description: "..."
      format: "..."
      required: true
```

## Rules

1. **Maximum 10 data files** per manifest. If more are needed, split into multiple analyzer invocations.
2. **All paths are relative to OUTPUT_DIR**. The analyzer resolves them to absolute paths.
3. **Required files**: If a required file is missing, the analyzer reports an error and stops.
4. **Optional files**: If an optional file is missing, the analyzer proceeds without it.
5. **Format hint**: Tells the analyzer how to parse the file.

## Example

```yaml
analyzer_manifest:
  task: "Analyze benchmark results and identify throughput bottlenecks"
  output_path: "agent-results/phase-03-result.md"
  files:
    - path: "results/benchmark_results.json"
      description: "Raw benchmark metrics from all sweep points"
      format: "json"
      required: true
    - path: "results/sweep_configs.json"
      description: "Sweep configuration used for benchmarking"
      format: "json"
      required: true
    - path: "results/bottlenecks.json"
      description: "Previously identified bottlenecks (may not exist on first run)"
      format: "json"
      required: false
```

## Usage

Phase agents that spawn analyzer subagents construct this manifest and include it in the subagent prompt. The analyzer reads `analysis-agent.md` (its role doc) and the manifest (its task-specific input). Total prompt size stays under ~80 lines + data file contents.
