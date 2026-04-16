#!/usr/bin/env python3
"""Validate a config key against master YAML.

Usage:
    python3 validate_config_key.py --config-file <path> --config-key <key>

Exit code 1 if the config key is missing. Prints related keys to help the user
recover quickly.
"""

import argparse
import difflib
import os
import sys

import yaml


def load_config_keys(path):
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a top-level mapping in {path}")
    return sorted(str(key) for key in data.keys())


def suggest_keys(config_key, available_keys, limit):
    base_prefix = config_key.rsplit("-", 1)[0] if "-" in config_key else config_key
    same_family = [key for key in available_keys if key.startswith(base_prefix)]
    if same_family:
        return same_family[:limit]

    close_matches = difflib.get_close_matches(
        config_key, available_keys, n=limit, cutoff=0.4
    )

    return [key for key in close_matches if key != config_key]


def main():
    parser = argparse.ArgumentParser(description="Validate a config key against master YAML")
    parser.add_argument("--config-file", required=True, help="Path to master YAML file")
    parser.add_argument("--config-key", required=True, help="Config key to validate")
    parser.add_argument(
        "--max-suggestions",
        type=int,
        default=6,
        help="Maximum number of related keys to print",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config_file):
        print(f"ERROR: Config file not found: {args.config_file}")
        return 1

    try:
        available_keys = load_config_keys(args.config_file)
    except Exception as exc:
        print(f"ERROR: Could not read config file {args.config_file}: {exc}")
        return 1

    if args.config_key in available_keys:
        print(f"Validated config key: {args.config_key}")
        print(f"CONFIG_KEY_OK={args.config_key}")
        return 0

    suggestions = suggest_keys(args.config_key, available_keys, args.max_suggestions)

    print(
        f"ERROR: Config key '{args.config_key}' was not found in {args.config_file}"
    )
    if suggestions:
        print("Related available keys:")
        for key in suggestions:
            print(f"  - {key}")
    else:
        print("No close matches were found in this config file.")

    print(
        "Pick one of the related keys above or add the missing config upstream "
        "before continuing."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
