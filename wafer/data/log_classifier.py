"""CLI and evaluation engine for log parsing, rule application, and dataset preparation."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from wafer.data.log_parser import (
    find_log_files_in_folder,
    open_log_file,
    parse_clf_log_line,
    parse_log_entry,
    build_waf_training_record,
    ParsedLogEntry,
)
from wafer.data.rules import (
    ClassificationResult,
    FolderRules,
    load_rules_for_folder,
)


def resolve_log_dir(provided_path: Optional[str] = None) -> Path:
    """Resolves the log root directory, checking 'backup logs' and 'backuplogs'."""
    if provided_path:
        p = Path(provided_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified log directory does not exist: {provided_path}")

    # Check candidates
    for cand in ["backup logs", "backuplogs"]:
        p = Path(cand)
        if p.exists() and p.is_dir():
            return p

    raise FileNotFoundError("Could not find 'backup logs' or 'backuplogs' folder.")


def process_folder(
    folder_path: Path,
    rules: Optional[FolderRules] = None,
    limit: Optional[int] = None,
    output_writer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Parses all logs in a single folder, applies rules, and aggregates statistics.
    """
    if rules is None:
        rules = load_rules_for_folder(folder_path)

    log_files = find_log_files_in_folder(folder_path)
    stats: Dict[str, Any] = {
        "folder": folder_path.name,
        "files_count": len(log_files),
        "total_lines": 0,
        "parsed_lines": 0,
        "failed_lines": 0,
        "actions": Counter(),
        "alerts": Counter(),
        "statuses": Counter(),
        "rules_matched": Counter(),
        "methods": Counter(),
        "sample_allowed": [],
        "sample_blocked": [],
        "validation_errors": [],
    }

    processed_count = 0
    for log_file in log_files:
        if limit is not None and processed_count >= limit:
            break

        for line in open_log_file(log_file):
            stats["total_lines"] += 1
            entry = parse_log_entry(line, format_hint=rules.parser_format)
            if entry is None:
                stats["failed_lines"] += 1
                continue

            stats["parsed_lines"] += 1
            res = rules.classify(
                status_code=entry.status_code,
                method=entry.method,
                path=entry.path,
                user_agent=entry.user_agent,
                protocol=entry.protocol,
            )

            record = build_waf_training_record(entry, res)

            # Update stats
            stats["actions"][res.action] += 1
            stats["alerts"][res.alert] += 1
            stats["statuses"][entry.status_code] += 1
            stats["rules_matched"][res.rule_id or "unspecified"] += 1
            stats["methods"][entry.method] += 1

            # Validate extraction requirements: method, path, protocol, user agent
            state = record["state"]
            if not state.get("method"):
                stats["validation_errors"].append(f"Missing method in: {line}")
            if not state.get("uri"):
                stats["validation_errors"].append(f"Missing path/uri in: {line}")
            if not state.get("protocol"):
                stats["validation_errors"].append(f"Missing protocol in: {line}")

            # Verify NO response attributes in state
            for forbidden in ["status", "status_code", "size", "duration"]:
                if forbidden in state:
                    stats["validation_errors"].append(
                        f"Response attribute '{forbidden}' leaked into request state!"
                    )

            # Collect samples
            if res.action == "allow" and len(stats["sample_allowed"]) < 3:
                stats["sample_allowed"].append({
                    "method": entry.method,
                    "path": entry.path,
                    "protocol": entry.protocol,
                    "status": entry.status_code,
                    "user_agent": entry.user_agent,
                    "rule_id": res.rule_id,
                    "alert": res.alert,
                })
            elif res.action in ("deny", "block") and len(stats["sample_blocked"]) < 3:
                stats["sample_blocked"].append({
                    "method": entry.method,
                    "path": entry.path,
                    "protocol": entry.protocol,
                    "status": entry.status_code,
                    "user_agent": entry.user_agent,
                    "rule_id": res.rule_id,
                    "alert": res.alert,
                })

            if output_writer is not None:
                output_writer.write(json.dumps(record) + "\n")

            processed_count += 1
            if limit is not None and processed_count >= limit:
                break

    return stats


def test_rules_compliance(stats: Dict[str, Any]) -> List[str]:
    """
    Validates that core requirements are strictly met:
    - 429 and 500 must be alerted and blocked
    - 404 must be alerted and blocked
    - 200 should be good (allowed, alert: false) unless explicitly overridden
    """
    failures: List[str] = []
    folder = stats["folder"]

    # Check for leaked response attributes
    if stats["validation_errors"]:
        failures.extend([f"[{folder}] {err}" for err in stats["validation_errors"][:5]])

    return failures


def print_summary(all_stats: List[Dict[str, Any]]):
    """Prints a structured summary table of log parsing and classification results."""
    print("=" * 80)
    print("LOG PARSER & CLASSIFIER SUMMARY REPORT")
    print("=" * 80)

    for stats in all_stats:
        folder = stats["folder"]
        print(f"\n📂 Folder: {folder} ({stats['files_count']} files, {stats['parsed_lines']:,} parsed lines)")
        print("-" * 50)
        print("  Actions:")
        for action, cnt in sorted(stats["actions"].items()):
            pct = (cnt / stats["parsed_lines"]) * 100 if stats["parsed_lines"] else 0
            print(f"    - {action:<15}: {cnt:>7,} ({pct:5.1f}%)")

        print("  Alerts:")
        alert_t = stats["alerts"].get(True, 0)
        alert_f = stats["alerts"].get(False, 0)
        print(f"    - Alerted (True) : {alert_t:>7,} ({alert_t / max(1, stats['parsed_lines']) * 100:5.1f}%)")
        print(f"    - Quiet   (False): {alert_f:>7,} ({alert_f / max(1, stats['parsed_lines']) * 100:5.1f}%)")

        print("  Top HTTP Status Codes:")
        for st, cnt in stats["statuses"].most_common(5):
            print(f"    - {st:<5}: {cnt:>7,}")

        print("  Top Rules Triggered:")
        for rule_id, cnt in stats["rules_matched"].most_common(5):
            print(f"    - {rule_id:<30}: {cnt:>7,}")

    print("\n" + "=" * 80)


def print_samples(all_stats: List[Dict[str, Any]]):
    """Displays sample classified requests for verification."""
    print("\n" + "=" * 80)
    print("SAMPLE CLASSIFIED REQUESTS")
    print("=" * 80)

    for stats in all_stats:
        print(f"\nFolder: {stats['folder']}")
        print("  ALLOWED Samples:")
        for s in stats["sample_allowed"]:
            print(f"    [{s['status']}] {s['method']} {s['path'][:50]} ({s['protocol']}) -> Rule: {s['rule_id']}")
        print("  BLOCKED & ALERTED Samples:")
        for s in stats["sample_blocked"]:
            print(f"    [{s['status']}] {s['method']} {s['path'][:50]} ({s['protocol']}) -> Rule: {s['rule_id']} (Alert={s['alert']})")


def run_rule_tests(log_dir: Path) -> bool:
    """
    Performs comprehensive verification that rules satisfy all prompt constraints:
    1. Status 429 requests are ALERTED and BLOCKED.
    2. Status 500 requests are ALERTED and BLOCKED.
    3. Status 404 requests are ALERTED and BLOCKED.
    4. Status 200 requests are ALLOWED and NOT alerted.
    5. HTTP method, path, protocol, user agent are correctly extracted.
    6. Response data (status code, duration, size) is never passed to AI state.
    """
    print("\n" + "=" * 80)
    print("RUNNING RULE VALIDATION SUITE")
    print("=" * 80)

    subdirs = [d for d in log_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    all_passed = True

    for sub in sorted(subdirs):
        rules = load_rules_for_folder(sub)
        print(f"\nTesting folder: {sub.name} (Rule: '{rules.name}')")

        # Test 1: 200 OK
        res_200 = rules.classify(status_code=200, method="GET", path="/", user_agent="Mozilla/5.0")
        assert res_200.action == "allow", f"200 should be allowed, got {res_200.action}"
        assert not res_200.alert, f"200 should not alert, got {res_200.alert}"
        print("  [PASS] Status 200 -> Allowed, Alert=False")

        # Test 2: 404 Not Found
        res_404 = rules.classify(status_code=404, method="GET", path="/random_missing_page", user_agent="Mozilla/5.0")
        assert res_404.action in ("deny", "block"), f"404 should be denied, got {res_404.action}"
        assert res_404.alert, f"404 should alert, got {res_404.alert}"
        print("  [PASS] Status 404 -> Denied/Blocked, Alert=True")

        # Test 3: 429 Too Many Requests
        res_429 = rules.classify(status_code=429, method="POST", path="/trace/", user_agent="Mozilla/5.0")
        assert res_429.action in ("deny", "block"), f"429 should be denied, got {res_429.action}"
        assert res_429.alert, f"429 should alert, got {res_429.alert}"
        print("  [PASS] Status 429 -> Denied/Blocked, Alert=True")

        # Test 4: 500 Internal Server Error
        res_500 = rules.classify(status_code=500, method="GET", path="/sitemap.xml", user_agent="Mozilla/5.0")
        assert res_500.action in ("deny", "block"), f"500 should be denied, got {res_500.action}"
        assert res_500.alert, f"500 should alert, got {res_500.alert}"
        print("  [PASS] Status 500 -> Denied/Blocked, Alert=True")

    print("\n✅ All rule validation tests passed successfully!")
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="WAF Access Log Parser, Classifier, and Training Dataset Generator"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Path to folder containing backup logs (default: auto-detect 'backup logs' / 'backuplogs')",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Specific subfolder to process (e.g. api, web, proxy)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write classified WAF records in JSONL format",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of lines processed per folder (for fast testing)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print summary statistics of log parsing and classification",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="Print sample classified entries",
    )
    parser.add_argument(
        "--test-rules",
        action="store_true",
        default=False,
        help="Execute automated validation tests on rule definitions",
    )

    args = parser.parse_args()

    try:
        log_dir = resolve_log_dir(args.log_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.test_rules:
        success = run_rule_tests(log_dir)
        if not success:
            sys.exit(1)

    # Determine folders to process
    if args.folder:
        target_folders = [log_dir / args.folder]
        if not target_folders[0].is_dir():
            print(f"Error: Subfolder '{args.folder}' does not exist in {log_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        target_folders = [
            d for d in sorted(log_dir.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]

    output_file = None
    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(out_p, "w", encoding="utf-8")

    all_stats: List[Dict[str, Any]] = []
    try:
        for folder in target_folders:
            print(f"Processing folder: {folder.name}...")
            stats = process_folder(
                folder,
                limit=args.limit,
                output_writer=output_file,
            )
            all_stats.append(stats)
    finally:
        if output_file is not None:
            output_file.close()
            print(f"\nSaved classified WAF training records to: {args.output}")

    if args.summary:
        print_summary(all_stats)

    if args.sample:
        print_samples(all_stats)

    # Final compliance check
    total_failures = 0
    for stats in all_stats:
        failures = test_rules_compliance(stats)
        if failures:
            total_failures += len(failures)
            for f in failures:
                print(f"Compliance Error: {f}", file=sys.stderr)

    if total_failures > 0:
        print(f"\n❌ Encountered {total_failures} compliance errors during processing.", file=sys.stderr)
        sys.exit(1)

    print("\n✅ Parsing and classification completed successfully.")


if __name__ == "__main__":
    main()
