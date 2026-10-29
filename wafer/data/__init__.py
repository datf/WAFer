"""Data pipeline, schemas, log parsing, and anti-memorization augmentation."""

from wafer.data.schema import HTTPRequestState, QuestionSpec, WAFRecord
from wafer.data.synthetic import generate_synthetic_waf_samples
from wafer.data.rules import (
    ClassificationResult,
    FolderRules,
    Rule,
    RuleCondition,
    load_rules_for_folder,
)
from wafer.data.log_parser import (
    ParsedLogEntry,
    build_waf_training_record,
    find_log_files_in_folder,
    is_gzip_file,
    open_log_file,
    parse_and_classify_file,
    parse_caddy_log_line,
    parse_apache_log_line,
    parse_clf_log_line,
    parse_and_sanitize_log_line,
    logs_to_waf_records,
)

try:
    from wafer.data.permutation import PermutedWAFDataset, permute_waf_record
except (ImportError, OSError):
    PermutedWAFDataset = None
    permute_waf_record = None

__all__ = [
    "HTTPRequestState",
    "QuestionSpec",
    "WAFRecord",
    "PermutedWAFDataset",
    "permute_waf_record",
    "generate_synthetic_waf_samples",
    "ClassificationResult",
    "FolderRules",
    "Rule",
    "RuleCondition",
    "load_rules_for_folder",
    "ParsedLogEntry",
    "build_waf_training_record",
    "find_log_files_in_folder",
    "is_gzip_file",
    "open_log_file",
    "parse_and_classify_file",
    "parse_caddy_log_line",
    "parse_apache_log_line",
    "parse_clf_log_line",
    "parse_and_sanitize_log_line",
    "logs_to_waf_records",
]
