# WAFer - Gemma-Clef System One WAF Decision Engine

Proactive use of AI to defend web applications against AI and automated threats.

## Access Log Parser & Rule Classifier

The log processing subsystem parses real-world access logs (Apache Combined Log Format and Caddy JSON logs), transparently decompresses `.gz` and text archives, extracts incoming request features, and generates labeled datasets to train the AI engine into blocking and alerting malicious requests.

### Core Architecture & Guarantees

1. **Simplified Request Inputs**:
   - `HTTP Method`: GET, POST, HEAD, PUT, DELETE, etc.
   - `Target URI/Path`: Path and query string (e.g. `/api/v1/resource`, `/webhook.php?timestamp=...`).
   - `Protocol`: HTTP/1.1, HTTP/2.0, HTTP/3.0.
   - `User-Agent`: Client User-Agent header string.

2. **Simplified Decision Outputs & Targets**:
   - `action`: `allow` | `deny`
   - `alert`: `true` | `false`
   - **Confidence Scores**: The neural engine emits calibrated probabilities $P(\text{deny})$ and $P(\text{alert})$ per request.
   - **Domain Invariant Guarantee**: Blocking strictly implies alerting ($P(\text{deny}) \ge 0.50 \implies \text{alert} = \text{true}$). An alert may be raised independently on suspicious requests without blocking ($P(\text{alert}) \ge 0.50, P(\text{deny}) < 0.50 \to \text{ALERT (200)}$).

3. **Pre-Response State Sanitization (Zero Data Leakage)**:
   - In production inference, the AI engine evaluates incoming requests *before* the upstream backend generates a response.
   - Downstream response attributes (`status`, `duration`, `size`, response headers) are strictly stripped from the input state fed to the neural backbone.
   - Response status codes in training logs are used solely to establish ground-truth labels and evaluation metrics.

4. **Classification Ground Truth (from Logs)**:
   - **Status 200 / 3xx (Benign)**: Allowed traffic (`action: allow`, `alert: false`).
   - **Status 404 (Probe / Scan)**: Denied and alerted (`action: deny`, `alert: true`).
   - **Status 429 (Rate Limit / Abuse)**: Denied and alerted (`action: deny`, `alert: true`).
   - **Status 500 (Injection / Server Error)**: Denied and alerted (`action: deny`, `alert: true`).

5. **Per-Folder Rule Specification**:
   Each subfolder in `backuplogs/` contains a `rules.yaml` configuring:
   - Parser format (`apache`/`clf` or `caddy`/`json`).
   - Custom overrides (e.g. whitelisting authorized webhook endpoints, denying known crawler/scanner signatures).
   - Status code defaults.

---

### Running the Parser & Classifier

#### Running via Make (Recommended)

A standard `Makefile` auto-detects `podman` or `docker`:

```bash
# Display help and available commands
make help

# Build the container image
make build

# Run the full test suite in container
make test

# Run unit tests or integration tests only
make test-unit
make test-integration

# Inspect sample traffic logs with confidence scores
make inspect

# Run log classifier and rules summary
make classify

# Update or verify uv.lock
make lock
make check-lock

# Clean caches and temporary checkpoints
make clean
```

#### Direct Python Execution

```bash
# Run rule validation and display summary across all folders
python -m wafer.data.log_classifier --test-rules --summary

# Inspect sample classified requests
python -m wafer.data.log_classifier --sample

# Export classified training records to JSONL
python -m wafer.data.log_classifier --output data/classified_logs.jsonl

# Process a specific folder with a record limit
python -m wafer.data.log_classifier --folder api --limit 500 --summary
```

#### Running Inside Docker / Podman Container directly

```bash
# Run log classifier with default test suite and summary report
./run_parser_in_container.sh

# Run with custom arguments inside container
./run_parser_in_container.sh --output data/waf_train_records.jsonl --summary

# Run unit tests inside container
./run_in_container.sh pytest tests/unit/test_rules.py tests/unit/test_log_classifier.py -v
```
