# Observability (Phase 4)

Metrics, dashboards, and the eval regression gate for the macOS agent.

## What emits what

- **Metrics** — the backend exposes Prometheus text at `GET /metrics`
  (`app/metrics.py`). Counters/histograms cover request outcomes, tokens,
  cost, provider errors, credit refusals, PII/abuse guardrail hits, and
  feedback.
- **Logs** — one JSON object per line to stdout (`app/logging.py`), including
  per-turn `cost_usd`, `error_type`, `pii_counts`, and feedback. Ship to Loki
  in deployment (Phase 5); locally just read the process output.
- **Ledger** — `usage_log` / `credit_accounts` tables hold the exact per-user
  cost that enforces the `$0.25`/month cap. Prometheus is the aggregate view;
  the DB is the source of truth for billing.

## Run the stack locally

```bash
# 1. Start the agent (from repo root)
.venv/bin/uvicorn app.main:app --app-dir Agent/backend --port 8000

# 2. Start Prometheus + Grafana
docker compose -f observability/docker-compose.yml up

# Grafana    → http://localhost:3000   (dashboard: "macOS Agent — Overview")
# Prometheus → http://localhost:9090
# Raw metrics → http://localhost:8000/metrics
```

Grafana boots with the Prometheus datasource and the dashboard already
provisioned (`grafana/provisioning/`). Anonymous viewer access is on for local
convenience — lock it down / use Grafana Cloud's hosted stack in Phase 5.

The dashboard panels map to the Phase 4 requirements: monthly spend, error
rates, and bad-feedback spikes, plus token throughput, latency percentiles,
and guardrail activity.

## Eval regression gate

`Agent/eval/regression_gate.py` compares the interactive benchmark's headline
number (agent vs GPT-4o resolution rate) against a committed baseline and
exits non-zero on a regression. Run it after any prompt or graph change:

```bash
.venv/bin/python Agent/eval/regression_gate.py            # uses cached results
.venv/bin/python Agent/eval/regression_gate.py --run      # regenerate first (costs API calls)
```
