# Observability (Phase 4)

Metrics, dashboards, and the eval regression gate for the macOS agent.

## What emits what

- **Metrics** — the backend exposes Prometheus text at `GET /metrics`
  (`app/metrics.py`). Counters/histograms cover request outcomes, tokens,
  cost, provider errors, credit refusals, PII/abuse guardrail hits, and
  feedback.
- **Logs** — one JSON object per line to stdout (`app/logging.py`), including
  per-turn `cost_usd`, `error_type`, `pii_counts`, and feedback. Failures carry
  an `exc` field with the full traceback. Shipped to Loki by Alloy in production
  (see below); locally just read the process output.
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

## Production: Grafana Cloud + Alloy

Production works differently from the local stack above, and the difference is
the usual source of "why is the dashboard empty".

**The app never pushes anything.** Grafana Alloy runs beside it on the Oracle VM
([`deploy/oracle/alloy/config.alloy`](../deploy/oracle/alloy/config.alloy)) and:

- scrapes `app:8000/metrics` every 30s over the private Compose network, sending
  `Authorization: Bearer $METRICS_TOKEN`, then remote-writes to Grafana Cloud;
- tails every container's JSON logs via the mounted Docker socket and pushes them
  to Loki with `service` and `container` labels.

**Public `/metrics` returning 401 is correct, not a fault.** When `APP_ENV` is
production the route requires the bearer token; only Alloy reads it, privately.
To check it by hand:

```bash
curl -H "Authorization: Bearer $METRICS_TOKEN" https://<domain>/metrics
```

The same token turns the deploy gate into its strongest form:

```bash
METRICS_TOKEN=... scripts/smoke_test.sh https://<domain>
```

### Importing the dashboard

Import [`grafana/dashboards/macos_agent.json`](grafana/dashboards/macos_agent.json)
and **pick your Cloud Prometheus datasource** from the `datasource` variable at
the top. The dashboard deliberately ships with that variable unpinned: it used to
hardcode the local provisioned uid (`macos-agent-prom`), which is dangling
anywhere else and left every panel showing "No data".

Logs are **not** on the dashboard — it is entirely Prometheus. Read them in
Grafana Cloud's Explore against the Loki datasource, filtering on
`{service="app"}`.

### When panels are empty

Check in this order:

1. **Has there been traffic?** Most panels use `increase(...)` over a window; a
   deployment with no successful chat turns has nothing to plot. This is by far
   the most common answer.
2. **Is the datasource variable set** to your Cloud Prometheus?
3. **Is Alloy healthy?** Its component-health UI is published on the VM's
   loopback only:
   ```bash
   ssh -N -L 12345:127.0.0.1:12345 ubuntu@<vm>   # then http://localhost:12345
   ```
   Never open 12345 in the VCN. `docker compose ... logs alloy` shows the same
   errors in logfmt.
4. **Counters reset on redeploy** — the container restarts, so raw counter
   values start from zero. Panels use `increase()`/`rate()` to tolerate that.

Note that `observability/prometheus.yml` (the local stack) has no `authorization`
block, so pointing local Prometheus at a production app 401s on every scrape.

## Eval regression gate

`Agent/eval/regression_gate.py` compares the interactive benchmark's headline
number (agent vs GPT-4o resolution rate) against a committed baseline and
exits non-zero on a regression. Run it after any prompt or graph change:

```bash
.venv/bin/python Agent/eval/regression_gate.py            # uses cached results
.venv/bin/python Agent/eval/regression_gate.py --run      # regenerate first (costs API calls)
```
