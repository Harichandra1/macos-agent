# Oracle Always Free deployment

This is the active production path for the macOS Agent. It keeps the FastAPI
service running on an Oracle Cloud Always Free Ampere A1 VM and keeps durable
application state in Neon Postgres. Caddy is the only public container; Alloy
collects private Prometheus metrics and Docker logs for Grafana Cloud.

## The VM does not build the image

`uv sync` over this dependency tree (langchain, langgraph, openai, psycopg)
peaks at roughly 1.5–2 GB of RAM. No Oracle Always Free shape can do that
reliably, and on the 1 GB `VM.Standard.E2.1.Micro` it will OOM-kill or thrash
indefinitely — this is the single most common way this deployment fails.

So GitHub Actions builds the image and publishes it to GHCR
([`.github/workflows/build-image.yml`](../../.github/workflows/build-image.yml)),
and the VM only pulls. **Never run `docker compose up --build` on the host.**

Images carry two tags: `latest`, and the full commit SHA they were built from.
`deploy.sh` and `rollback.sh` pin `APP_IMAGE_TAG` to the commit they check out,
so the running container and the checked-out compose/Caddy config are always
the same revision, and rollback never rebuilds anything.

Runtime is far lighter than the build: one stateless FastAPI worker at roughly
400–600 MB. The knowledge base is in Qdrant Cloud and all state is in Neon, so
nothing heavy runs on the VM.

## Cost-safety gate

This runbook is intentionally limited to Oracle Always Free resources. Before
creating anything, verify that the VM shape is within the Always Free quota
(Ampere A1 up to 4 OCPU / 24 GB total, or an E2.1.Micro), that no paid image,
block-volume upgrade, load balancer, reserved
IP, or paid support option is selected, and that the account shows an explicit
price of `$0`. Stop immediately if the console requests a paid upgrade or
shows a non-zero estimate. The application cannot guarantee a provider-side
billing outcome, so enable Oracle budget/spend alerts and review the estimate
before confirming each resource.

## Services

| Service | Exposure | Purpose |
|---|---|---|
| `app` | Docker network only | FastAPI, landing page, chat, migrations |
| `caddy` | ports 80/443 | HTTPS, certificate renewal, reverse proxy |
| `duckdns` | outbound only | Keeps the free DNS record pointed at the VM |
| `alloy` | Docker network only | Metrics remote-write and Loki log shipping |

## Choosing the VM shape

Use **`VM.Standard.A1.Flex`** (Ampere, arm64). Always Free covers 4 OCPU and
24 GB across all A1 instances, but that full allocation is the hardest to get —
request **1 OCPU / 6 GB**, which is ample for the runtime footprint above and
far more likely to be available.

If Oracle returns *"Out of host capacity"*, that is a region-wide shortage, not
an account problem. Retry across the other availability domains in your home
region, and retry at intervals — capacity is released continuously.

`VM.Standard.E2.1.Micro` (1/8 OCPU, 1 GB, amd64) also works now that the host
no longer builds, though with no headroom. The CI workflow publishes both
`linux/arm64` and `linux/amd64`, so either shape pulls the right image with no
config change.

Create the VM in the tenancy's home region, add an SSH public key, and allow
TCP ports 22, 80, and 443 in the VCN security list. Do not open port 8000,
9090, or 3000.

## One-time host setup

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git python3
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
git clone https://github.com/Harichandra1/macos-agent.git /opt/macos-agent
cd /opt/macos-agent
cp deploy/oracle/.env.production.example .env.production
chmod 600 .env.production
```

Log out and back in once after adding the user to the Docker group.

On the 1 GB E2 micro only, add swap so the kernel has fallback headroom:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Making the image pullable

Run the **Build and publish serving image** workflow once (push to `main`, or
trigger it manually from the Actions tab). Then, in GitHub → your profile →
Packages → `macos-agent` → Package settings, set visibility to **Public** so
the VM can pull anonymously.

GHCR packages default to private even when the repository is public. If you
prefer to keep the package private, authenticate the host instead, using a
personal access token with the `read:packages` scope:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u Harichandra1 --password-stdin
```

## External setup before first launch

1. Create the DuckDNS subdomain `macosagent-hari.duckdns.org`, point it to the
   Oracle VM public IP, and put the DuckDNS token in `.env.production`.
2. Create a Neon Postgres database and use its **direct** SSL connection string
   as `DATABASE_URL`; do not use a transaction-pooler URL for the long-lived
   LangGraph checkpointer connection.
3. Create a Google Web OAuth client. Add
   `https://macosagent-hari.duckdns.org` to Authorized JavaScript origins and
   add the hosted `/`, `/privacy`, and `/terms` pages to the consent branding.
4. Create a Grafana Cloud access policy token with Prometheus write and Loki
   write permissions. Copy the Prometheus remote-write URL/instance ID and
   Loki push URL/instance ID into `.env.production`.

## Launch and verify

The first launch uses the same script as every later update — it validates the
compose config, waits for the commit's image to exist in GHCR, pulls it, and
starts the stack:

```bash
cd /opt/macos-agent
chmod 600 .env.production
./deploy/oracle/deploy.sh
```

Then verify from your laptop:

```bash
curl -fsS https://macosagent-hari.duckdns.org/health
curl -fsS -o /dev/null -w '%{http_code}\n' \
  https://macosagent-hari.duckdns.org/metrics
```

The health request should return `status: ok`. Public `/metrics` should return
401; Alloy reads it privately using `METRICS_TOKEN`. Open the landing page,
follow the `/app` link, sign in with Google, and complete one real test turn.

Import `observability/grafana/dashboards/macos_agent.json` into Grafana Cloud
and select the Cloud Prometheus datasource. Logs appear in Explore through the
Cloud Loki datasource with `service` and `container` labels.

## Updates, rollback, and recovery

Normal update:

```bash
cd /opt/macos-agent
./deploy/oracle/deploy.sh
```

`deploy.sh` fast-forwards to `origin/main`, then waits for CI to finish
publishing that commit's image before switching. If you push and deploy in the
same minute, expect it to sit at *"Waiting for … to be published"* for a few
minutes — that is the workflow still running, not a hang.

Rollback to a known-good commit. Because CI tags every image with its commit
SHA, this repoints at an already-built image rather than rebuilding:

```bash
cd /opt/macos-agent
./deploy/oracle/rollback.sh <known-good-commit>
```

Migrations remain forward-only (see [`Agent/DEPLOY.md`](../../Agent/DEPLOY.md)),
so rolling the code back is safe against a newer schema.

If Oracle reclaims the VM, provision a new Always Free VM, point DuckDNS at its
new public IP, clone the repository, copy the same `.env.production`, and run:

```bash
REPO_URL=https://github.com/Harichandra1/macos-agent \
  ./deploy/oracle/recover.sh
```

Neon contains the users, credits, feedback, and LangGraph checkpoints, so VM
recovery does not require restoring application state.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Deploy hangs on *"Waiting for … to be published"*, then times out | CI has not published that commit — check the Actions run for a failure. The workflow builds every commit on `main`, so a missing tag means the run failed or never started. |
| `denied` / `manifest unknown` when pulling | The GHCR package is still private. Set it to Public, or `docker login ghcr.io` on the host with a `read:packages` token. |
| `exec format error` in the app container | Architecture mismatch — the tag resolved to the wrong platform. Confirm both matrix legs of the build workflow succeeded and the merge job ran. |
| The build takes forever or the box freezes | Something ran `--build` on the VM. Use `deploy.sh`; the host is not a build machine. |
| Google Sign-In fails after a redeploy | `APP_DOMAIN` must stay registered in the OAuth client's Authorized JavaScript origins. Changing the domain requires updating Google Cloud Console too. |
