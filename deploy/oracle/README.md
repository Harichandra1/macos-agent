# Oracle Always Free deployment

This is the active production path for the macOS Agent. It keeps the FastAPI
service running on an Oracle Cloud Always Free Ampere A1 VM and keeps durable
application state in Neon Postgres. Caddy is the only public container; Alloy
collects private Prometheus metrics and Docker logs for Grafana Cloud.

## Cost-safety gate

This runbook is intentionally limited to Oracle Always Free resources. Before
creating anything, verify that the VM shape is Ampere A1 within the Always
Free quota, that no paid image, block-volume upgrade, load balancer, reserved
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

## One-time Oracle setup

Create an Oracle Cloud Always Free Ampere A1 VM in the tenancy's home region,
using no more than 2 OCPUs and 12 GB RAM. Add an SSH public key and allow TCP
ports 22, 80, and 443 in the VCN security list. Do not open port 8000, 9090, or
3000.

Install Git and Docker, then clone the repository:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
git clone https://github.com/Harichandra1/macos-agent.git /opt/macos-agent
cd /opt/macos-agent
cp deploy/oracle/.env.production.example .env.production
chmod 600 .env.production
```

Log out and back in once after adding the user to the Docker group.

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

```bash
cd /opt/macos-agent
chmod 600 .env.production
docker compose --env-file .env.production \
  -f deploy/oracle/docker-compose.yml config >/dev/null
docker compose --env-file .env.production \
  -f deploy/oracle/docker-compose.yml up -d --build
docker compose --env-file .env.production \
  -f deploy/oracle/docker-compose.yml ps
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

Rollback to a known-good commit:

```bash
cd /opt/macos-agent
./deploy/oracle/rollback.sh <known-good-commit>
```

If Oracle reclaims the VM, provision a new Always Free VM, point DuckDNS at its
new public IP, clone the repository, copy the same `.env.production`, and run:

```bash
REPO_URL=https://github.com/Harichandra1/macos-agent \
  ./deploy/oracle/recover.sh
```

Neon contains the users, credits, feedback, and LangGraph checkpoints, so VM
recovery does not require restoring application state.
