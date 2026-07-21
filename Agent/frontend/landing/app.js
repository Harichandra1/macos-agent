// The landing page deliberately has no build step. It reads only the public
// health endpoint. Production Prometheus metrics are protected and collected
// internally by Grafana Alloy; this page never exposes operational counters.

const $ = (id) => document.getElementById(id);

async function loadLiveStatus() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    const health = await response.json();
    $("metric-health").textContent = health.status === "ok" ? "online" : "degraded";
    $("metric-health").classList.toggle("metric-good", health.status === "ok");
    $("metric-health-note").textContent = health.ready
      ? "agent ready"
      : "API reachable · agent not ready";
    $("metric-requests").textContent = "private";
    $("metric-success").textContent = "Grafana";
    $("metric-cost").textContent = "tracked";
  } catch (_) {
    $("metric-health").textContent = "offline";
    $("metric-health-note").textContent = "health unavailable";
  }
}

loadLiveStatus();
