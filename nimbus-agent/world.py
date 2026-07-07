"""NimbusWorks: a deterministic fictional company world model.
Everything downstream (CPT corpus, SFT/reasoning/tool/MCP datasets, all evals, the MCP
server state, and incident simulations) derives from these tables, so gold answers are
exact and the corpus is guaranteed absent from any pretraining data."""
import random

SEED = 7
COMPANY = "NimbusWorks"
CLI = "nbx"

TEAMS = {
    "Core Platform":  {"lead": "Farida Rahman",  "oncall_slack": "#oncall-coreplat"},
    "Data Services":  {"lead": "Tomasz Krol",    "oncall_slack": "#oncall-data"},
    "Edge & Traffic": {"lead": "Ines Delgado",   "oncall_slack": "#oncall-edge"},
    "Payments":       {"lead": "Kwame Mensah",   "oncall_slack": "#oncall-payments"},
}

# service -> (team, port, slo_ms, depends_on, description)
SERVICES = {
    "gatekeeper":   ("Edge & Traffic", 7443, 120, ["tokensmith"],
                     "API gateway; terminates TLS and routes external requests"),
    "tokensmith":   ("Core Platform", 7020, 40,  ["vaultkeep"],
                     "issues and validates NimbusAuth session tokens"),
    "vaultkeep":    ("Core Platform", 7030, 25,  [],
                     "secrets and key-material store"),
    "ledgerline":   ("Payments",      7210, 200, ["tokensmith", "quillbase"],
                     "payment ledger; records all transactions"),
    "quillbase":    ("Data Services", 7305, 60,  [],
                     "primary OLTP database proxy"),
    "streamforge":  ("Data Services", 7320, 90,  ["quillbase"],
                     "event streaming and change-data-capture pipeline"),
    "glacierdocs":  ("Data Services", 7340, 300, ["quillbase"],
                     "document archive and retrieval service"),
    "pulseboard":   ("Core Platform", 7050, 150, ["streamforge"],
                     "internal metrics and alerting dashboard"),
    "mistral-cache":("Edge & Traffic", 7460, 15, [],
                     "edge response cache (LRU, 64GB per node)"),
    "courierbot":   ("Payments",      7230, 250, ["ledgerline", "streamforge"],
                     "sends receipts and payment notifications"),
}

# error code -> (service, meaning, runbook fix)
ERRORS = {
    "NBX-1101": ("tokensmith", "token signing key rotation failed",
                 "run `nbx keys rotate --service tokensmith --force`, then restart tokensmith"),
    "NBX-1102": ("tokensmith", "token validation latency above SLO",
                 "check vaultkeep health first; if healthy, scale tokensmith to 4 replicas"),
    "NBX-2201": ("quillbase", "connection pool exhausted",
                 "raise pool_max to 512 via `nbx config set quillbase pool_max 512` and restart"),
    "NBX-2202": ("quillbase", "replication lag above 30s",
                 "pause streamforge consumers, let lag drain below 5s, then resume"),
    "NBX-3301": ("gatekeeper", "upstream connect timeout",
                 "check tokensmith health; gatekeeper retries do NOT apply to POST routes"),
    "NBX-3302": ("mistral-cache", "cache hit rate below 40%",
                 "verify cache keys include the region prefix; flush with `nbx cache flush --node all`"),
    "NBX-4401": ("ledgerline", "double-entry imbalance detected",
                 "freeze writes with `nbx ledger freeze`, run `nbx ledger reconcile`, page Payments lead"),
    "NBX-5501": ("streamforge", "consumer group stalled",
                 "run `nbx stream rewind --group <group> --to-checkpoint` then unpause"),
}

POLICIES = [
    ("deploy-freeze", "Production deploys are frozen every Friday 18:00 UTC through Monday 06:00 UTC."),
    ("sev1", "A Sev1 incident requires paging the service team lead within 10 minutes and posting to #incident-bridge."),
    ("rollback", "Any deploy showing error-rate increase above 2% within 15 minutes must be rolled back with `nbx deploy rollback <service>`."),
    ("secrets", "Secrets may only be read through vaultkeep; direct file access is a fireable offense."),
    ("capacity", "Services must run at least 2 replicas in prod; payments services at least 3."),
]

CLI_COMMANDS = {
    "nbx status <service>": "show health, replica count, error rate, p99 latency of a service",
    "nbx logs <service> --tail N": "show last N log lines of a service",
    "nbx deploy rollback <service>": "roll a service back to the previous release",
    "nbx scale <service> --replicas N": "set replica count",
    "nbx config set <service> <key> <value>": "set a config value (requires restart)",
    "nbx restart <service>": "rolling restart of a service",
    "nbx keys rotate --service <service> --force": "force key-material rotation",
    "nbx ledger freeze": "freeze ledgerline writes (Sev1 payments action)",
    "nbx cache flush --node all": "flush mistral-cache on every node",
}

def rng():
    return random.Random(SEED)

def dependents_of(svc):
    return sorted(s for s, v in SERVICES.items() if svc in v[3])
