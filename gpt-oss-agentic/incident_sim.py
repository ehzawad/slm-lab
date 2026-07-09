#!/usr/bin/env python3
"""Incident-Response simulator: a stateful, unsaturated agentic environment.

A SCENARIO injects a single ROOT-CAUSE fault at one NimbusWorks service. The
fault cascades to every transitive dependent, so downstream services *appear*
down while the real cause is upstream. The agent must diagnose the root, apply
the fault-type-specific fix within a tool-call budget, and verify recovery.

Anti-brute-force by construction:
  - bad_config      -> set_config(correct key/value) THEN restart. A bare
                       restart does NOT clear it.
  - bad_deploy      -> rollback. A restart re-launches the same bad code.
  - pool_exhausted  -> set_config(pool_max>=512) THEN restart. Restart alone
                       keeps the tiny pool.
  - dependency_outage(crash) -> restart the ROOT. Restarting a symptom
                       (cascaded dependent) does nothing.
So "restart everything" cannot fix config/deploy/pool faults, only crashes.

Everything derives from nimbus-agent/world.py so gold answers are exact and the
world is guaranteed absent from any pretraining corpus.
"""
import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nimbus-agent"))
import world  # SERVICES, ERRORS, dependents_of, SEED
from world import SERVICES

FAULT_TYPES = ("bad_config", "bad_deploy", "pool_exhausted", "crash")

# Per-service canonical config key + correct value for bad_config faults.
CONFIG_FIX = {
    "gatekeeper":    ("upstream_timeout_ms", 2000),
    "tokensmith":    ("validation_timeout_ms", 3000),
    "vaultkeep":     ("cache_ttl_s", 300),
    "ledgerline":    ("batch_size", 100),
    "quillbase":     ("statement_timeout_ms", 5000),
    "streamforge":   ("consumer_max_poll", 500),
    "glacierdocs":   ("fetch_timeout_ms", 4000),
    "pulseboard":    ("refresh_interval_s", 30),
    "mistral-cache": ("region_prefix_required", "true"),
    "courierbot":    ("retry_backoff_ms", 250),
}
# Services where a connection-pool fault is physically meaningful.
POOL_SERVICES = ("quillbase", "tokensmith", "ledgerline", "glacierdocs")
POOL_MAX_MIN = 512

FAULT_CODE = {
    "bad_config":     "NBX-6600",
    "bad_deploy":     "NBX-6700",
    "pool_exhausted": "NBX-2201",
    "crash":          "NBX-6900",
}
CASCADE_CODE = "NBX-3301"
FAULT_HINT = {
    "bad_config":     "config value '{key}' out of valid range; correct it and restart",
    "bad_deploy":     "error-rate spiked immediately after the last deploy; roll back the release",
    "pool_exhausted": "connection pool exhausted; raise pool_max to >=512 and restart",
    "crash":          "process not responding to health checks; needs a restart",
}


def _topo_order():
    """Dependencies before dependents (Kahn)."""
    indeg = {s: len(SERVICES[s][3]) for s in SERVICES}
    order, q = [], [s for s in SERVICES if indeg[s] == 0]
    while q:
        n = q.pop(0)
        order.append(n)
        for s in SERVICES:
            if n in SERVICES[s][3]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    q.append(s)
    return order


TOPO = _topo_order()
MUTATING = {"restart", "scale", "rollback", "set_config"}


class IncidentSim:
    """Holds mutable per-service state and exposes the tool callables."""

    def __init__(self, scenario, max_calls=15):
        self.scenario = scenario
        self.max_calls = max_calls
        self.calls = 0
        self.redundant = 0
        self.restart_counts = {}
        self.services = {}
        for s, meta in SERVICES.items():
            self.services[s] = {
                "status": "healthy",
                "error_rate": 0.001,
                "replicas": 3 if meta[0] == "Payments" else 2,
                "config_ok": True,
                "deployed_bad": False,
                "pool_max": POOL_MAX_MIN,
                "config": {},
                "root_fault": None,   # active fault type at THIS service, or None
                "fix_applied": False, # pre-restart fix done (config/pool)
                "cascade": False,
            }
        self._inject(scenario)
        self._recompute()

    # ---- fault injection ---------------------------------------------------
    def _inject(self, sc):
        st = self.services[sc["root_service"]]
        ft = sc["fault_type"]
        st["root_fault"] = ft
        if ft == "bad_config":
            st["config_ok"] = False
        elif ft == "bad_deploy":
            st["deployed_bad"] = True
        elif ft == "pool_exhausted":
            st["pool_max"] = 32

    # ---- health fixpoint over the DAG -------------------------------------
    def _recompute(self):
        for s in TOPO:
            st = self.services[s]
            deps = SERVICES[s][3]
            dep_bad = any(self.services[d]["status"] != "healthy" for d in deps)
            if st["root_fault"] is not None:
                ft = st["root_fault"]
                st["status"] = "down" if ft in ("bad_deploy", "crash") else "degraded"
                st["error_rate"] = 0.42
                st["cascade"] = False
            elif dep_bad:
                st["status"] = "down"
                st["error_rate"] = 0.31
                st["cascade"] = True
            else:
                st["status"] = "healthy"
                st["error_rate"] = 0.001
                st["cascade"] = False

    def _unhealthy(self):
        return sum(1 for st in self.services.values() if st["status"] != "healthy")

    def _progress_sig(self):
        return tuple((s, st["root_fault"], st["fix_applied"], st["deployed_bad"])
                     for s, st in self.services.items())

    # ---- TOOLS (read) ------------------------------------------------------
    def get_status(self, service):
        """Return health, error_rate, replica count and pool_max of a service."""
        if service not in self.services:
            return f"error: unknown service {service}"
        st = self.services[service]
        return json.dumps({"service": service, "status": st["status"],
                           "error_rate": round(st["error_rate"], 3),
                           "replicas": st["replicas"], "pool_max": st["pool_max"]})

    def get_logs(self, service):
        """Return the most recent error CODE and hint for a service (cause signal)."""
        if service not in self.services:
            return f"error: unknown service {service}"
        st = self.services[service]
        if st["root_fault"] is not None:
            ft = st["root_fault"]
            key = CONFIG_FIX.get(service, ("", ""))[0]
            return f"{FAULT_CODE[ft]} {FAULT_HINT[ft].format(key=key)}"
        if st["cascade"]:
            bad_dep = next((d for d in SERVICES[service][3]
                            if self.services[d]["status"] != "healthy"), "?")
            return (f"{CASCADE_CODE} upstream connect timeout; dependency "
                    f"'{bad_dep}' is unhealthy (symptom, not root cause)")
        return "no errors in the last 5m"

    def get_dependencies(self, service):
        """Return the direct dependencies and direct dependents of a service."""
        if service not in self.services:
            return f"error: unknown service {service}"
        return json.dumps({"service": service,
                           "depends_on": SERVICES[service][3],
                           "dependents": world.dependents_of(service)})

    def check_all(self):
        """Return the status of every service at once."""
        return json.dumps({s: self.services[s]["status"] for s in SERVICES})

    # ---- TOOLS (mutate) ----------------------------------------------------
    def restart(self, service):
        """Rolling-restart a service. Only clears crash faults, or config/pool
        faults whose corrected config was already applied via set_config."""
        if service not in self.services:
            return f"error: unknown service {service}"
        st = self.services[service]
        self.restart_counts[service] = self.restart_counts.get(service, 0) + 1
        rf = st["root_fault"]
        if rf == "crash":
            st["root_fault"] = None
        elif rf in ("bad_config", "pool_exhausted") and st["fix_applied"]:
            st["root_fault"] = None
        self._recompute()
        return f"restarted {service}; status now {st['status']}"

    def scale(self, service, replicas):
        """Set the replica count of a service."""
        if service not in self.services:
            return f"error: unknown service {service}"
        self.services[service]["replicas"] = int(replicas)
        self._recompute()
        return f"scaled {service} to {int(replicas)} replicas; status {self.services[service]['status']}"

    def rollback(self, service):
        """Roll a service back to its previous release. Fixes bad_deploy faults."""
        if service not in self.services:
            return f"error: unknown service {service}"
        st = self.services[service]
        if st["root_fault"] == "bad_deploy":
            st["deployed_bad"] = False
            st["root_fault"] = None
        self._recompute()
        return f"rolled back {service}; status now {st['status']}"

    def set_config(self, service, key, value):
        """Set a config value (takes effect on the next restart)."""
        if service not in self.services:
            return f"error: unknown service {service}"
        st = self.services[service]
        st["config"][key] = value
        sc = self.scenario
        if service == sc["root_service"]:
            if st["root_fault"] == "bad_config" and key == sc["fix_key"] and str(value) == str(sc["fix_value"]):
                st["config_ok"] = True
                st["fix_applied"] = True
            elif st["root_fault"] == "pool_exhausted" and key == "pool_max":
                try:
                    if int(value) >= POOL_MAX_MIN:
                        st["pool_max"] = int(value)
                        st["fix_applied"] = True
                except (TypeError, ValueError):
                    pass
        self._recompute()
        return f"set {key}={value} on {service} (restart required to apply)"

    # ---- dispatch + scoring ------------------------------------------------
    def dispatch(self, name, args):
        method = getattr(self, name, None)
        if method is None:
            return f"error: unknown tool {name}"
        self.calls += 1
        try:
            if name in MUTATING:
                before_bad, before_prog = self._unhealthy(), self._progress_sig()
                out = method(**args)
                if self._unhealthy() >= before_bad and self._progress_sig() == before_prog:
                    self.redundant += 1
            else:
                out = method(**args)
            return out
        except TypeError as e:
            self.calls -= 1
            return f"error: bad arguments for {name}: {e}"

    def score(self):
        all_healthy = all(st["status"] == "healthy" for st in self.services.values())
        correct = self.services[self.scenario["root_service"]]["root_fault"] is None
        within = self.calls <= self.max_calls
        return {
            "solved": bool(all_healthy and correct and within),
            "steps": self.calls,
            "correct_root_cause": bool(correct),
            "redundant_calls": self.redundant,
        }


# --------------------------------------------------------------------------
# Tool schema (OpenAI function-calling / MCP-compatible)
# --------------------------------------------------------------------------
def _svc_enum():
    return sorted(SERVICES.keys())


TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_status", "description": IncidentSim.get_status.__doc__,
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string", "enum": _svc_enum()}}, "required": ["service"]}}},
    {"type": "function", "function": {
        "name": "get_logs", "description": IncidentSim.get_logs.__doc__,
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string", "enum": _svc_enum()}}, "required": ["service"]}}},
    {"type": "function", "function": {
        "name": "get_dependencies", "description": IncidentSim.get_dependencies.__doc__,
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string", "enum": _svc_enum()}}, "required": ["service"]}}},
    {"type": "function", "function": {
        "name": "check_all", "description": IncidentSim.check_all.__doc__,
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "restart", "description": IncidentSim.restart.__doc__,
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string", "enum": _svc_enum()}}, "required": ["service"]}}},
    {"type": "function", "function": {
        "name": "scale", "description": IncidentSim.scale.__doc__,
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string", "enum": _svc_enum()},
            "replicas": {"type": "integer"}}, "required": ["service", "replicas"]}}},
    {"type": "function", "function": {
        "name": "rollback", "description": IncidentSim.rollback.__doc__,
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string", "enum": _svc_enum()}}, "required": ["service"]}}},
    {"type": "function", "function": {
        "name": "set_config", "description": IncidentSim.set_config.__doc__,
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string", "enum": _svc_enum()},
            "key": {"type": "string"}, "value": {}}, "required": ["service", "key", "value"]}}},
]


# --------------------------------------------------------------------------
# Scenario generation (deterministic)
# --------------------------------------------------------------------------
def _make_scenario(root, ft):
    key, val = CONFIG_FIX[root]
    if ft == "pool_exhausted":
        key, val = "pool_max", POOL_MAX_MIN
    cascade = sorted(_transitive_dependents(root))
    depth = _cascade_depth(root)
    return {
        "id": f"{ft}:{root}",
        "root_service": root,
        "fault_type": ft,
        "fix_key": key,
        "fix_value": val,
        "error_code": FAULT_CODE[ft],
        "cascade": cascade,
        "cascade_depth": depth,
    }


def _transitive_dependents(root):
    seen, stack = set(), [root]
    while stack:
        n = stack.pop()
        for d in world.dependents_of(n):
            if d not in seen:
                seen.add(d)
                stack.append(d)
    return seen


def _cascade_depth(root):
    """Longest dependent chain length below root (0 = leaf, no dependents)."""
    deps = world.dependents_of(root)
    if not deps:
        return 0
    return 1 + max(_cascade_depth(d) for d in deps)


def build_scenarios(n=24):
    """~n diverse, deterministic scenarios spanning fault type x service x depth."""
    rng = world.rng()
    combos = []
    for ft in FAULT_TYPES:
        pool = POOL_SERVICES if ft == "pool_exhausted" else list(SERVICES.keys())
        for root in pool:
            combos.append((root, ft))
    # deterministic shuffle, then favour cascade diversity by stable sort on depth
    rng.shuffle(combos)
    combos.sort(key=lambda rc: (_cascade_depth(rc[0]), rc[1]))
    scenarios, seen = [], set()
    # round-robin over fault types to keep the set balanced
    by_ft = {ft: [c for c in combos if c[1] == ft] for ft in FAULT_TYPES}
    while len(scenarios) < n and any(by_ft.values()):
        for ft in FAULT_TYPES:
            if by_ft[ft] and len(scenarios) < n:
                root, _ = by_ft[ft].pop(0)
                sc = _make_scenario(root, ft)
                if sc["id"] not in seen:
                    seen.add(sc["id"])
                    scenarios.append(sc)
    return scenarios


# --------------------------------------------------------------------------
# Expert (gold) trajectory: diagnose -> fix -> verify
# --------------------------------------------------------------------------
def expert_trajectory(scenario):
    """Gold tool sequence [(name, args), ...] that solves the scenario for SFT."""
    root = scenario["root_service"]
    ft = scenario["fault_type"]
    seq = [("check_all", {})]
    if scenario["cascade"]:
        symptom = scenario["cascade"][-1]  # a deep dependent that merely looks broken
        seq.append(("get_logs", {"service": symptom}))
        seq.append(("get_dependencies", {"service": symptom}))
    seq.append(("get_status", {"service": root}))
    seq.append(("get_logs", {"service": root}))  # confirms the fault type
    if ft == "bad_config":
        seq.append(("set_config", {"service": root,
                                    "key": scenario["fix_key"], "value": scenario["fix_value"]}))
        seq.append(("restart", {"service": root}))
    elif ft == "pool_exhausted":
        seq.append(("set_config", {"service": root, "key": "pool_max", "value": POOL_MAX_MIN}))
        seq.append(("restart", {"service": root}))
    elif ft == "bad_deploy":
        seq.append(("rollback", {"service": root}))
    elif ft == "crash":
        seq.append(("restart", {"service": root}))
    seq.append(("check_all", {}))
    return seq


if __name__ == "__main__":
    scs = build_scenarios(24)
    print(f"built {len(scs)} scenarios")
    from collections import Counter
    print("fault-type mix:", dict(Counter(s["fault_type"] for s in scs)))
    print("cascade-depth mix:", dict(Counter(s["cascade_depth"] for s in scs)))
    # self-check: every expert trajectory solves its scenario
    fails = []
    for sc in scs:
        sim = IncidentSim(sc)
        for name, args in expert_trajectory(sc):
            sim.dispatch(name, args)
        r = sim.score()
        if not r["solved"]:
            fails.append((sc["id"], r))
    print(f"expert solves {len(scs)-len(fails)}/{len(scs)} scenarios")
    if fails:
        print("FAILURES:", fails)
