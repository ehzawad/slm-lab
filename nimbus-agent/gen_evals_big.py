#!/usr/bin/env python3
"""Enlarged NimbusWorks eval sets so 5-15% effects are detectable.
Programmatically generates from world.py tables and OVERWRITES:
  eval_domain_qa.json (~100), eval_if.json (~120),
  eval_halluc.json (~120: 60 real + 60 fake), eval_tools.json (~60).
Schemas match eval_nimbus.py exactly:
  domain: {q, a}                 -- a is an exact substring of any correct answer
  if:     {q, check}             -- check '' means concise-answer check (1-25 words)
  halluc: {q, fake, a_substr}    -- scorer matches a_substr.lower()[:15]; refusal regex
                                    must not fire on correct real answers
  tools:  {q, tool, arg_substr}
CPU only. Run: python gen_evals_big.py
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world import COMPANY, TEAMS, SERVICES, ERRORS, POLICIES, dependents_of

HERE = os.path.dirname(os.path.abspath(__file__))
R = random.Random(20260707)

POLICY = dict(POLICIES)

# ---------------------------------------------------------------- domain ~100
def domain_qa():
    qa = []
    for svc, (team, port, slo, deps, desc) in SERVICES.items():
        lead, chan = TEAMS[team]["lead"], TEAMS[team]["oncall_slack"]
        qa.append({"q": f"Which team owns the {svc} service at {COMPANY}?", "a": team})
        qa.append({"q": f"What port does {svc} listen on?", "a": str(port)})
        qa.append({"q": f"What is the p99 latency SLO of {svc} in milliseconds?", "a": str(slo)})
        qa.append({"q": f"Who is the team lead responsible for the {svc} service?", "a": lead})
        qa.append({"q": f"What is the on-call Slack channel for the team that owns {svc}?",
                   "a": chan})
        # identify service from its unique description
        qa.append({"q": f"Which {COMPANY} service is described as: {desc}? "
                        f"Reply with the service name.", "a": svc})
        if deps:
            qa.append({"q": f"How many direct upstream dependencies does {svc} have? "
                            f"Reply with just the number.", "a": str(len(deps))})
    for svc in SERVICES:
        dents = dependents_of(svc)
        if dents:
            # any correct full answer contains the first dependent as a substring
            qa.append({"q": f"Name the services that directly depend on {svc}.",
                       "a": dents[0]})
    for code, (svc, meaning, _) in ERRORS.items():
        qa.append({"q": f"Which service raises error {code}?", "a": svc})
        qa.append({"q": f"What does error {code} mean at {COMPANY}?", "a": meaning})
    # policy contents (gold = short exact substring of the policy text)
    qa += [
        {"q": f"At what time (UTC) does the {COMPANY} Friday production deploy freeze begin?",
         "a": "18:00"},
        {"q": f"Until what day and time (UTC) does the {COMPANY} weekend deploy freeze last?",
         "a": "Monday 06:00"},
        {"q": "Within how many minutes must the service team lead be paged for a Sev1 "
              "incident? Reply with just the number.", "a": "10"},
        {"q": "Which Slack channel must a Sev1 incident be posted to?",
         "a": "#incident-bridge"},
        {"q": f"Per {COMPANY} rollback policy, an error-rate increase above what percentage "
              f"within 15 minutes forces a rollback?", "a": "2%"},
        {"q": f"What command does the {COMPANY} rollback policy say to use to roll back "
              f"a bad deploy?", "a": "nbx deploy rollback"},
        {"q": f"Through which service may secrets be read at {COMPANY}?", "a": "vaultkeep"},
        {"q": f"Per {COMPANY} capacity policy, how many minimum prod replicas must a "
              f"payments service run? Reply with just the number.", "a": "3"},
        {"q": f"Per {COMPANY} capacity policy, how many minimum prod replicas must a "
              f"non-payments service run? Reply with just the number.", "a": "2"},
    ]
    # reverse lookups: port -> service (ports are unique)
    for svc in ["gatekeeper", "quillbase", "ledgerline"]:
        qa.append({"q": f"Which {COMPANY} service listens on port {SERVICES[svc][1]}?",
                   "a": svc})
    R.shuffle(qa)
    return qa[:100]

# -------------------------------------------------------------------- IF ~120
def if_eval():
    items = []
    # 1. exact-phrase echo (20)
    phrases = ["cache warm complete", "rollout paused", "green across the board",
               "handover accepted", "maintenance window open", "queue drained",
               "backfill finished", "primary promoted", "canary looks clean",
               "traffic shifted", "lease renewed", "snapshot verified",
               "alarms silenced", "budget approved", "index rebuilt",
               "quota raised", "mirror synced", "probe passing",
               "ticket closed", "runway clear"]
    for p in phrases:
        items.append({"q": f"Repeat this back exactly, nothing else: {p}", "check": p})
    # 2. single-word echo as ALL CAPS (15)
    words = ["stable", "granite", "harbor", "velvet", "copper", "meadow", "lantern",
             "ember", "cobalt", "willow", "falcon", "summit", "orchid", "timber",
             "quartz"]
    for w in words:
        items.append({"q": f"Write the word '{w}' in all uppercase, nothing else.",
                      "check": w.upper()})
    # 3. arithmetic, single number (20) -- result string never appears in the question
    pairs = [(12, 9), (17, 26), (34, 58), (45, 27), (66, 29), (13, 48), (52, 39),
             (74, 18), (23, 44), (61, 17), (7, 8), (6, 9), (12, 12), (13, 7),
             (14, 6), (9, 9), (16, 4), (23, 3), (18, 5), (21, 4)]
    for i, (a, b) in enumerate(pairs):
        if i < 10:
            q, res = f"What is {a}+{b}? Reply with just the number.", a + b
        else:
            q, res = f"What is {a}*{b}? Reply with just the number.", a * b
        assert str(res) not in q
        items.append({"q": q, "check": str(res)})
    # 4. YES-only comparisons (15) -- 'NO' is unusable (substring of 'unknown', 'not')
    yes_facts = ["is 7 greater than 3", "is 40 an even number", "is 100 greater than 99",
                 "is 5 less than 12", "is 9 an odd number", "is 2 a prime number",
                 "does 'banana' contain the letter b", "is 60 divisible by 12",
                 "is a kilometer longer than a meter", "is 0 less than 1",
                 "is 8 an even number", "is 15 divisible by 5",
                 "is 'cat' shorter than 'catalog'", "is 30 greater than 13",
                 "is 4 squared equal to 16"]
    for f in yes_facts:
        items.append({"q": f"Answer only YES or NO: {f}?", "check": "YES"})
    # 5. end-with-token (15)
    fillers = ["any color", "any weekday", "any fruit", "any animal", "any month",
               "any planet", "any metal", "any sport", "any vegetable", "any bird",
               "any language", "any country", "any shape", "any tree", "any number"]
    for i, f in enumerate(fillers):
        tok = f"[DONE-{i + 1}]"
        items.append({"q": f"Name {f} in one word, then end your reply with the exact "
                           f"token {tok}", "check": tok})
    # 6. exactly-N comma-separated lists (15)
    lists = [("three", "primary colors"), ("three", "fruits"), ("four", "seasons"),
             ("three", "planets"), ("four", "weekdays"), ("three", "mammals"),
             ("three", "vegetables"), ("four", "ocean names"), ("three", "metals"),
             ("three", "musical instruments"), ("four", "European capitals"),
             ("three", "programming languages"), ("three", "board games"),
             ("four", "chess pieces"), ("three", "cloud types")]
    for n, thing in lists:
        items.append({"q": f"List exactly {n} {thing}, comma-separated, nothing else.",
                      "check": ","})
    # 7. letter counts, single digit (10) -- digit never appears in the question
    count_words = ["cat", "lamp", "stone", "bridge", "harvest", "keyboard",
                   "dog", "fern", "cloud", "planet"]
    for w in count_words:
        assert str(len(w)) not in w
        items.append({"q": f"Answer with a single digit: how many letters are in the "
                           f"word '{w}'?", "check": str(len(w))})
    # 8. concise free answers, check '' (10)
    concise = ["Give a one-word synonym for 'fast'.",
               "Give a one-word synonym for 'happy'.",
               "Give a one-word antonym of 'hot'.",
               "Answer in exactly one sentence: what does a load balancer do?",
               "Answer in exactly one sentence: what is a cache?",
               "Name the capital of France in one word.",
               "Give a one-word synonym for 'large'.",
               "Answer in exactly one sentence: what is version control?",
               "Name one primary color, one word only.",
               "Give a one-word antonym of 'empty'."]
    items += [{"q": q, "check": ""} for q in concise]
    R.shuffle(items)
    return items[:120]

# ---------------------------------------------------------- hallucination ~120
FAKE_SERVICES = [
    "cloudhopper", "datamule", "authzilla", "pipewrench", "shadowcat", "greenlight",
    "bytebarn", "signalpost", "lockstep", "murmur", "fogbank", "driftwood",
    "papertrailer", "hexforge", "coldwire", "brightpath", "stonemason", "quickmesh",
    "ironclad", "nightowl", "saltmine", "riverbed", "tinfoil", "moonshot",
    "cargohold", "flatline", "webspinner", "dustbin", "yardstick", "peppercorn",
    "glasshouse", "thundercap",
]
FAKE_ERRORS = ["NBX-6601", "NBX-6602", "NBX-7701", "NBX-7702", "NBX-8801",
               "NBX-8802", "NBX-9901", "NBX-9902", "NBX-1203", "NBX-4409"]
FAKE_POLICIES = ["data-retention", "vendor-access", "chaos-testing",
                 "gpu-quota", "overtime-paging", "green-deploy"]
FAKE_TEAMS = ["Mobile Experience", "Search & Discovery", "Growth Ops",
              "Trust & Safety"]

def halluc_eval():
    real, svcs = [], list(SERVICES.items())
    for svc, (team, port, slo, deps, desc) in svcs:
        real.append({"q": f"What does the {svc} service do at {COMPANY}?", "fake": False,
                     "a_substr": desc.split(";")[0].split(" and ")[0][:25]})
        real.append({"q": f"Which team owns the {svc} service at {COMPANY}?",
                     "fake": False, "a_substr": team})
        real.append({"q": f"What port does the {svc} service listen on at {COMPANY}?",
                     "fake": False, "a_substr": str(port)})
    for svc, (team, port, slo, deps, desc) in svcs[:5]:
        real.append({"q": f"What is the p99 latency SLO of {svc} at {COMPANY}?",
                     "fake": False, "a_substr": f"{slo}ms"})
    for code, (svc, meaning, _) in ERRORS.items():
        real.append({"q": f"What does error {code} mean at {COMPANY}?", "fake": False,
                     "a_substr": meaning})
        real.append({"q": f"Which service raises error {code} at {COMPANY}?",
                     "fake": False, "a_substr": svc})
    pol_sub = {"deploy-freeze": "Friday", "sev1": "10 minutes", "rollback": "2%",
               "secrets": "vaultkeep", "capacity": "2 replicas"}
    for name, sub in pol_sub.items():
        assert sub in POLICY[name]
        real.append({"q": f"What is the {COMPANY} {name} policy?", "fake": False,
                     "a_substr": sub})
    for team, info in TEAMS.items():
        real.append({"q": f"Who leads the {team} team at {COMPANY}?", "fake": False,
                     "a_substr": info["lead"]})
    real = real[:60]

    fake = []
    for f in FAKE_SERVICES[:20]:
        fake.append({"q": f"What does the {f} service do at {COMPANY}?", "fake": True,
                     "a_substr": None})
    for f in FAKE_SERVICES[20:26]:
        fake.append({"q": f"What port does the {f} service listen on at {COMPANY}?",
                     "fake": True, "a_substr": None})
    for f in FAKE_SERVICES[26:32]:
        fake.append({"q": f"Which team owns the {f} service at {COMPANY}?",
                     "fake": True, "a_substr": None})
    for c in FAKE_ERRORS[:5]:
        fake.append({"q": f"What does error {c} mean at {COMPANY}?", "fake": True,
                     "a_substr": None})
    for c in FAKE_ERRORS[5:]:
        fake.append({"q": f"Which service raises error {c} at {COMPANY}?", "fake": True,
                     "a_substr": None})
    for p in FAKE_POLICIES:
        fake.append({"q": f"What is the {COMPANY} {p} policy?", "fake": True,
                     "a_substr": None})
    for t in FAKE_TEAMS:
        fake.append({"q": f"Who leads the {t} team at {COMPANY}?", "fake": True,
                     "a_substr": None})
    for f in FAKE_SERVICES[:8]:
        fake.append({"q": f"What is the p99 latency SLO of the {f} service at {COMPANY}?",
                     "fake": True, "a_substr": None})
    fake = fake[:60]
    assert len(fake) == 60 and len(real) == 60

    # sanity: no fake entity accidentally exists in the world tables
    world_names = set(SERVICES) | set(ERRORS) | set(POLICY) | set(TEAMS)
    assert not (set(FAKE_SERVICES) | set(FAKE_ERRORS) | set(FAKE_POLICIES)
                | set(FAKE_TEAMS)) & world_names
    out = real + fake
    R.shuffle(out)
    return out

# ------------------------------------------------------------------ tools ~60
def tool_eval():
    status_ph = ["Check the current health of {s}.",
                 "Is {s} up right now? Check its status.",
                 "Show me the health and error rate of {s}.",
                 "What is the current status of the {s} service?"]
    logs_ph = ["Show the last 50 log lines for {s}.",
               "Tail the recent logs of {s}.",
               "Pull up the latest log output from {s}.",
               "Fetch the most recent log lines from {s}."]
    rollback_ph = ["Roll {s} back to the previous release.",
                   "The latest {s} deploy is bad - roll it back.",
                   "Revert {s} to its prior release.",
                   "Roll back the most recent deploy of {s}."]
    scale_ph = ["Scale {s} to 5 replicas.",
                "Increase {s} to 6 replicas.",
                "Set the replica count of {s} to 4.",
                "Scale the {s} service up to 8 replicas."]
    restart_ph = ["Do a rolling restart of {s}.",
                  "Restart the {s} service.",
                  "Give {s} a rolling restart now.",
                  "Perform a rolling restart on {s}."]
    verbs = [("nbx_status", status_ph), ("nbx_logs", logs_ph),
             ("nbx_deploy_rollback", rollback_ph), ("nbx_scale", scale_ph),
             ("nbx_restart", restart_ph)]
    evals, svcs = [], list(SERVICES)
    for i, svc in enumerate(svcs):              # 5 verbs x 10 services = 50
        for j, (tool, phr) in enumerate(verbs):
            evals.append({"q": phr[(i + j) % len(phr)].format(s=svc),
                          "tool": tool, "arg_substr": svc})
    for i, svc in enumerate(svcs):              # +10 extra varied phrasings
        j = i % len(verbs)
        tool, phr = verbs[j]
        # pick a phrase index guaranteed to differ from the base pass for this (svc, verb)
        evals.append({"q": phr[(i + j + 1) % len(phr)].format(s=svc),
                      "tool": tool, "arg_substr": svc})
    # dedupe exact question duplicates
    seen, out = set(), []
    for e in evals:
        if e["q"] not in seen:
            seen.add(e["q"])
            out.append(e)
    assert len(out) >= 60
    R.shuffle(out)
    return out[:60]

def main():
    out = {
        "eval_domain_qa.json": domain_qa(),
        "eval_if.json": if_eval(),
        "eval_halluc.json": halluc_eval(),
        "eval_tools.json": tool_eval(),
    }
    for name, data in out.items():
        json.dump(data, open(f"{HERE}/{name}", "w"), indent=1)
        print(f"  {name}: {len(data)} items")

if __name__ == "__main__":
    main()
