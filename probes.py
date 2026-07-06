"""Probe datasets: reasoning (known answers) + agentic tool-calls."""

# --- Reasoning probe: math with deterministic integer answers ---
REASONING = [
    ("Natalia sold clips to 48 friends in April, then sold half as many in May. "
     "How many clips did she sell altogether?", 72),
    ("Weng earns $12 per hour for babysitting. Yesterday she babysat for 50 minutes. "
     "How much did she earn, in dollars?", 10),
    ("Betty wants a $100 wallet and currently has half of the money. Her parents give her $15, "
     "and her grandparents give twice as much as her parents. How many more dollars does she need?", 5),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. "
     "How many bolts in total does it take?", 3),
    ("James writes a 3-page letter to each of 2 different friends twice a week. "
     "How many pages does he write in a year (52 weeks)?", 624),
    ("What is the remainder when 7^100 is divided by 13?", 9),
    ("If x + y = 10 and x*y = 21, what is x^2 + y^2?", 58),
    ("How many positive divisors does 60 have?", 12),
    ("A train travels 60 km in 45 minutes. What is its average speed in km/h?", 80),
    ("Three consecutive integers sum to 72. What is the largest of them?", 25),
]

# --- Agentic tool-call probe ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

# (user prompt, expected tool name, key arg substring check(s))
TOOLCALLS = [
    ("What's the weather in Tokyo in celsius?", "get_weather", ["tokyo"]),
    ("Search for the latest news on RTX A5000 GPU pricing.", "search_web", ["a5000"]),
    ("What is 3847 multiplied by 219?", "calculator", ["3847", "219"]),
    ("Tell me the current temperature in Paris.", "get_weather", ["paris"]),
    ("Find recent research papers about GRPO reinforcement learning.", "search_web", ["grpo"]),
    ("Compute 15% of 2400.", "calculator", ["2400"]),
    ("Is it raining in London right now?", "get_weather", ["london"]),
    ("Look up who won the 2022 World Cup.", "search_web", ["world cup"]),
]
