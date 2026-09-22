"""
generate_synthetic_msmarco.py -- Week 1: synthetic corpus that mimics the
STRUCTURE of MS MARCO Passage Ranking closely enough to build and unit-test
a real retrieval pipeline in an environment that cannot reach Hugging Face.

This is NOT a substitute for real MS MARCO numbers. It exists only so that
BM25Index, the MRR@10 evaluator, and the eventual dense/cross-encoder code
can be built and tested against something with genuine (but modest)
vocabulary-mismatch difficulty, the same way PlayerLens's synthetic data was
built to match BG/NBD's own generative assumptions rather than being an
arbitrary toy.

MS MARCO's real schema (what we mimic):
    passages : {pid: str -> passage_text: str}
    queries  : {qid: str -> query_text: str}
    qrels    : {qid: str -> set of relevant pid(s)}   (mostly 1 relevant pid/query)

How the synthetic mismatch is created:
    Each topic has a set of CORE_TERMS and, for a fraction of them, a
    SYNONYM mapping (e.g. "heart attack" -> "myocardial infarction").
    Passages are generated using a mix of core-term and synonym phrasing.
    For every passage we generate ONE query that paraphrases it using the
    OPPOSITE phrasing choice for a sample of terms -- i.e. if the passage
    used the plain term, the query is built to sometimes use the synonym,
    and vice versa. This creates genuine cases where a purely lexical
    matcher (BM25) cannot find the exact words in the query, which is
    exactly the failure mode described in the SearchLens spec ("heart
    attack" vs "myocardial infarction").

    Distractor passages (same topic, different specific fact) and
    cross-topic passages that share a few generic words are also added, so
    BM25 has real negatives to rank against, not just unrelated topics.

Determinism: everything is seeded (default 42) so results are reproducible.
"""
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"

# Each topic: (topic_name, core_terms, synonym_map, fact_templates)
# fact_templates are sentence templates with {term} slots filled from
# core_terms or their synonyms. Each template instantiated with different
# fillers becomes a distinct passage "fact" within the topic.
TOPICS = [
    {
        "name": "cardiology",
        "terms": ["heart attack", "chest pain", "blocked artery", "high blood pressure", "cholesterol"],
        "synonyms": {
            "heart attack": "myocardial infarction",
            "blocked artery": "coronary artery occlusion",
            "high blood pressure": "hypertension",
        },
        "templates": [
            "A {term} occurs when blood flow to part of the heart muscle is reduced or blocked, causing tissue damage.",
            "Common warning signs of {term} include shortness of breath, sweating, and pain radiating to the arm.",
            "{term} is a major risk factor for cardiovascular disease and is often managed with medication and diet.",
            "Doctors diagnose {term} using blood tests, an ECG, and imaging of the coronary arteries.",
            "Lifestyle changes such as exercise and reduced salt intake can help control {term} over time.",
        ],
    },
    {
        "name": "astronomy",
        "terms": ["black hole", "event horizon", "neutron star", "supernova", "dark matter"],
        "synonyms": {
            "black hole": "gravitational singularity",
            "event horizon": "point of no return",
            "supernova": "stellar explosion",
        },
        "templates": [
            "A {term} forms when a massive star collapses under its own gravity at the end of its life.",
            "Scientists detect a {term} indirectly by observing its gravitational effect on nearby matter.",
            "The {term} plays a central role in modern theories of how galaxies formed and evolved.",
            "Telescopes such as the Event Horizon Telescope have captured direct images related to a {term}.",
            "Researchers use simulations to model how a {term} interacts with surrounding stars and gas.",
        ],
    },
    {
        "name": "cooking",
        "terms": ["sourdough starter", "caramelization", "emulsification", "blanching", "deglazing"],
        "synonyms": {
            "sourdough starter": "wild yeast culture",
            "caramelization": "sugar browning",
            "deglazing": "pan sauce lifting",
        },
        "templates": [
            "{term} is a technique that relies on precise temperature control to develop flavor.",
            "Professional chefs recommend practicing {term} repeatedly before attempting it in a timed service.",
            "The chemistry behind {term} involves breaking down sugars or proteins to create new compounds.",
            "A common mistake home cooks make with {term} is rushing the process and undercooking the result.",
            "{term} is featured heavily in French culinary training as a foundational skill.",
        ],
    },
    {
        "name": "software_engineering",
        "terms": ["race condition", "memory leak", "load balancer", "database index", "cache invalidation"],
        "synonyms": {
            "race condition": "concurrency bug",
            "memory leak": "unreleased memory allocation",
            "cache invalidation": "stale cache eviction",
        },
        "templates": [
            "A {term} can cause a production system to behave unpredictably under high traffic.",
            "Engineers use monitoring tools and stress tests to detect a {term} before it reaches customers.",
            "Fixing a {term} often requires careful code review and understanding of the underlying runtime.",
            "A {term} is one of the most commonly cited causes of subtle, hard-to-reproduce bugs.",
            "Distributed systems are especially prone to a {term} because of network delays and partial failures.",
        ],
    },
    {
        "name": "climate_science",
        "terms": ["greenhouse effect", "carbon sequestration", "ocean acidification", "permafrost thaw", "albedo effect"],
        "synonyms": {
            "greenhouse effect": "atmospheric heat trapping",
            "carbon sequestration": "carbon capture and storage",
            "ocean acidification": "declining ocean pH",
        },
        "templates": [
            "{term} is one of the key physical processes driving long-term changes in the Earth's climate.",
            "Climate models incorporate {term} to project how global temperatures will change over decades.",
            "Policymakers debate how much investment should go toward mitigating {term} versus adapting to it.",
            "Recent satellite data has improved scientists' ability to measure {term} at a global scale.",
            "{term} interacts with other feedback loops in ways that are still not fully understood.",
        ],
    },
    {
        "name": "finance",
        "terms": ["compound interest", "diversified portfolio", "inflation hedge", "yield curve", "liquidity risk"],
        "synonyms": {
            "compound interest": "interest on interest",
            "diversified portfolio": "spread-out investment mix",
            "yield curve": "term structure of interest rates",
        },
        "templates": [
            "Understanding {term} is essential for anyone planning long-term retirement savings.",
            "Financial advisors often use {term} to explain how small early decisions compound over decades.",
            "A sudden shift in {term} can signal changing expectations about future economic growth.",
            "Investors monitor {term} closely during periods of high market volatility.",
            "{term} is a core concept taught in introductory finance and economics courses.",
        ],
    },
]


def _fill(template: str, term: str) -> str:
    return template.format(term=term)


def generate_corpus(seed: int = 42, passages_per_topic: int = 40, queries_per_topic: int = 15):
    rng = random.Random(seed)

    passages = {}  # pid -> text
    queries = {}   # qid -> text
    qrels = {}     # qid -> set(pid)

    pid_counter = 0
    qid_counter = 0

    # Track, per topic, the (template_idx, term) -> pid used, so we can
    # generate paraphrased queries against a specific passage later.
    topic_passage_index = {t["name"]: [] for t in TOPICS}

    for topic in TOPICS:
        terms = topic["terms"]
        synonyms = topic["synonyms"]
        templates = topic["templates"]

        for _ in range(passages_per_topic):
            template = rng.choice(templates)
            term = rng.choice(terms)
            # Randomly choose plain term or synonym phrasing for the passage
            use_synonym = term in synonyms and rng.random() < 0.5
            surface_term = synonyms[term] if use_synonym else term
            text = _fill(template, surface_term)

            pid = f"p{pid_counter}"
            passages[pid] = text
            topic_passage_index[topic["name"]].append((pid, term, use_synonym))
            pid_counter += 1

        # Build queries against a sample of this topic's passages
        sampled = rng.sample(topic_passage_index[topic["name"]], k=min(queries_per_topic, len(topic_passage_index[topic["name"]])))
        for pid, term, passage_used_synonym in sampled:
            # Query paraphrase: flip the phrasing choice relative to the passage
            # whenever a synonym exists, to create real vocabulary mismatch.
            if term in synonyms:
                query_uses_synonym = not passage_used_synonym
                query_term = synonyms[term] if query_uses_synonym else term
            else:
                query_term = term

            query_openers = [
                f"what is {query_term}",
                f"explain {query_term}",
                f"how does {query_term} work",
                f"why does {query_term} matter",
                f"causes of {query_term}",
            ]
            query_text = rng.choice(query_openers)

            qid = f"q{qid_counter}"
            queries[qid] = query_text
            qrels[qid] = {pid}
            qid_counter += 1

    return passages, queries, qrels


def save_corpus(passages, queries, qrels, out_dir: Path = RAW_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "passages.jsonl", "w") as f:
        for pid, text in passages.items():
            f.write(json.dumps({"pid": pid, "text": text}) + "\n")

    with open(out_dir / "queries.jsonl", "w") as f:
        for qid, text in queries.items():
            f.write(json.dumps({"qid": qid, "text": text}) + "\n")

    with open(out_dir / "qrels.json", "w") as f:
        json.dump({qid: sorted(pids) for qid, pids in qrels.items()}, f, indent=2)


if __name__ == "__main__":
    passages, queries, qrels = generate_corpus()
    save_corpus(passages, queries, qrels)
    print(f"Generated {len(passages)} passages, {len(queries)} queries, {len(qrels)} qrels entries.")
    print(f"Wrote to {RAW_DIR}")