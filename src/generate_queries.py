"""
BNS Evaluation Query Augmentation Script
=========================================
Expands 50 seed queries from data/evaluation_queries.json into 300 queries
using: synonym replacement, persona shifting, contextual variation, and
negative augmentation — while preserving correct_section mappings.

Output → data/evaluation_queries_augmented.json
"""

import json
import random
import re
import copy
import os

# ── Configuration ──────────────────────────────────────────────────────────
SEED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_queries.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_queries_augmented.json")
TARGET_TOTAL = 300
RANDOM_SEED = 42

# ── Synonym / Replacement Dictionaries ─────────────────────────────────────
LEGAL_SYNONYMS = {
    "murder": ["intentional killing", "homicide with intent", "deliberate slaying"],
    "theft": ["stealing", "pilferage", "larceny", "dishonest taking of property"],
    "robbery": ["forcible theft", "mugging", "violent taking of property"],
    "cheating": ["fraud", "deception for gain", "swindling"],
    "hurt": ["bodily harm", "physical injury", "causing pain"],
    "grievous hurt": ["serious bodily injury", "severe harm", "grave physical injury"],
    "kidnapping": ["abduction from guardianship", "unlawful taking away", "snatching a person"],
    "extortion": ["coercive demand for property", "threatening to obtain money", "blackmail for gain"],
    "rape": ["sexual assault", "forced sexual act", "non-consensual sexual intercourse"],
    "defamation": ["damaging reputation", "slander", "libel"],
    "mischief": ["property damage", "deliberate destruction", "vandalism"],
    "forgery": ["fabrication of documents", "making false documents", "counterfeiting documents"],
    "dacoity": ["gang robbery", "armed gang theft", "banditry"],
    "assault": ["physical attack", "use of criminal force", "violent confrontation"],
    "stalking": ["persistent unwanted following", "obsessive pursuit", "repeated harassment"],
    "voyeurism": ["secret observation", "unauthorized watching of private acts", "covert surveillance of intimate acts"],
    "criminal intimidation": ["threatening behaviour", "making criminal threats", "menacing conduct"],
    "criminal conspiracy": ["agreement to commit crime", "conspiring for illegal act", "unlawful planning"],
    "dowry death": ["death linked to dowry demands", "bride burning for dowry", "matrimonial death for dowry"],
    "abetment": ["instigating a crime", "aiding in offence", "facilitating an illegal act"],
    "trespass": ["unlawful entry", "unauthorized intrusion", "illegal encroachment"],
    "culpable homicide": ["causing death without intent to kill", "unintentional killing", "death by dangerous act"],
    "acid attack": ["throwing corrosive substance", "acid throwing", "chemical assault"],
    "sexual harassment": ["unwelcome sexual conduct", "inappropriate sexual advances", "sexual misconduct"],
    "wrongful confinement": ["illegal detention", "unlawful imprisonment", "holding captive"],
    "negligence": ["carelessness", "reckless disregard", "lack of due care"],
    "property": ["belongings", "assets", "possessions"],
    "person": ["individual", "human being", "citizen"],
    "offence": ["crime", "violation", "illegal act"],
    "punishment": ["penalty", "sentence", "legal consequence"],
    "dies": ["succumbs to injuries", "loses their life", "passes away due to injuries"],
    "kills": ["causes the death of", "fatally injures", "takes the life of"],
    "commits": ["perpetrates", "carries out", "engages in"],
    "weapon": ["instrument of harm", "lethal object", "arm"],
}

# ── Persona Name Banks ─────────────────────────────────────────────────────
INDIAN_NAMES = [
    "Ramesh", "Suresh", "Priya", "Anita", "Vikram", "Neha", "Rajesh", "Sunita",
    "Amit", "Kavita", "Manoj", "Pooja", "Deepak", "Meera", "Sanjay", "Rekha",
    "Ravi", "Geeta", "Arun", "Lata", "Rakesh", "Seema", "Vijay", "Nisha",
    "Gopal", "Durga", "Mohan", "Sarita", "Krishna", "Radha", "Arjun", "Sita"
]

# Single-letter placeholders used in BNS-style queries
SINGLE_LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")

# ── Context Variation Templates ────────────────────────────────────────────
LOCATION_CONTEXTS = [
    "in a busy marketplace in Mumbai",
    "near a railway station in Delhi",
    "in a residential colony in Bangalore",
    "at a crowded bus stop in Kolkata",
    "in a rural village in Uttar Pradesh",
    "outside a temple in Varanasi",
    "in a parking lot of a shopping mall in Hyderabad",
    "near a school in Chennai",
    "at a wedding ceremony in Jaipur",
    "in a factory premises in Pune",
]

TIME_CONTEXTS = [
    "late at night around 2 AM",
    "during the early morning hours",
    "in broad daylight",
    "during the evening rush hour",
    "on a Sunday afternoon",
    "during a festival celebration",
    "just after midnight",
    "around noon on a weekday",
]


# ── Augmentation Functions ─────────────────────────────────────────────────

def synonym_replace(query_text: str, n_replacements: int = 2) -> str:
    """Replace up to n legal terms with their synonyms."""
    result = query_text
    replaceable = []
    for term, synonyms in LEGAL_SYNONYMS.items():
        if term.lower() in result.lower():
            replaceable.append((term, synonyms))

    random.shuffle(replaceable)
    count = 0
    for term, synonyms in replaceable:
        if count >= n_replacements:
            break
        synonym = random.choice(synonyms)
        # Case-insensitive replacement (preserve first occurrence's case roughly)
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(synonym, result, count=1)
        count += 1

    return result


def persona_shift(query_text: str) -> str:
    """Replace single-letter names (A, B, C...) with Indian names."""
    result = query_text
    used_names = set()

    # Find single-letter identifiers used as names (surrounded by word boundaries)
    letters_found = re.findall(r'\b([A-Z])\b', result)
    # Filter to only those that appear as standalone characters (likely name placeholders)
    unique_letters = []
    for letter in letters_found:
        if letter not in unique_letters and letter not in {'I', 'A'} or (letter == 'A' and f" A " in f" {result} "):
            unique_letters.append(letter)

    available_names = [n for n in INDIAN_NAMES if n not in used_names]
    random.shuffle(available_names)

    mapping = {}
    for i, letter in enumerate(unique_letters[:len(available_names)]):
        name = available_names[i]
        mapping[letter] = name
        used_names.add(name)

    for letter, name in mapping.items():
        # Replace standalone letter (word-boundary match)
        result = re.sub(rf'\b{letter}\b', name, result)

    return result


def add_context(query_text: str) -> str:
    """Add location or time context to scenario queries."""
    location = random.choice(LOCATION_CONTEXTS)
    time_ctx = random.choice(TIME_CONTEXTS)

    # Try to insert context after the first sentence or first comma
    sentences = query_text.split(".")
    if len(sentences) >= 2:
        insertion_point = random.choice(["location", "time", "both"])
        if insertion_point == "location":
            sentences[0] = sentences[0].rstrip() + f" {location}"
        elif insertion_point == "time":
            sentences[0] = sentences[0].rstrip() + f" {time_ctx}"
        else:
            sentences[0] = sentences[0].rstrip() + f" {location} {time_ctx}"
        return ".".join(sentences)
    else:
        return f"{query_text.rstrip('?').rstrip('.')} {location}?"


def rephrase_question_stem(query_text: str) -> str:
    """Change the question phrasing at the start."""
    stems = [
        ("What is the", "Under BNS, what is the"),
        ("What offence", "Which legal violation"),
        ("What section", "Which provision of BNS"),
        ("What does the BNS say", "What provision under BNS deals with"),
        ("Under what provision", "Which BNS section governs the situation"),
        ("Under BNS", "According to the Bharatiya Nyaya Sanhita"),
        ("What BNS provision", "Which section of BNS"),
        ("Which section", "What provision"),
        ("Which provision", "Which BNS section"),
        ("What offence has", "What crime has"),
        ("Is A guilty", "Can A be held liable"),
        ("Is B guilty", "Can B be held liable"),
        ("What offence does", "What crime does"),
        ("What sections apply", "Under which provisions can they be charged"),
        ("Under what sections can", "Which BNS provisions apply when"),
    ]
    result = query_text
    random.shuffle(stems)
    for original, replacement in stems:
        if original.lower() in result.lower():
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            result = pattern.sub(replacement, result, count=1)
            break
    return result


def generate_augmented_queries(seed_queries: list, target: int) -> list:
    """
    Generate augmented queries from seed queries to reach the target count.
    """
    random.seed(RANDOM_SEED)
    augmented = []

    # Include all original seeds
    for q in seed_queries:
        augmented.append(copy.deepcopy(q))

    remaining = target - len(seed_queries)
    queries_per_seed = remaining // len(seed_queries)
    extra = remaining % len(seed_queries)

    seed_augment_counts = [queries_per_seed] * len(seed_queries)
    # Distribute extra queries across seeds
    for i in range(extra):
        seed_augment_counts[i] += 1

    aug_id = len(seed_queries) + 1

    for idx, seed in enumerate(seed_queries):
        count_needed = seed_augment_counts[idx]
        category = seed.get("category", "direct")

        # Decide which augmentation strategies to apply based on category
        strategies = []

        if category == "direct":
            strategies = ["synonym", "rephrase", "synonym+rephrase"]
        elif category == "paraphrased":
            strategies = ["synonym", "rephrase", "persona"]
        elif category == "scenario":
            strategies = ["persona", "context", "synonym+context", "persona+context", "rephrase"]
        elif category == "multi-section":
            strategies = ["persona", "context", "synonym", "persona+context", "rephrase"]
        elif category == "confusing":
            strategies = ["rephrase", "persona", "synonym", "context"]

        for j in range(count_needed):
            new_query = copy.deepcopy(seed)
            new_query["query_id"] = f"Q{aug_id:03d}"
            new_query["source_seed"] = seed["query_id"]

            strategy = strategies[j % len(strategies)]

            text = seed["query"]

            if "synonym" in strategy and "context" in strategy:
                text = synonym_replace(text, n_replacements=2)
                text = add_context(text)
            elif "persona" in strategy and "context" in strategy:
                text = persona_shift(text)
                text = add_context(text)
            elif "synonym" in strategy and "rephrase" in strategy:
                text = synonym_replace(text, n_replacements=1)
                text = rephrase_question_stem(text)
            elif strategy == "synonym":
                text = synonym_replace(text, n_replacements=random.randint(1, 3))
            elif strategy == "persona":
                text = persona_shift(text)
            elif strategy == "context":
                text = add_context(text)
            elif strategy == "rephrase":
                text = rephrase_question_stem(text)

            new_query["query"] = text
            augmented.append(new_query)
            aug_id += 1

    return augmented


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # Resolve paths relative to this script
    seed_path = os.path.abspath(SEED_PATH)
    output_path = os.path.abspath(OUTPUT_PATH)

    print(f"Loading seed queries from: {seed_path}")
    with open(seed_path, "r", encoding="utf-8") as f:
        seed_queries = json.load(f)

    print(f"Loaded {len(seed_queries)} seed queries")
    print(f"Target total: {TARGET_TOTAL}")

    augmented = generate_augmented_queries(seed_queries, TARGET_TOTAL)

    print(f"Generated {len(augmented)} total queries")

    # Category distribution
    cats = {}
    for q in augmented:
        cat = q.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1
    print(f"Category distribution: {json.dumps(cats, indent=2)}")

    # Validate section IDs against BNS chunks
    bns_path = os.path.join(os.path.dirname(seed_path), "bns_chunks.json")
    if os.path.exists(bns_path):
        with open(bns_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        valid_sections = set(c["section_id"] for c in chunks)

        errors = []
        for q in augmented:
            secs = q["correct_section"] if isinstance(q["correct_section"], list) else [q["correct_section"]]
            for s in secs:
                if s not in valid_sections:
                    errors.append((q["query_id"], s))

        if errors:
            print(f"\n⚠ Section ID errors found: {errors}")
        else:
            print("✓ All section IDs validated against BNS")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=4, ensure_ascii=False)

    print(f"\n✓ Output written to: {output_path}")
    print(f"  Total queries: {len(augmented)}")


if __name__ == "__main__":
    main()
