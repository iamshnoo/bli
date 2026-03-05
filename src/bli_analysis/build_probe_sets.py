#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordfreq import top_n_list


ASCII_WORD_RE = re.compile(r"^[a-z]+$")
ASCII_PHRASE_RE = re.compile(r"^[a-z ]+$")


def _uniq_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_term(term: str) -> str:
    text = str(term).strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid_term(term: str) -> bool:
    if not term:
        return False
    if len(term) < 3 or len(term) > 32:
        return False
    if not ASCII_PHRASE_RE.fullmatch(term):
        return False
    return True


CATEGORY_SPECS: list[dict[str, object]] = [
    {
        "key": "values_norms",
        "display_name": "Values and social norms",
        "framework_basis": "Hofstede; Schwartz; Inglehart-Welzel; GLOBE",
        "citations": ["hofstede2001culture", "schwartz2006theory", "inglehart2005modernization", "house2004culture"],
        "keywords": (
            "individualism", "collectivism", "hierarchy", "egalitarian", "authority", "autonomy",
            "obedience", "duty", "freedom", "honor", "shame", "virtue", "vice", "norm", "custom",
            "etiquette", "respect", "modesty", "purity", "pollution", "trust", "solidarity",
            "competition", "cooperation", "merit", "fairness", "justice", "taboo", "face",
            "reputation", "hospitality", "charity", "conformity", "dissent", "restraint", "indulgence",
            "secular", "sacred", "tradition", "modernity",
        ),
        "seed_terms": (
            "individualism", "collectivism", "hierarchy", "equality", "authority", "autonomy", "obedience",
            "duty", "freedom", "honor", "shame", "virtue", "vice", "respect", "modesty", "purity",
            "pollution", "face", "reputation", "hospitality", "charity", "cooperation", "competition",
            "fairness", "justice", "taboo", "tradition", "modernity", "sacred", "secular", "conformity",
            "dissent", "restraint", "indulgence", "solidarity", "merit", "trust", "loyalty", "integrity",
            "discretion", "restraint", "humility", "discipline", "self control", "deference",
        ),
        "modifiers": ("civic", "moral", "social", "public", "private", "traditional", "modern", "collective", "institutional", "normative"),
    },
    {
        "key": "family_kinship",
        "display_name": "Family and kinship",
        "framework_basis": "HRAF kinship domains; Schwartz Embeddedness; WVS family values",
        "citations": ["schwartz2006theory", "inglehart2005modernization", "hershcovich2022challenges"],
        "keywords": (
            "family", "kin", "mother", "father", "parent", "child", "ancestor", "lineage", "clan",
            "household", "marriage", "wedding", "bride", "groom", "dowry", "inheritance", "matri",
            "patri", "in law", "surname", "heir", "caretaker", "grand", "nephew", "niece", "cousin",
            "widow", "widower", "firstborn", "engagement", "courtship",
        ),
        "seed_terms": (
            "family", "kinship", "mother", "father", "parent", "child", "daughter", "son", "uncle",
            "aunt", "cousin", "grandmother", "grandfather", "nephew", "niece", "ancestor", "lineage",
            "clan", "household", "joint family", "nuclear family", "marriage", "wedding", "bride",
            "groom", "engagement", "dowry", "bridewealth", "inheritance", "widow", "widower",
            "courtship", "matchmaker", "surname", "patronymic", "matronymic", "firstborn", "heir",
            "filial piety", "caregiving", "motherhood", "fatherhood", "in laws", "brother in law",
            "sister in law", "ancestral home", "household labor",
        ),
        "modifiers": ("extended", "nuclear", "intergenerational", "domestic", "marital", "ancestral", "household", "kinship", "familial", "lineage"),
    },
    {
        "key": "religion_ritual",
        "display_name": "Religion and ritual life",
        "framework_basis": "Inglehart Traditional-Secular; WVS religiosity domains",
        "citations": ["inglehart2005modernization", "hofstede2001culture", "arora2023probing"],
        "keywords": (
            "religion", "ritual", "prayer", "temple", "church", "mosque", "synagogue", "shrine",
            "pilgrimage", "scripture", "monk", "priest", "imam", "nun", "saint", "martyr", "prophet",
            "zakat", "tithe", "karma", "altar", "incense", "rosary", "blessing", "curse", "sabbath",
            "halal", "kosher", "fast", "fasting", "sacrifice", "cemetery", "burial", "cremation",
        ),
        "seed_terms": (
            "religion", "ritual", "prayer", "temple", "church", "mosque", "synagogue", "shrine",
            "pilgrimage", "scripture", "monk", "priest", "imam", "nun", "saint", "martyr", "prophet",
            "zakat", "tithe", "karma", "altar", "incense", "rosary", "blessing", "curse", "sabbath",
            "halal", "kosher", "fasting", "sacrifice", "monastery", "convent", "abbey", "minaret",
            "pagoda", "stupa", "ancestral altar", "funeral rite", "mourning rite", "memorial service",
            "ancestor worship", "tabernacle", "prayer mat", "chanting", "hymn", "sermon", "liturgy",
        ),
        "modifiers": ("sacred", "ritual", "religious", "ceremonial", "devotional", "communal", "seasonal", "pilgrimage", "liturgical", "doctrinal"),
    },
    {
        "key": "food_cuisine",
        "display_name": "Food and cuisine",
        "framework_basis": "Food anthropology; cultural-value probing in NLP",
        "citations": ["hershcovich2022challenges", "arora2023probing", "berlin1969basic"],
        "keywords": (
            "rice", "bread", "noodle", "dumpling", "curry", "spice", "tea", "coffee", "meat",
            "vegetarian", "vegan", "soup", "stew", "barbecue", "fermented", "pickled", "banquet",
            "feast", "street food", "dessert", "sauce", "grain", "fruit", "vegetable", "kebab",
            "shawarma", "ramen", "sushi", "kimchi", "pho", "biryani", "falafel", "pasta", "cheese",
            "wine", "beer", "chai", "espresso",
        ),
        "seed_terms": (
            "rice", "bread", "noodle", "dumpling", "curry", "spice", "tea", "coffee", "meat",
            "vegetarian", "vegan", "soup", "stew", "barbecue", "fermented", "pickled", "banquet",
            "feast", "street food", "dessert", "sauce", "grain", "fruit", "vegetable", "kebab",
            "shawarma", "ramen", "sushi", "kimchi", "pho", "biryani", "falafel", "pasta", "cheese",
            "wine", "beer", "chai", "espresso", "udon", "soba", "congee", "bibimbap", "hummus",
            "tabbouleh", "paella", "risotto", "couscous", "injera", "tortilla", "tamale", "arepa",
            "empanada", "mochi", "matcha", "jalebi", "halwa", "flatbread", "naan", "roti", "chapati",
            "paratha", "soy sauce", "fish sauce", "ghee", "paneer", "pickles", "broth", "porridge",
            "lentil", "chickpea", "cassava", "plantain", "okra", "eggplant",
        ),
        "modifiers": ("traditional", "regional", "street", "home", "festive", "ceremonial", "spicy", "savory", "fermented", "seasonal"),
    },
    {
        "key": "festivals_holidays",
        "display_name": "Festivals and holidays",
        "framework_basis": "Ritual calendar domains in cross-cultural surveys and ethnography",
        "citations": ["inglehart2005modernization", "hershcovich2022challenges", "arora2023probing"],
        "keywords": (
            "festival", "holiday", "new year", "easter", "ramadan", "eid", "diwali", "vesak",
            "hanukkah", "passover", "nowruz", "carnival", "thanksgiving", "harvest", "obon",
            "tet", "songkran", "dragon boat", "mid autumn", "qingming", "yom kippur", "rosh",
            "purim", "sukkot", "advent",
        ),
        "seed_terms": (
            "festival", "holiday", "new year", "easter", "ramadan", "eid", "diwali", "vesak",
            "hanukkah", "passover", "nowruz", "carnival", "thanksgiving", "harvest festival", "obon",
            "tet", "songkran", "dragon boat festival", "mid autumn festival", "qingming", "yom kippur",
            "rosh hashanah", "purim", "sukkot", "advent", "holi", "navratri", "onam", "pongal",
            "durga puja", "janmashtami", "good friday", "pentecost", "epiphany", "all saints day",
            "lunar new year", "thingyan", "day of the dead", "mardi gras", "saint day", "martinmas",
            "st patricks day", "independence day", "mawlid", "ashura", "hajj", "umrah", "setsubun",
        ),
        "modifiers": ("religious", "seasonal", "national", "civic", "community", "spring", "harvest", "winter", "family", "public"),
    },
    {
        "key": "clothing_appearance",
        "display_name": "Clothing and appearance codes",
        "framework_basis": "Gender-role and modesty norms in GLOBE/Hofstede-aligned studies",
        "citations": ["house2004culture", "hofstede2001culture", "hershcovich2022challenges"],
        "keywords": (
            "dress", "clothing", "garment", "veil", "hijab", "niqab", "burqa", "sari", "kimono",
            "hanbok", "turban", "abaya", "thobe", "dupatta", "lehenga", "kurta", "dhoti", "qipao",
            "uniform", "formal", "casual", "modest", "ornate", "jewelry", "necklace", "earring",
            "bracelet", "ring", "headscarf", "tattoo", "piercing", "mask",
        ),
        "seed_terms": (
            "dress", "clothing", "garment", "veil", "hijab", "niqab", "burqa", "sari", "kimono",
            "hanbok", "turban", "abaya", "thobe", "dupatta", "lehenga", "kurta", "dhoti", "qipao",
            "uniform", "formal", "casual", "modest", "ornate", "jewelry", "necklace", "earring",
            "bracelet", "ring", "headscarf", "tattoo", "piercing", "mask", "gown", "tuxedo",
            "wedding dress", "sash", "sandals", "slippers", "bangle", "anklet", "kaftan", "poncho",
            "sombrero", "kippah", "yarmulke", "mourning dress", "ceremonial mask",
        ),
        "modifiers": ("formal", "casual", "traditional", "ceremonial", "modest", "ornate", "daily", "festival", "regional", "symbolic"),
    },
    {
        "key": "symbols_colors",
        "display_name": "Symbols, colors, and cultural objects",
        "framework_basis": "Color-term universals and symbol systems in cultural anthropology",
        "citations": ["berlin1969basic", "hershcovich2022challenges", "arora2023probing"],
        "keywords": (
            "white", "red", "black", "blue", "green", "yellow", "purple", "orange", "gold", "silver",
            "dragon", "phoenix", "lotus", "bamboo", "olive", "maple", "totem", "emblem", "anthem",
            "flag", "icon", "script", "calligraphy", "lantern", "incense", "opera", "dance", "music",
            "folklore", "myth", "legend",
        ),
        "seed_terms": (
            "white", "red", "black", "blue", "green", "yellow", "purple", "orange", "gold", "silver",
            "dragon", "phoenix", "lotus", "bamboo", "olive", "maple", "totem", "emblem", "anthem",
            "flag", "icon", "script", "calligraphy", "lantern", "incense", "opera", "dance", "music",
            "folklore", "myth", "legend", "oral tradition", "epic", "fable", "choir", "orchestra",
            "drumming", "chanting", "tango", "flamenco", "bharatanatyam", "kabuki", "lion dance",
            "dragon dance", "totem pole", "prayer beads", "rosary", "chopsticks", "tea ceremony",
            "martial arts",
        ),
        "modifiers": ("cultural", "national", "ceremonial", "historic", "traditional", "public", "sacred", "iconic", "folk", "shared"),
    },
    {
        "key": "governance_law",
        "display_name": "Governance, institutions, and law",
        "framework_basis": "Hofstede PDI/UAI and comparative political-institutional domains",
        "citations": ["hofstede2001culture", "house2004culture", "hershcovich2022challenges"],
        "keywords": (
            "democracy", "monarchy", "republic", "federal", "unitary", "parliament", "senate",
            "constitution", "bureaucracy", "meritocracy", "oligarchy", "aristocracy", "tribunal",
            "mediation", "civil law", "common law", "customary law", "council", "chieftain",
            "citizen", "state", "nation", "border", "governance", "policy", "regulation",
        ),
        "seed_terms": (
            "democracy", "monarchy", "republic", "federal", "unitary", "parliament", "senate",
            "constitution", "bureaucracy", "meritocracy", "oligarchy", "aristocracy", "tribunal",
            "mediation", "civil law", "common law", "customary law", "council", "chieftain",
            "citizen", "state", "nation", "border", "governance", "policy", "regulation",
            "empire", "colonial", "decolonization", "judiciary", "legislature", "executive",
            "ministry", "administration", "charter", "ordinance", "treaty", "diplomacy", "taxation",
            "public office", "village council", "city council", "mayor", "governor",
        ),
        "modifiers": ("public", "state", "civic", "local", "national", "legal", "administrative", "constitutional", "collective", "institutional"),
    },
    {
        "key": "social_identity",
        "display_name": "Social identity, migration, and belonging",
        "framework_basis": "WVS social trust and identity blocks; cross-cultural NLP identity dimensions",
        "citations": ["inglehart2005modernization", "hershcovich2022challenges", "arora2023probing"],
        "keywords": (
            "identity", "ethnicity", "diaspora", "migration", "immigrant", "refugee", "minority",
            "majority", "assimilation", "integration", "segregation", "pluralism", "multicultural",
            "indigenous", "native", "foreign", "local", "global", "community", "belonging", "language",
            "dialect", "heritage", "xenophobia", "stereotype", "prejudice",
        ),
        "seed_terms": (
            "identity", "ethnicity", "diaspora", "migration", "immigrant", "refugee", "minority",
            "majority", "assimilation", "integration", "segregation", "pluralism", "multiculturalism",
            "indigenous", "native", "foreign", "local", "global", "community", "belonging", "language",
            "dialect", "heritage", "xenophobia", "stereotype", "prejudice", "solidarity", "citizenship",
            "acculturation", "hybridity", "homeland", "exile", "settler", "cosmopolitan",
            "transnational", "identity politics", "mother tongue", "second language",
        ),
        "modifiers": ("local", "global", "cross border", "minority", "majority", "migrant", "diaspora", "community", "national", "multilingual"),
    },
    {
        "key": "daily_customs",
        "display_name": "Daily customs and life-cycle practices",
        "framework_basis": "Ethnographic daily-life domains; value-practice interfaces in cultural surveys",
        "citations": ["schwartz2006theory", "inglehart2005modernization", "hershcovich2022challenges"],
        "keywords": (
            "greeting", "handshake", "bow", "hug", "kiss", "gift", "hosting", "hospitality",
            "meal", "banquet", "tea", "funeral", "mourning", "burial", "cremation", "birth", "naming",
            "coming of age", "rite", "ceremony", "weekday", "weekend", "workday", "restday", "queue",
            "punctual", "lateness", "privacy", "public", "neighbor",
        ),
        "seed_terms": (
            "greeting", "handshake", "bow", "hug", "kiss", "gift giving", "hosting", "hospitality",
            "shared meal", "banquet", "tea time", "funeral", "mourning", "burial", "cremation", "birth",
            "naming ceremony", "coming of age", "rite of passage", "ceremony", "weekday", "weekend",
            "workday", "rest day", "queue", "punctuality", "lateness", "privacy", "public space",
            "neighbor", "household visit", "guest", "host", "table manners", "home cooking",
            "festival meal", "fast day", "memorial day", "ancestor day", "seasonal ritual",
        ),
        "modifiers": ("daily", "household", "community", "public", "private", "seasonal", "ceremonial", "social", "local", "shared"),
    },
]


AXIS_SPECS: list[dict[str, object]] = [
    {"left": "individualism", "right": "collectivism", "category": "values_norms", "citations": ["hofstede2001culture", "schwartz2006theory"]},
    {"left": "hierarchy", "right": "equality", "category": "values_norms", "citations": ["hofstede2001culture", "schwartz2006theory"]},
    {"left": "duty", "right": "freedom", "category": "values_norms", "citations": ["inglehart2005modernization"]},
    {"left": "honor", "right": "shame", "category": "values_norms", "citations": ["house2004culture"]},
    {"left": "obedience", "right": "autonomy", "category": "values_norms", "citations": ["schwartz2006theory"]},
    {"left": "tradition", "right": "modernity", "category": "values_norms", "citations": ["inglehart2005modernization"]},
    {"left": "restraint", "right": "indulgence", "category": "values_norms", "citations": ["hofstede2001culture"]},
    {"left": "conformity", "right": "dissent", "category": "values_norms", "citations": ["schwartz2006theory"]},
    {"left": "solidarity", "right": "competition", "category": "values_norms", "citations": ["house2004culture"]},
    {"left": "certainty", "right": "ambiguity", "category": "values_norms", "citations": ["hofstede2001culture"]},
    {"left": "mother", "right": "father", "category": "family_kinship", "citations": ["house2004culture"]},
    {"left": "elder", "right": "youth", "category": "family_kinship", "citations": ["hofstede2001culture"]},
    {"left": "kin", "right": "stranger", "category": "family_kinship", "citations": ["schwartz2006theory"]},
    {"left": "lineage", "right": "mobility", "category": "family_kinship", "citations": ["inglehart2005modernization"]},
    {"left": "clan", "right": "individual", "category": "family_kinship", "citations": ["schwartz2006theory"]},
    {"left": "sacred", "right": "secular", "category": "religion_ritual", "citations": ["inglehart2005modernization"]},
    {"left": "ritual", "right": "routine", "category": "religion_ritual", "citations": ["house2004culture"]},
    {"left": "prayer", "right": "reason", "category": "religion_ritual", "citations": ["inglehart2005modernization"]},
    {"left": "pilgrimage", "right": "tourism", "category": "religion_ritual", "citations": ["hershcovich2022challenges"]},
    {"left": "taboo", "right": "permissive", "category": "religion_ritual", "citations": ["hofstede2001culture"]},
    {"left": "democracy", "right": "monarchy", "category": "governance_law", "citations": ["hofstede2001culture"]},
    {"left": "federal", "right": "centralized", "category": "governance_law", "citations": ["house2004culture"]},
    {"left": "customary", "right": "codified", "category": "governance_law", "citations": ["hofstede2001culture"]},
    {"left": "authority", "right": "deliberation", "category": "governance_law", "citations": ["house2004culture"]},
    {"left": "patronage", "right": "meritocracy", "category": "governance_law", "citations": ["hofstede2001culture"]},
    {"left": "rice", "right": "bread", "category": "food_cuisine", "citations": ["hershcovich2022challenges"]},
    {"left": "tea", "right": "coffee", "category": "food_cuisine", "citations": ["hershcovich2022challenges"]},
    {"left": "spicy", "right": "mild", "category": "food_cuisine", "citations": ["hershcovich2022challenges"]},
    {"left": "vegetarian", "right": "meat", "category": "food_cuisine", "citations": ["arora2023probing"]},
    {"left": "fermented", "right": "fresh", "category": "food_cuisine", "citations": ["hershcovich2022challenges"]},
    {"left": "fasting", "right": "feasting", "category": "festivals_holidays", "citations": ["inglehart2005modernization"]},
    {"left": "lunar", "right": "solar", "category": "festivals_holidays", "citations": ["hershcovich2022challenges"]},
    {"left": "mourning", "right": "celebration", "category": "festivals_holidays", "citations": ["inglehart2005modernization"]},
    {"left": "ancestor", "right": "novelty", "category": "festivals_holidays", "citations": ["schwartz2006theory"]},
    {"left": "pilgrim", "right": "spectator", "category": "festivals_holidays", "citations": ["arora2023probing"]},
    {"left": "veiled", "right": "unveiled", "category": "clothing_appearance", "citations": ["house2004culture"]},
    {"left": "formal", "right": "casual", "category": "clothing_appearance", "citations": ["hofstede2001culture"]},
    {"left": "ornate", "right": "plain", "category": "clothing_appearance", "citations": ["hershcovich2022challenges"]},
    {"left": "modest", "right": "revealing", "category": "clothing_appearance", "citations": ["house2004culture"]},
    {"left": "uniform", "right": "personalized", "category": "clothing_appearance", "citations": ["hofstede2001culture"]},
    {"left": "white", "right": "red", "category": "symbols_colors", "citations": ["berlin1969basic"]},
    {"left": "black", "right": "gold", "category": "symbols_colors", "citations": ["berlin1969basic"]},
    {"left": "dragon", "right": "eagle", "category": "symbols_colors", "citations": ["hershcovich2022challenges"]},
    {"left": "lotus", "right": "rose", "category": "symbols_colors", "citations": ["hershcovich2022challenges"]},
    {"left": "script", "right": "image", "category": "symbols_colors", "citations": ["arora2023probing"]},
    {"left": "local", "right": "global", "category": "social_identity", "citations": ["inglehart2005modernization"]},
    {"left": "indigenous", "right": "diaspora", "category": "social_identity", "citations": ["hershcovich2022challenges"]},
    {"left": "assimilation", "right": "pluralism", "category": "social_identity", "citations": ["hershcovich2022challenges"]},
    {"left": "majority", "right": "minority", "category": "social_identity", "citations": ["arora2023probing"]},
    {"left": "homeland", "right": "migration", "category": "social_identity", "citations": ["inglehart2005modernization"]},
]


NEGATIVE_CONTROL_SEEDS: tuple[str, ...] = (
    "water", "stone", "sand", "earth", "sky", "sun", "moon", "star", "rain", "snow", "wind", "cloud",
    "river", "lake", "ocean", "mountain", "valley", "forest", "tree", "leaf", "seed", "root", "bark",
    "bird", "fish", "dog", "cat", "cow", "horse", "sheep", "goat", "bone", "blood", "heart", "lung",
    "brain", "eye", "ear", "nose", "mouth", "tooth", "tongue", "hand", "foot", "arm", "leg", "finger",
    "toe", "skin", "hair", "head", "neck", "back", "chest", "number", "count", "sum", "difference",
    "product", "ratio", "fraction", "decimal", "percent", "angle", "circle", "square", "triangle",
    "line", "point", "vector", "matrix", "tensor", "value", "unit", "meter", "liter", "gram", "second",
    "minute", "hour", "day", "week", "year", "mass", "length", "width", "height", "volume", "area",
    "speed", "force", "energy", "power", "motion", "density", "pressure", "temperature", "solid",
    "liquid", "gas", "atom", "molecule", "cell", "basic", "simple", "plain", "constant", "stable",
)


def _phrase_contains(term: str, keyword: str) -> bool:
    if " " in keyword:
        return bool(re.search(rf"\b{re.escape(keyword)}\b", term))
    return keyword in term.split()


def infer_cultural_category(term: str) -> str | None:
    t = normalize_term(term)
    if not is_valid_term(t):
        return None
    for spec in CATEGORY_SPECS:
        for kw in spec["keywords"]:  # type: ignore[index]
            k = normalize_term(str(kw))
            if _phrase_contains(t, k):
                return str(spec["key"])
    return None


def build_cultural_terms(target_count: int) -> tuple[list[str], dict[str, str], dict[str, int]]:
    categories = [str(s["key"]) for s in CATEGORY_SPECS]
    quota = target_count // len(categories)
    if quota * len(categories) != target_count:
        raise RuntimeError(
            f"Target {target_count} is not divisible by category count {len(categories)}; "
            "use a round number that supports balanced category quotas."
        )

    per_cat: dict[str, list[str]] = {}
    for spec in CATEGORY_SPECS:
        cat = str(spec["key"])
        seeds = _uniq_keep_order([normalize_term(x) for x in spec["seed_terms"]])  # type: ignore[index]
        seeds = [x for x in seeds if is_valid_term(x)]

        expanded = list(seeds)
        modifiers = [normalize_term(x) for x in spec.get("modifiers", [])]  # type: ignore[arg-type]
        for mod in modifiers:
            for base in seeds:
                candidate = normalize_term(f"{mod} {base}")
                if is_valid_term(candidate):
                    expanded.append(candidate)
                if len(_uniq_keep_order(expanded)) >= quota:
                    break
            if len(_uniq_keep_order(expanded)) >= quota:
                break

        expanded = _uniq_keep_order(expanded)
        if len(expanded) < quota:
            # Deterministic second-pass expansion with concept suffixes.
            suffixes = ("practice", "norm", "tradition", "value", "domain")
            for base in seeds:
                for suffix in suffixes:
                    candidate = normalize_term(f"{base} {suffix}")
                    if is_valid_term(candidate):
                        expanded.append(candidate)
                    if len(_uniq_keep_order(expanded)) >= quota:
                        break
                if len(_uniq_keep_order(expanded)) >= quota:
                    break
            expanded = _uniq_keep_order(expanded)

        if len(expanded) < quota:
            raise RuntimeError(
                f"Category '{cat}' has only {len(expanded)} terms; need at least {quota} for round-balance target."
            )
        per_cat[cat] = expanded

    selected: list[str] = []
    cat_map: dict[str, str] = {}

    for cat in categories:
        for term in per_cat[cat][:quota]:
            if term not in cat_map:
                cat_map[term] = cat
                selected.append(term)

    if len(selected) < target_count:
        for term in [normalize_term(x) for x in top_n_list("en", 120000)]:
            if len(selected) >= target_count:
                break
            if not is_valid_term(term):
                continue
            if term in cat_map:
                continue
            cat = infer_cultural_category(term)
            if cat is None:
                continue
            cat_map[term] = cat
            selected.append(term)

    if len(selected) < target_count:
        raise RuntimeError(
            f"Unable to reach cultural target {target_count}; only collected {len(selected)} terms after expansion."
        )

    selected = selected[:target_count]
    counts: dict[str, int] = {cat: 0 for cat in categories}
    for term in selected:
        counts[cat_map[term]] += 1
    return selected, {t: cat_map[t] for t in selected}, counts


def build_probe_set(
    anchor_target: int,
    cultural_target: int,
    negative_control_target: int,
    axis_target: int,
) -> dict:
    semantic_axis_specs = AXIS_SPECS[:axis_target]
    if len(semantic_axis_specs) < axis_target:
        raise RuntimeError(f"Requested {axis_target} axes but only {len(AXIS_SPECS)} are defined.")

    semantic_axes = [[str(x["left"]), str(x["right"])] for x in semantic_axis_specs]
    semantic_axes = [[normalize_term(a), normalize_term(b)] for a, b in semantic_axes]

    cultural_terms, cultural_probe_categories, category_counts = build_cultural_terms(cultural_target)

    axis_words = sorted({w for pair in semantic_axes for w in pair})
    reserved = set(cultural_terms) | set(axis_words)

    stop = set(ENGLISH_STOP_WORDS)
    stop.update(
        {
            "thing", "things", "something", "anything", "everything", "someone", "anyone",
            "today", "tomorrow", "yesterday", "yes", "no", "ok", "okay", "etc", "mr", "mrs", "ms",
        }
    )

    negative_control_words: list[str] = []
    for raw in NEGATIVE_CONTROL_SEEDS:
        w = normalize_term(raw)
        if not ASCII_WORD_RE.fullmatch(w):
            continue
        if w in reserved:
            continue
        negative_control_words.append(w)
    negative_control_words = _uniq_keep_order(negative_control_words)

    if len(negative_control_words) < negative_control_target:
        for w in [normalize_term(x) for x in top_n_list("en", 120000)]:
            if len(negative_control_words) >= negative_control_target:
                break
            if not ASCII_WORD_RE.fullmatch(w):
                continue
            if w in reserved or w in stop:
                continue
            if infer_cultural_category(w) is not None:
                continue
            if w in negative_control_words:
                continue
            negative_control_words.append(w)

    if len(negative_control_words) < negative_control_target:
        raise RuntimeError(
            f"Unable to build {negative_control_target} negative controls; got {len(negative_control_words)}."
        )
    negative_control_words = negative_control_words[:negative_control_target]

    reserved_with_controls = reserved | set(negative_control_words)
    common_words = [normalize_term(w) for w in top_n_list("en", 50000)]
    neutral_anchors: list[str] = []
    for w in common_words:
        if len(neutral_anchors) >= anchor_target:
            break
        if w in reserved_with_controls or w in stop:
            continue
        if not ASCII_WORD_RE.fullmatch(w):
            continue
        if len(w) < 3 or len(w) > 14:
            continue
        neutral_anchors.append(w)
    neutral_anchors = _uniq_keep_order(neutral_anchors)

    if len(neutral_anchors) < anchor_target:
        raise RuntimeError(f"Unable to build {anchor_target} neutral anchors; got {len(neutral_anchors)}.")

    all_probe_words = _uniq_keep_order(neutral_anchors + cultural_terms + negative_control_words + axis_words)

    category_frameworks = {
        str(spec["key"]): {
            "display_name": spec["display_name"],
            "framework_basis": spec["framework_basis"],
            "citations": spec["citations"],
            "selected_count": category_counts.get(str(spec["key"]), 0),
        }
        for spec in CATEGORY_SPECS
    }

    semantic_axis_metadata = [
        {
            "index": i + 1,
            "endpoint_1": normalize_term(str(spec["left"])),
            "endpoint_2": normalize_term(str(spec["right"])),
            "category": str(spec["category"]),
            "citations": spec["citations"],
        }
        for i, spec in enumerate(semantic_axis_specs)
    ]

    return {
        "metadata": {
            "anchor_target": anchor_target,
            "neutral_anchor_count": len(neutral_anchors),
            "cultural_probe_target": cultural_target,
            "cultural_probe_count": len(cultural_terms),
            "negative_control_target": negative_control_target,
            "negative_control_count": len(negative_control_words),
            "semantic_axis_target": axis_target,
            "semantic_axis_count": len(semantic_axes),
            "all_probe_word_count": len(all_probe_words),
            "category_count": len(category_frameworks),
        },
        "category_frameworks": category_frameworks,
        "neutral_anchor_words": neutral_anchors,
        "negative_control_words": negative_control_words,
        "cultural_probe_words": cultural_terms,
        "cultural_probe_categories": cultural_probe_categories,
        "semantic_axes": semantic_axes,
        "semantic_axis_metadata": semantic_axis_metadata,
        "all_probe_words": all_probe_words,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build grouped, literature-grounded probe sets for BLI analysis")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/scratch/amukher6/bli/data/probes/probe_sets.json"),
    )
    parser.add_argument("--anchor-target", type=int, default=3000)
    parser.add_argument("--cultural-target", type=int, default=1000)
    parser.add_argument("--negative-control-target", type=int, default=100)
    parser.add_argument("--axis-target", type=int, default=50)
    args = parser.parse_args()

    payload = build_probe_set(
        anchor_target=args.anchor_target,
        cultural_target=args.cultural_target,
        negative_control_target=args.negative_control_target,
        axis_target=args.axis_target,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Wrote probes: {args.output}")
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
