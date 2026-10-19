"""Turn a product title into comparable attributes and precise search queries.

The cross-store matcher (``app/matcher.py``) needs more than word overlap to
decide that two listings are the same product: brand, model code, variant
words ("Pro", "Ultra", "FE"), and hard specs (128 GB vs 256 GB, 5 kg vs 10 kg,
55 inch vs 43 inch) all matter. This module extracts those from free text.

Everything here is deterministic string processing; no network access.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORDS = re.compile(r"\d+\.\d+|[a-z0-9]+(?:[-'][a-z0-9]+)*")

STOPWORDS = {
    "a", "an", "the", "of", "for", "with", "and", "or", "in", "on", "to", "by", "at", "from", "new",
    "latest", "original", "genuine", "authentic", "brand", "buy", "online", "best", "price", "india",
    "free", "delivery", "offer", "sale", "deal", "combo", "set", "pack", "pcs", "pc", "only", "upto",
    "ram", "rom", "storage", "display", "camera", "battery", "processor", "warranty", "year", "years",
    "non", "not", "no", "is", "are", "as", "this", "that", "mobile", "phone", "smartphone",
}

COLORS = {
    "black", "white", "blue", "red", "green", "yellow", "pink", "purple", "grey", "gray", "silver", "gold",
    "orange", "brown", "beige", "navy", "maroon", "violet", "cyan", "teal", "lavender", "graphite", "titanium",
    "charcoal", "graygreen", "lilac", "mint", "cream", "multicolor", "multi", "rose", "midnight", "starlight",
    "awesome", "waterfall", "mystic", "lime", "sky", "ocean", "coral", "olive", "khaki", "turquoise",
}

GENDERS = {"men": "men", "mens": "men", "men's": "men", "women's": "women", "boys'": "boys", "girls'": "girls", "man": "men", "male": "men", "boys": "boys", "boy": "boys",
           "women": "women", "womens": "women", "woman": "women", "ladies": "women", "female": "women",
           "girls": "girls", "girl": "girls", "kids": "kids", "kid": "kids", "unisex": "unisex"}

VARIANT_WORDS = {
    "pro", "max", "ultra", "plus", "lite", "mini", "fe", "neo", "se", "air", "edge", "turbo", "gt",
    "fold", "flip", "xl", "xs",
}

ACCESSORY_WORDS = {
    "case", "cases", "cover", "covers", "skin", "skins", "protector", "protectors", "tempered", "guard",
    "guards", "pouch", "sleeve", "strap", "straps", "stand", "holder", "mount", "sticker", "stickers",
    "decal", "refill", "refills", "spare", "replacement", "adapter", "adaptor", "cable", "cables",
    "charger", "chargers", "lens", "stylus", "keycap", "keycaps", "bumper", "flip-cover", "screenguard",
    "sling", "lanyard", "wallpaper", "film", "tripod", "gimbal", "dock", "grip", "ring", "popsocket",
}

# words that say what kind of thing a listing is, not which product it is
GENERIC_NAME_WORDS = {
    "men", "mens", "men's", "women", "womens", "women's", "boys", "girls", "kids", "unisex", "running", "road", "sports",
    "sport", "casual", "formal", "smart", "wireless", "bluetooth", "digital", "adult", "liquid", "matte", "shoes",
    "shoe", "sneakers", "watch", "phone", "laptop", "tv", "led", "ac", "split", "inverter", "star", "model", "series",
    "edition", "color", "colour", "size", "regular", "fit", "slim", "cotton", "printed", "solid",
}

# coarse product types, used only when both sides expose one
PRODUCT_TYPES = {
    "phone": {"smartphone", "mobile", "phone", "smartphones", "mobiles"},
    "laptop": {"laptop", "notebook", "macbook", "chromebook"},
    "tv": {"tv", "television", "qled", "oled", "smarttv"},
    "audio": {"earbuds", "earphones", "earphone", "headphones", "headphone", "neckband", "tws", "headset", "earbud", "airpods", "speaker", "soundbar"},
    "watch": {"watch", "smartwatch", "smartwatches", "watches"},
    "footwear": {"shoes", "shoe", "sneakers", "sandals", "slippers", "flip", "boots", "loafers", "footwear"},
    "tshirt": {"tshirt", "t-shirt", "tee", "shirt", "polo"},
    "jeans": {"jeans", "trousers", "pants", "chinos"},
    "bag": {"backpack", "trolley", "luggage", "duffel", "rucksack", "handbag"},
    "foundation": {"foundation", "concealer", "compact"},
    "lipstick": {"lipstick", "lipgloss", "lip", "liner"},
    "skincare": {"serum", "moisturizer", "moisturiser", "sunscreen", "lotion", "cream", "facewash", "cleanser"},
    "haircare": {"shampoo", "conditioner", "hairoil"},
    "appliance": {"refrigerator", "fridge", "washing", "microwave", "oven", "cooker", "mixer", "grinder", "kettle", "purifier", "heater", "fan", "cooler", "geyser", "iron", "vacuum"},
    "ac": {"ac", "conditioner"},
    "camera": {"camera", "dslr", "mirrorless", "gopro"},
    "tablet": {"tablet", "ipad"},
}

CATEGORY_KEYWORDS = {
    "electronics": {"smartphone", "mobile", "phone", "laptop", "tv", "television", "earbuds", "earphones", "headphones",
                    "neckband", "tws", "smartwatch", "tablet", "camera", "monitor", "router", "ssd", "processor", "gb",
                    "refrigerator", "washing", "microwave", "speaker", "soundbar", "printer", "keyboard", "mouse", "charger",
                    "powerbank", "oven", "cooker", "mixer", "geyser", "purifier", "5g", "ac", "inverter", "split", "conditioner",
                    "fridge", "cooler", "heater", "dishwasher", "chimney", "vacuum", "trimmer", "shaver", "dryer"},
    "beauty": {"foundation", "lipstick", "serum", "moisturizer", "moisturiser", "sunscreen", "shampoo", "conditioner", "perfume",
               "kajal", "eyeliner", "mascara", "facewash", "lotion", "concealer", "compact", "makeup", "skincare", "cleanser",
               "cream", "nailpolish", "deodorant", "hairoil"},
    "fashion": {"shoes", "sneakers", "sneaker", "sandals", "sandal", "slides", "flip-flops", "boots", "sports-shoes", "tee", "tshirts", "tshirt", "t-shirt", "shirt", "jeans", "kurta", "saree", "dress", "jacket",
                "trousers", "hoodie", "sweatshirt", "leggings", "slippers", "loafers", "polo", "shorts", "track", "lehenga"},
    "bags": {"backpack", "trolley", "luggage", "duffel", "rucksack", "handbag", "wallet", "suitcase"},
    "grocery": {"chips", "biscuit", "chocolate", "tea", "coffee", "noodles", "atta", "rice", "dal", "oil", "ghee", "masala",
                "snack", "cereal", "juice", "sauce", "spice", "salt", "sugar", "milk", "curd", "pasta", "oats"},
}

# Product condition. A refurbished / pre-owned / open-box unit is a different offer from a new one even
# when the title is otherwise identical, so it must never verify against a new source (or the reverse).
_NOT = r"(?<!\bnot )(?<!\bnon )(?<!\bnon-)(?<!\bno )"          # "not refurbished", "non-refurbished"
_CONDITION_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("refurbished", re.compile(
        _NOT + r"\b(?:(?:certified|factory|grade\s*[abc])\s+)?(?:refurbished|refurb|reconditioned|re-?certified)\b")),
    # "renewed" is also skincare wording ("renewed formula"), so only the marketplace forms count
    ("refurbished", re.compile(
        _NOT + r"[(\[]\s*(?:amazon\s+)?renewed\s*[)\]]|\bamazon\s+renewed\b|\bcertified\s+renewed\b|^renewed\b|\brenewed\s*$|\s-\s*renewed\b")),
    ("open_box", re.compile(_NOT + r"\bopen[\s-]?box(?:ed)?\b(?!\s+delivery)")),     # "open box delivery" is a delivery option
    # "used for/with ..." is feature wording; "used" as a listing grade is not
    ("used", re.compile(
        _NOT + r"\b(?:pre[\s-]?owned|second[\s-]?hand|previously\s+owned|like[\s-]new)\b"
        + r"|" + _NOT + r"\bused\b(?!\s+(?:for|to|with|by|in|as|on|at|when|during|after|before|from|and|or)\b)")),
)


def detect_condition(text: str | None) -> str:
    """"new" unless the listing says refurbished / open_box / used (pre-owned, like new, second hand)."""
    lowered = normalise(text)
    for condition, pattern in _CONDITION_RULES:
        if pattern.search(lowered):
            return condition
    return "new"


def strip_condition_words(text: str | None) -> str:
    """The title without its condition wording, so "Refurbished"/"Renewed" never count as name words."""
    lowered = normalise(text)
    for _, pattern in _CONDITION_RULES:
        lowered = pattern.sub(" ", lowered)
    return re.sub(r"[(\[]\s*[,\-]*\s*[)\]]", " ", lowered)


_UNIT_NORMALISE = {
    "gb": "gb", "tb": "tb", "kg": "kg", "kgs": "kg", "g": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g",
    "ml": "ml", "l": "l", "ltr": "l", "ltrs": "l", "litre": "l", "litres": "l", "liter": "l", "liters": "l",
    "inch": "inch", "inches": "inch", "cm": "cm", "mm": "mm", "w": "w", "watt": "w", "watts": "w", "mah": "mah",
    "hz": "hz", "mp": "mp", "ton": "ton", "tons": "ton", "pcs": "pcs", "pc": "pcs", "pieces": "pcs", "piece": "pcs",
    "count": "pcs", "ct": "pcs", "star": "star", "stars": "star",
}
# units whose mismatch means "a different SKU" rather than marketing wording
STRONG_UNITS = {"ram_gb", "storage_gb", "g", "ml", "ton", "pcs", "star"}   # "inch" is added for TVs/laptops by the matcher

_SPEC = re.compile(
    r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*(gb|tb|kgs?|gms?|grams?|g|ml|ltrs?|litres?|liters?|l|inch(?:es)?|cm|mm|"
    r"watts?|w|mah|hz|mp|tons?|pcs|pc|pieces?|count|ct|stars?)(?![a-z])|(\d+(?:\.\d+)?)\s*[\"”″]",
    re.I,
)
_PACK = re.compile(r"(?:pack|set|combo)\s*of\s*(\d+)|(\d+)\s*(?:-|\s)?(?:pack|pcs|pieces|count)\b|\bx\s*(\d+)\b", re.I)


@dataclass
class ProductIdentity:
    title: str
    brand: str | None
    head_tokens: list[str]                      # leading, meaningful words of the title
    tokens: set[str]                            # all meaningful words
    model_codes: set[str] = field(default_factory=set)      # alphanumeric model codes, hyphens removed
    code_forms: dict[str, str] = field(default_factory=dict)  # normalised code -> spelling used in the title
    bare_numbers: set[str] = field(default_factory=set)     # "15" in "iphone 15"
    variants: set[str] = field(default_factory=set)
    specs: dict[str, set[float]] = field(default_factory=dict)
    genders: set[str] = field(default_factory=set)
    colors: set[str] = field(default_factory=set)
    types: set[str] = field(default_factory=set)
    is_accessory: bool = False
    condition: str = "new"                      # "new" | "refurbished" | "open_box" | "used"
    category: str = "general"


def normalise(text: str | None) -> str:
    text = (text or "").lower().replace("’", "'").replace("&", " and ")
    return re.sub(r"\s+", " ", text).strip()


def clean_brand(brand: str | None) -> str | None:
    if not brand:
        return None
    brand = re.sub(r"^(?:visit the|brand:?)\s*", "", brand.strip(), flags=re.I)
    brand = re.sub(r"\s+store$", "", brand, flags=re.I).strip(" -|:")
    return brand.lower() or None


_INCLUDED_IN_BOX = re.compile(
    r"\b(?:without|with|no|incl\w*|free|plus)\s+(?:an?\s+|the\s+|fast\s+|original\s+|\d+\s*w\s+)?"
    r"(?:charger|adapter|adaptor|cable|case|cover|earphones?|earbuds?|strap|stylus|protector)s?\b"
    r"|\b(?:charger|adapter|adaptor|cable|case|cover)s?\s+(?:included|in the box|in box|not included)\b"
    r"|\+\s*(?:charger|case|cover|cable)\b"
)


def _words(text: str) -> list[str]:
    return _WORDS.findall(normalise(text))


_RAM_AFTER = re.compile(r"\s*(?:ram|memory|ddr\d*|lpddr\d*)\b")
_RAM_BEFORE = re.compile(r"(?:ram|memory)\s*[:\-]?\s*$")
_STORAGE_AFTER = re.compile(r"\s*(?:ssd|hdd|rom|storage|internal|emmc|ufs|nvme)\b")
_STORAGE_BEFORE = re.compile(r"(?:ssd|hdd|rom|storage)\s*[:\-]?\s*$")


def _memory_specs(lowered: str, matches: list) -> dict:
    """Split GB/TB figures into RAM and storage using labels, else by magnitude."""
    ram: set = set()
    storage: set = set()
    unlabeled: list = []
    for match, gigabytes in matches:
        after = lowered[match.end(): match.end() + 12]
        before = lowered[max(0, match.start() - 10): match.start()]
        if _RAM_AFTER.match(after) or _RAM_BEFORE.search(before):
            ram.add(gigabytes)
        elif _STORAGE_AFTER.match(after) or _STORAGE_BEFORE.search(before) or (match.group(2) or "").lower() == "tb":
            storage.add(gigabytes)
        else:
            unlabeled.append(gigabytes)
    for value in unlabeled:
        # 6GB,128GB -> RAM 6, storage 128; a lone 8GB is memory, a lone 256GB is storage
        if value >= 32 or (ram and not storage and value > max(ram)):
            storage.add(value)
        else:
            ram.add(value)
    out: dict = {}
    if ram:
        out["ram_gb"] = {round(v, 2) for v in ram}
    if storage:
        out["storage_gb"] = {round(v, 2) for v in storage}
    return out


def extract_specs(text: str) -> dict[str, set[float]]:
    """Return {unit: {values}} for storage, RAM, weight, volume, size, power, pack size..."""
    specs: dict[str, set[float]] = {}
    lowered = normalise(text)
    memory: list = []
    for match in _SPEC.finditer(lowered):
        if match.group(3):                                   # 6.7"
            unit, raw = "inch", match.group(3)
        else:
            raw, unit_text = match.group(1), match.group(2).lower()
            unit = _UNIT_NORMALISE.get(unit_text)
            if unit == "g" and unit_text == "g" and re.fullmatch(r"[2-6]", raw):
                continue                                      # "5g" is the network, not 5 grams
        if not unit:
            continue
        value = float(raw)
        if unit in ("gb", "tb"):
            memory.append((match, value * 1024 if unit == "tb" else value))
            continue
        # normalise to base units so 1.5 kg == 1500 g and 1 l == 1000 ml
        if unit == "kg":
            unit, value = "g", value * 1000
        elif unit == "l":
            unit, value = "ml", value * 1000
        elif unit == "cm":
            unit, value = "inch", round(value / 2.54, 1)
        elif unit == "mm" and value >= 200:
            unit, value = "inch", round(value / 25.4, 1)
        specs.setdefault(unit, set()).add(round(value, 2))
    specs.update(_memory_specs(lowered, memory))
    for match in _PACK.finditer(lowered):
        number = next((g for g in match.groups() if g), None)
        if number and 1 < int(number) <= 100:
            specs.setdefault("pcs", set()).add(float(number))
    return specs


def _model_codes(words: list[str], spec_words: set[str]) -> dict[str, str]:
    noise = {"5g", "4g", "3g", "4k", "8k", "3d", "2k", "1080p", "720p", "2in1", "3in1", "ip67", "ip68", "ip54", "usb-c", "type-c"}
    noise_pattern = re.compile(r"\d+-?in-?\d+|ip\d{2}|\d+k|\d+p|\d+hz|\d+gb|\d+mp")
    codes: dict[str, str] = {}
    for word in words:
        if word in noise or word in spec_words or len(word) < 3 or noise_pattern.fullmatch(word):
            continue
        if re.search(r"[a-z]", word) and re.search(r"\d", word):
            codes.setdefault(word.replace("-", "").replace("'", ""), word)
    return codes


def _variants(head: list[str]) -> set[str]:
    out = set()
    for index, word in enumerate(head):
        if word not in VARIANT_WORDS:
            continue
        if word == "air" and index + 1 < len(head) and head[index + 1] in {"condition", "conditioner", "conditioning", "purifier", "fryer", "cooler", "fresh"}:
            continue
        out.add(word)
    return out


def parse_identity(title: str | None, brand: str | None = None) -> ProductIdentity:
    title = title or ""
    original_title = title
    condition = detect_condition(title)
    if condition != "new":
        title = strip_condition_words(title)         # "Refurbished"/"Renewed" describe the offer, not the product name
    # drop the marketing tail: "Name (variant) | feature | feature" -> keep text before the first '|'
    headline = re.split(r"\s*\|\s*", title, maxsplit=1)[0]
    words = _words(headline)
    lead = headline.split("(", 1)[0] if "(" in headline and len(_words(headline.split("(", 1)[0])) >= 3 else headline
    lead_words = _words(lead)
    all_words = _words(title)

    specs = extract_specs(title)
    spec_words = {m.group(0).replace(" ", "").lower() for m in _SPEC.finditer(normalise(title))
                  if not re.fullmatch(r"[2-6]g", m.group(0).replace(" ", "").lower())}   # 5g is a network, not a spec
    spec_words |= {re.sub(r"[^a-z0-9]", "", w) for w in spec_words}

    brand_clean = clean_brand(brand)
    lead_text = normalise(lead)
    for match in sorted(_SPEC.finditer(lead_text), key=lambda m: m.start(), reverse=True):
        if not re.fullmatch(r"[2-6]\s*g", match.group(0)):
            lead_text = lead_text[: match.start()] + " " + lead_text[match.end():]
    lead_words = _words(lead_text)
    head: list[str] = []
    for word in lead_words:
        if word in STOPWORDS or word in COLORS or word in spec_words or re.fullmatch(r"\d+(?:gb|tb)", word):
            continue
        head.append(word)
        if len(head) >= 8:
            break
    if not brand_clean and head:
        brand_clean = head[0]

    tokens = {w for w in all_words if w not in STOPWORDS and w not in COLORS and w not in spec_words}
    codes = _model_codes(all_words, spec_words)
    bare = {w for w in head if re.fullmatch(r"\d{1,3}", w)}
    # numbers that belong to a spec ("128" in "128 gb") are not model numbers
    spec_values = {str(int(v)) for values in specs.values() for v in values if float(v).is_integer()}
    bare -= spec_values

    genders = {GENDERS[w] for w in all_words if w in GENDERS}
    types = {kind for kind, vocab in PRODUCT_TYPES.items() if vocab & set(words)}
    # Accessory words count only in the headline, and not when the title merely says what is (not) in the
    # box: "... | Without Charger" or "with Case" describe a phone, not a charger or a case.
    headline_text = _INCLUDED_IN_BOX.sub(" ", normalise(headline))
    accessory = bool(ACCESSORY_WORDS & set(_words(headline_text))) or bool(re.search(
        r"\bcompatible with\b|\bfor (?:samsung|apple|iphone|oneplus|redmi|xiaomi|realme|oppo|vivo|poco|moto)", headline_text))

    identity = ProductIdentity(
        title=original_title, brand=brand_clean, head_tokens=head, tokens=tokens, model_codes=set(codes), code_forms=codes, bare_numbers=bare,
        # variants come from the headline only: "RAM Plus" in a feature list is not a variant
        variants=_variants(head),
        specs=specs, genders=genders,
        colors={w for w in all_words if w in COLORS}, types=types, is_accessory=accessory, condition=condition,
    )
    identity.category = infer_category(identity, all_words)
    return identity


# Brands that settle the category on their own when the title has no product-type word.
BRAND_CATEGORY = {
    "fashion": {"adidas", "nike", "puma", "reebok", "skechers", "asics", "fila", "levis", "levi's", "woodland", "bata",
                "campus", "wrangler", "lee", "pepe", "uspolo", "zara", "roadster", "hrx", "wildcraft", "decathlon"},
    "electronics": {"samsung", "apple", "oneplus", "xiaomi", "redmi", "realme", "oppo", "vivo", "poco", "motorola", "nokia",
                    "sony", "lg", "panasonic", "bosch", "whirlpool", "haier", "voltas", "daikin", "hp", "dell", "lenovo",
                    "asus", "acer", "boat", "jbl", "philips", "bajaj", "prestige", "havells"},
    "beauty": {"maybelline", "lakme", "loreal", "mamaearth", "nivea", "himalaya", "lotus", "biotique", "garnier",
               "neutrogena", "cetaphil", "minimalist", "plum", "wow", "dove", "sugar", "nykaa"},
}


def _stem(word: str) -> str:
    """Crude plural folding so "sneaker" and "sneakers" are the same word."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def infer_category(identity: ProductIdentity, words: list[str] | None = None) -> str:
    """Word-boundary category guess (the legacy substring classifier is left untouched)."""
    vocabulary = {_stem(w) for w in (words or _words(identity.title))}
    if identity.is_accessory and (vocabulary & {"case", "cover", "protector", "guard", "tempered", "charger", "cable"}):
        return "electronics"
    scores = {name: len(vocabulary & {_stem(k) for k in keys}) for name, keys in CATEGORY_KEYWORDS.items()}
    if "ram_gb" in identity.specs or "storage_gb" in identity.specs:
        scores["electronics"] += 2
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    brand_words = set((identity.brand or "").split()) | set(identity.head_tokens[:2])
    for category, brands in BRAND_CATEGORY.items():
        if brand_words & brands:
            return category
    return "general"


def build_queries(identity: ProductIdentity, max_queries: int = 3) -> list[str]:
    """Search strings from most to least specific.

    1. brand + model words + key specs        ("samsung galaxy m34 5g 128gb")
    2. brand + model words                    ("samsung galaxy m34")
    3. first four meaningful words            (fallback when the store has no exact model)
    """
    brand = identity.brand or ""
    brand_words = set(brand.split())
    head = [w for w in identity.head_tokens if w not in brand_words]

    core = ([brand] if brand else []) + head[:4]
    # keep the model code(s) even when they appear late in the title
    for code in sorted(identity.model_codes):
        if code not in core and not any(code == c.replace("-", "") for c in core):
            core.append(identity.code_forms.get(code, code))
    core = [w for w in core if w][:6]

    spec_terms: list[str] = []
    if "5g" in identity.tokens and "5g" not in core:
        spec_terms.append("5g")
    storage = identity.specs.get("storage_gb")
    if storage:
        spec_terms.append(f"{int(max(storage))}gb")
    for unit in ("ml", "g", "ton", "pcs"):
        values = identity.specs.get(unit)
        if values:
            number = sorted(values)[0]
            number = int(number) if float(number).is_integer() else number
            spec_terms.append(f"{number}{unit}" if unit in ("ml", "g") else f"{number} {unit}")
    if "inch" in identity.specs and identity.types & {"tv", "laptop", "tablet"}:
        number = sorted(identity.specs["inch"])[0]
        spec_terms.append(f"{int(number) if float(number).is_integer() else number} inch")

    queries: list[str] = []
    for candidate in (
        " ".join(core + spec_terms),
        " ".join(core),
        " ".join(([brand] if brand else []) + head[:3]) if (brand or head) else "",
    ):
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate and candidate not in queries:
            queries.append(candidate)
    return queries[:max_queries]
