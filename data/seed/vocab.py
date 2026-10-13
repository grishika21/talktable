"""Word lists and text templates for the demo data generator (`generate.py`).

Kept separate so the generator logic stays readable. Nothing in here is random on its
own: `generate.py` picks from these lists with a seeded RNG.
"""

from __future__ import annotations

# --- Categories --------------------------------------------------------------
# (name, relative share of the catalog, typical price range in USD, page range)
CATEGORIES: list[tuple[str, float, tuple[int, int], tuple[int, int]]] = [
    ("Fiction", 1.5, (9, 28), (180, 620)),
    ("Mystery", 1.2, (8, 26), (220, 480)),
    ("Thriller", 1.1, (8, 26), (240, 520)),
    ("Science Fiction", 1.0, (9, 30), (260, 680)),
    ("Fantasy", 1.1, (10, 35), (300, 950)),
    ("Romance", 1.2, (6, 20), (200, 420)),
    ("Historical Fiction", 0.8, (10, 30), (280, 640)),
    ("Horror", 0.6, (8, 24), (220, 480)),
    ("Biography", 0.7, (14, 38), (240, 720)),
    ("History", 0.7, (15, 45), (260, 900)),
    ("Science", 0.7, (14, 48), (200, 560)),
    ("Self-Help", 0.8, (10, 30), (160, 380)),
    ("Business", 0.7, (15, 50), (180, 420)),
    ("Children's", 0.9, (5, 18), (24, 160)),
    ("Poetry", 0.4, (8, 22), (48, 220)),
]

NONFICTION = {"Biography", "History", "Science", "Self-Help", "Business"}

# --- People & places ---------------------------------------------------------
# Author nationalities: (Faker locale, country, weight)
AUTHOR_ORIGINS: list[tuple[str, str, float]] = [
    ("en_US", "United States", 30),
    ("en_GB", "United Kingdom", 16),
    ("en_CA", "Canada", 7),
    ("en_AU", "Australia", 5),
    ("en_IE", "Ireland", 4),
    ("en_NZ", "New Zealand", 2),
    ("en_IN", "India", 8),
    ("fr_FR", "France", 6),
    ("de_DE", "Germany", 5),
    ("es_ES", "Spain", 4),
    ("it_IT", "Italy", 3),
    ("es_MX", "Mexico", 3),
    ("pt_BR", "Brazil", 3),
    ("nl_NL", "Netherlands", 2),
    ("sv_SE", "Sweden", 2),
    ("pl_PL", "Poland", 2),
    ("ja_JP", "Japan", 2),
]

# Customer cities: (city, Faker locale used for the customer's name, weight)
CUSTOMER_CITIES: list[tuple[str, str, float]] = [
    ("New York", "en_US", 8),
    ("Los Angeles", "en_US", 5),
    ("Chicago", "en_US", 4),
    ("Houston", "en_US", 3),
    ("San Francisco", "en_US", 3),
    ("Seattle", "en_US", 3),
    ("Boston", "en_US", 3),
    ("Austin", "en_US", 2),
    ("London", "en_GB", 6),
    ("Manchester", "en_GB", 2),
    ("Edinburgh", "en_GB", 1.5),
    ("Dublin", "en_IE", 2),
    ("Toronto", "en_CA", 4),
    ("Vancouver", "en_CA", 2),
    ("Sydney", "en_AU", 3),
    ("Melbourne", "en_AU", 2),
    ("Mumbai", "en_IN", 3),
    ("Bengaluru", "en_IN", 3),
    ("Hyderabad", "en_IN", 2),
    ("Delhi", "en_IN", 2),
    ("Berlin", "de_DE", 2),
    ("Munich", "de_DE", 1),
    ("Paris", "fr_FR", 2),
    ("Madrid", "es_ES", 1.5),
    ("Mexico City", "es_MX", 1.5),
    ("São Paulo", "pt_BR", 1.5),
    ("Amsterdam", "nl_NL", 1.2),
    ("Stockholm", "sv_SE", 1),
    ("Milan", "it_IT", 1),
]

# Reserved example domains (RFC 2606), so no generated email can belong to a real person.
EMAIL_DOMAINS = ["example.com", "example.org", "example.net", "mail.example.com"]

# --- Book titles -------------------------------------------------------------
ADJECTIVES = [
    "Silent", "Hidden", "Broken", "Golden", "Last", "Forgotten", "Crimson", "Distant",
    "Midnight", "Endless", "Secret", "Wild", "Burning", "Quiet", "Lost", "Winter",
    "Shattered", "Hollow", "Bright", "Iron", "Paper", "Glass", "Velvet", "Electric",
    "Ancient", "Salt", "Silver", "Northern", "Drowned", "Painted", "Restless", "Borrowed",
    "Fallen", "Crooked", "Sleeping", "Final", "Scarlet", "Amber", "Frozen", "Wandering",
]
NOUNS = [
    "River", "Garden", "Empire", "Kingdom", "Shadow", "Mirror", "Promise", "Harbor",
    "Orchard", "Labyrinth", "Lighthouse", "Archive", "Compass", "Crown", "Storm", "Signal",
    "Horizon", "Map", "Letter", "Island", "Forest", "City", "Machine", "Bridge", "Tide",
    "Ember", "Station", "House", "Witness", "Stranger", "Orbit", "Throne", "Key", "Clock",
    "Lantern", "Sea", "Fire", "Door", "Feather", "Mountain",
]
PLACES = [
    "Ashford", "the North", "Saltmarsh", "Venice", "Kyoto", "the Valley", "Blackwater",
    "Marrakesh", "the Moors", "Elmstead", "the Coast", "Avalon", "Lisbon", "the Old Quarter",
    "Havenport", "Prague", "the Steppe", "Ravenhill", "Cairo", "Wrenfield",
]
VERBS = ["Falls", "Burns", "Breaks", "Wakes", "Sings", "Returns", "Sleeps"]
OCCUPATIONS = [
    "Clockmaker", "Lighthouse Keeper", "Cartographer", "Weaver", "Glassblower",
    "Beekeeper", "Bookbinder", "Tailor", "Astronomer", "Ferryman", "Apothecary", "Baker",
]
RELATIVES = ["Daughter", "Son", "Wife", "Apprentice", "Sister", "Secret"]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
STARS = ["Andromeda", "Helios", "Kepler", "Orion", "Vega", "Lyra", "Titan", "Proxima"]

HISTORY_TOPICS = [
    "Silk Road", "Printing Press", "Atlantic", "Ottoman Empire", "Cold War",
    "Industrial Revolution", "Roman Republic", "Spice Trade", "Mediterranean",
    "Hanseatic League", "Great Depression", "Space Race",
]
SCIENCE_SINGULAR = ["Universe", "Brain", "Gene", "Ocean", "Immune System", "Atom", "Cell", "Climate"]
SCIENCE_PLURAL = ["Black Holes", "Forests", "Microbes", "Stars", "Numbers", "Bees", "Volcanoes"]

SELFHELP_PEOPLE = ["Highly Focused", "Calm", "Resilient", "Creative", "Happy", "Productive"]
SELFHELP_GERUNDS = ["Saying No", "Slowing Down", "Deep Work", "Letting Go", "Starting Over", "Listening", "Rest"]
SELFHELP_NOUNS = ["Potential", "Confidence", "Creativity", "Inner Calm", "Best Self", "Focus"]
SELFHELP_THINGS = ["Morning", "Mindset", "Career", "Life", "Night's Sleep"]
SELFHELP_LITTLE = ["Calm", "Courage", "Kindness", "Focus", "Gratitude"]

BUSINESS_ADJ = ["Lean", "Patient", "Frugal", "Remote", "Resilient", "Data-Driven", "Quiet", "Humble", "Global"]
BUSINESS_VALUES = ["Empathy", "Clarity", "Purpose", "Data", "Curiosity", "Candor", "Focus", "Trust"]
BUSINESS_SCALE = ["Growth", "Trust", "Culture", "Innovation", "Strategy", "Hiring"]
BUSINESS_PLAYBOOKS = ["Product", "Sales", "Negotiation", "Pricing", "Leadership", "Marketing", "Hiring", "Fundraising"]
BUSINESS_OBJECTS = ["Spreadsheet", "Pitch Deck", "Org Chart", "Quarterly Report", "Bottom Line", "Mission Statement"]

ANIMALS = ["Fox", "Bear", "Owl", "Rabbit", "Dragon", "Penguin", "Hedgehog", "Otter", "Elephant", "Tiger", "Mouse"]
CHILDREN_ADJ = ["Brave", "Sleepy", "Curious", "Tiny", "Grumpy", "Magic", "Rainbow", "Busy"]
CHILDREN_NOUNS = ["Balloon", "Moon", "Treehouse", "Rocket", "Picnic", "Snowstorm", "Birthday", "Adventure"]

# Title patterns per category. Placeholders are filled by generate.py.
TITLE_PATTERNS: dict[str, list[str]] = {
    "Fiction": [
        "The {adj} {noun}", "{noun} of {place}", "The {noun} of {place}", "A {noun} in {place}",
        "The {adj} {nouns}", "Beyond the {adj} {noun}", "When the {noun} {verb}",
        "{first}'s {noun}", "The Last {noun} of {place}", "{adj} {nouns}",
        "The {occupation}'s {relative}",
    ],
    "Mystery": [
        "Murder at {place}", "The {place} Affair", "Death in the {adj} {noun}", "The {adj} Witness",
        "A Body in the {noun}", "The {occupation} Who Knew Too Much", "The Case of the {adj} {noun}",
    ],
    "Thriller": [
        "The {star} Protocol", "{adj} Signal", "Seven Days in {place}", "The {adj} Hour",
        "The {noun} Directive", "No Way Out of {place}", "The {place} Contract",
    ],
    "Science Fiction": [
        "{adj} Orbit", "Project {star}", "The {star} Signal", "Children of {star}",
        "The {adj} Machine", "Echoes from {star}", "The Last Station on {star}",
    ],
    "Fantasy": [
        "The {adj} Throne", "Crown of {nouns}", "The {noun} of {place}", "The {adj} Crown",
        "Daughter of {place}", "The Dragon of {place}", "A Song of {adj} {nouns}",
    ],
    "Romance": [
        "{season} in {place}", "A {adj} Promise", "Love in {place}", "The {occupation}'s Heart",
        "Meet Me at the {noun}", "One {season} in {place}", "Letters from {place}",
    ],
    "Historical Fiction": [
        "The {occupation}'s {relative}", "The {occupation} of {place}", "{place}, {year}",
        "The {adj} {noun} of {place}", "A {season} in {place}",
    ],
    "Horror": [
        "The {adj} House", "What Waits in the {noun}", "The Thing in {place}",
        "Don't Open the {noun}", "The {adj} Ones", "Night at {place}",
    ],
    "Biography": [
        "{full}: A Life", "The Life and Times of {full}", "Becoming {full}",
        "{full}: The Untold Story", "Memoirs of a {occupation}",
    ],
    "History": [
        "A Short History of the {h_topic}", "The {h_topic}: A New History", "Inside the {h_topic}",
        "The Rise and Fall of {place}", "Empires of the {h_topic}", "The {h_topic} Years",
    ],
    "Science": [
        "Inside the {s_one}", "The {s_one} Revolution", "Understanding the {s_one}",
        "The Hidden Life of {s_many}", "The Secret World of {s_many}", "A Brief Tour of the {s_one}",
        "The Story of the {s_one}", "Why {s_many} Matter",
    ],
    "Self-Help": [
        "The {n} Habits of {sh_people} People", "The Art of {sh_gerund}", "Unlock Your {sh_noun}",
        "{days} Days to a Better {sh_thing}", "The Little Book of {sh_little}",
    ],
    "Business": [
        "The {b_adj} Startup", "Leading with {b_value}", "{b_scale} at Scale",
        "The {b_play} Playbook", "Beyond the {b_object}", "The {b_value} Advantage",
    ],
    "Children's": [
        "{animal} and the {c_adj} {c_noun}", "The {c_adj} Little {animal}", "Where Do {animals} Sleep?",
        "{first} the Brave {animal}", "{animal}'s Big {c_noun}",
    ],
    "Poetry": [
        "Songs of the {noun}", "{adj} Verses", "Poems for a {adj} {season}",
        "The {noun} and Other Poems", "Letters to the {noun}",
    ],
}

# Non-fiction titles often get a subtitle ("The Lean Startup: How Small Teams Win").
SUBTITLES: dict[str, list[str]] = {
    "Business": [
        "How Great Teams Win", "Lessons from Two Decades of Building Companies",
        "A Practical Guide for New Managers", "Why Some Companies Thrive and Others Fail",
        "Building Products People Love", "The Hidden Rules of Work", "Turning Ideas into Revenue",
        "What Founders Get Wrong", "Leading Through Uncertainty", "A Field Guide for Leaders",
    ],
    "Self-Help": [
        "Small Changes, Big Results", "A Guide to a Calmer Life", "How to Stop Overthinking",
        "Simple Habits for Every Day", "Finding Balance in a Busy World", "A 30-Day Plan",
        "Why Less Is More", "Rewire Your Routine",
    ],
    "Science": [
        "What It Tells Us About Ourselves", "A Journey Through Modern Science",
        "The Discoveries That Changed Everything", "How Scientists Cracked the Code",
        "A Guide for the Curious", "From the Big Bang to Today", "The Science of Everyday Life",
    ],
    "History": [
        "How It Shaped the Modern World", "Power, Trade and Ambition", "The People Who Made It",
        "A Story of Ambition and Ruin", "From Rise to Collapse", "Voices from the Past",
    ],
}

# --- Review text -------------------------------------------------------------
REVIEW_OPENERS: dict[int, list[str]] = {
    5: ["Absolutely loved it.", "A new favourite.", "Couldn't put it down.",
        "Five stars, no hesitation.", "Wonderful from start to finish."],
    4: ["Really enjoyed this one.", "A very good read.", "Solid and satisfying.",
        "Well worth the time.", "Better than I expected."],
    3: ["It was okay.", "Mixed feelings about this one.", "Decent, but not memorable.",
        "Some good moments, some slow ones."],
    2: ["Disappointing.", "Not what I expected.", "Struggled to stay interested.",
        "Had high hopes, but it fell flat."],
    1: ["Couldn't finish it.", "A waste of money.", "One of the worst books I've read this year.",
        "Really did not work for me."],
}

# Details by (genre group, tone). Tone: "pos" for 4-5 stars, "mid" for 3, "neg" for 1-2.
REVIEW_DETAILS: dict[tuple[str, str], list[str]] = {
    ("fiction", "pos"): [
        "The characters felt real and the dialogue was sharp.",
        "The pacing kept me turning pages late into the night.",
        "The ending tied everything together beautifully.",
        "The world-building is rich without being overwhelming.",
        "Beautiful prose. I reread several passages.",
    ],
    ("fiction", "mid"): [
        "The middle section was slow, but the ending picked up.",
        "Interesting premise, uneven execution.",
        "Good characters, predictable plot.",
    ],
    ("fiction", "neg"): [
        "The plot dragged and the twist was obvious.",
        "The characters were flat and hard to care about.",
        "The ending felt rushed and unearned.",
        "Too many subplots that went nowhere.",
        "The writing was clumsy in places.",
    ],
    ("nonfiction", "pos"): [
        "Clear, well researched and genuinely useful.",
        "Explains complex ideas without dumbing them down.",
        "Full of practical takeaways I've already started using.",
        "The author's research really shows.",
    ],
    ("nonfiction", "mid"): [
        "Some useful chapters, others felt like filler.",
        "Good overview, but nothing new if you know the topic.",
    ],
    ("nonfiction", "neg"): [
        "Mostly padding; the key ideas would fit in a blog post.",
        "Repetitive and short on evidence.",
        "Felt outdated and superficial.",
        "Dense and poorly organised.",
    ],
    ("children", "pos"): [
        "My kids asked for it every night.",
        "Lovely illustrations and a sweet story.",
        "Perfect for reading aloud.",
    ],
    ("children", "mid"): [
        "Cute, but a bit long for bedtime.",
        "Nice pictures, thin story.",
    ],
    ("children", "neg"): [
        "My kids lost interest quickly.",
        "The story was confusing for young readers.",
    ],
    ("poetry", "pos"): [
        "Every poem landed. I keep coming back to it.",
        "Spare, musical and moving.",
    ],
    ("poetry", "mid"): [
        "A few standout poems in an uneven collection.",
    ],
    ("poetry", "neg"): [
        "Most of the poems felt forced.",
        "Couldn't connect with the collection at all.",
    ],
}

REVIEW_CLOSERS: dict[str, list[str]] = {
    "pos": ["Highly recommend.", "Already bought a copy for a friend.",
            "Can't wait for the author's next book.", ""],
    "mid": ["Fine for a long flight.", "Might appeal to die-hard fans.", ""],
    "neg": ["Would not recommend.", "Skip this one.", ""],
}

RETURN_COMPLAINTS = [
    "Returned it.",
    "The copy arrived damaged, so I sent it back.",
    "Sent it back after a few chapters.",
]
