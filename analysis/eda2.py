import pandas as pd
import re
from collections import Counter

df = pd.read_csv("/Users/swagatachakraborty/Claude_workspace/teleport/data/travel_search_queries.csv")
df['query_lower'] = df['query'].str.lower()
df['word_count']  = df['query'].str.split().str.len()

# ── Recompute flags from eda.py ───────────────────────────────────────────────
category_signals = {
    'Flight':       r'\b(flight|flights|fly|airline|airways|airfare|plane|depart|arrive|layover|nonstop|round.?trip|one.?way|economy|business.?class|first.?class)\b',
    'Stay':         r'\b(hotel|hotels|resort|resorts|hostel|motel|airbnb|accommodation|stay|room|suite|lodge|villa|rental|inn|bed and breakfast|b&b|checkout|check.?in)\b',
    'Things to Do': r'\b(things to do|attraction|tour|tours|activity|activities|sightseeing|festival|museum|beach|hike|hiking|adventure|experience|event|nightlife|restaurant|food|cuisine|spa|theme park|cruise)\b',
    'Home':         r'\b(travel tips|travel guide|travel blog|travel insurance|visa|passport|currency|budget travel|solo travel|family vacation|honeymoon|packing|travel rewards|vaccination|best time to visit|travel advisory|cost of living|expat|moving to|relocat)\b',
    'Railways':     r'\b(train|trains|rail|railway|railways|high.?speed rail|bullet train|amtrak|eurail|subway|metro|transit|tram)\b',
}

def detect_cats(q):
    return [c for c, p in category_signals.items() if re.search(p, q.lower())]

df['detected_cats'] = df['query'].apply(detect_cats)
df['n_detected']    = df['detected_cats'].apply(len)
df['multi_cat']     = df['n_detected'] >= 2
df['wrong_cat']     = df.apply(lambda r: len(r['detected_cats']) > 0 and r['category'] not in r['detected_cats'], axis=1)
df['overlap_flag']  = df['multi_cat'] | df['wrong_cat']

vague_pat = r'^(travel|trip|vacation|holiday|explore|visit|tourism|journey|getaway|weekend|escape)(\s+\w+)?$'
df['is_vague'] = (
    (df['word_count'] <= 3) &
    (df['query_lower'].str.match(vague_pat) | (df['n_detected'] == 0))
)

# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("A. VAGUE QUERIES — CATEGORY DISTRIBUTION")
print("=" * 70)
vague_df   = df[df['is_vague']]
vague_cats = vague_df['category'].value_counts().reset_index()
vague_cats.columns = ['Category', 'Count']
vague_cats['% of Vague'] = (vague_cats['Count'] / len(vague_df) * 100).round(1)
vague_cats['% of Category Total'] = vague_cats.apply(
    lambda r: round(r['Count'] / len(df[df['category'] == r['Category']]) * 100, 1), axis=1
)
print(f"Total vague queries : {len(vague_df)} ({len(vague_df)/len(df)*100:.1f}% of all queries)\n")
print(vague_cats.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("B. MULTI-CATEGORY OVERLAP — CATEGORY DISTRIBUTION")
print("=" * 70)
overlap_df   = df[df['overlap_flag']]
overlap_cats = overlap_df['category'].value_counts().reset_index()
overlap_cats.columns = ['Category', 'Count']
overlap_cats['% of Overlap'] = (overlap_cats['Count'] / len(overlap_df) * 100).round(1)
overlap_cats['% of Category Total'] = overlap_cats.apply(
    lambda r: round(r['Count'] / len(df[df['category'] == r['Category']]) * 100, 1), axis=1
)
print(f"Total overlapping queries : {len(overlap_df)} ({len(overlap_df)/len(df)*100:.1f}% of all queries)\n")
print(overlap_cats.to_string(index=False))

print("\n--- Most common category pair combinations ---")
pair_counter = Counter()
for cats in overlap_df['detected_cats']:
    cats_sorted = tuple(sorted(cats))
    if len(cats_sorted) >= 2:
        for i in range(len(cats_sorted)):
            for j in range(i+1, len(cats_sorted)):
                pair_counter[(cats_sorted[i], cats_sorted[j])] += 1
pair_df = pd.DataFrame(pair_counter.most_common(10), columns=['Category Pair', 'Count'])
print(pair_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("C. DESTINATION MENTIONS")
print("=" * 70)
print("(Assumption: matching against a curated list of ~200 popular travel cities & countries)\n")

cities = [
    "amsterdam","athens","atlanta","auckland","bali","bangkok","barcelona",
    "beijing","berlin","bogota","bora bora","brussels","budapest","buenos aires",
    "cairo","cancun","cape town","cartagena","chicago","copenhagen","dallas",
    "delhi","dubai","dublin","edinburgh","florence","geneva","hanoi","havana",
    "hong kong","honolulu","istanbul","jakarta","johannesburg","karachi","kathmandu",
    "kuala lumpur","kyoto","lagos","las vegas","lima","lisbon","london","los angeles",
    "madrid","maldives","manila","marrakech","melbourne","mexico city","miami",
    "milan","montreal","moscow","mumbai","munich","nairobi","new orleans",
    "new york","oslo","paris","prague","quebec city","queenstown","reykjavik",
    "rio de janeiro","rome","san diego","san francisco","santiago","sarajevo",
    "seattle","seoul","shanghai","singapore","stockholm","sydney","taipei","tokyo",
    "toronto","ulaanbaatar","vancouver","venice","vienna","warsaw","washington dc",
    "zurich","amalfi coast","phuket","vientiane","krakow","montevideo","dubrovnik",
    "riga","doha","abu dhabi","muscat","accra","casablanca","tunis","dar es salaam",
    "addis ababa","seattle","portland","denver","phoenix","houston","boston",
    "atlanta","nashville","orlando","miami","tampa","new zealand","maldives",
    "fiji","hawaii","puerto rico","jamaica","bahamas","costa rica","belize",
    "peru","colombia","argentina","brazil","chile","mexico","japan","china",
    "india","thailand","vietnam","indonesia","malaysia","philippines","australia",
    "canada","france","germany","italy","spain","portugal","greece","turkey",
    "egypt","morocco","south africa","kenya","tanzania","iceland","norway",
    "sweden","denmark","finland","netherlands","belgium","switzerland","austria",
    "czech republic","poland","hungary","croatia","scotland","ireland","wales",
    "england","united kingdom","united states","new zealand","singapore",
    "goa","pattaya","chiang mai","siem reap","cambodia","nepal","sri lanka",
    "maldives","seychelles","mauritius","zanzibar","santorini","mykonos",
    "cinque terre","tuscany","provence","normandy","andalusia","catalonia",
]

def find_destinations(query):
    q = query.lower()
    return [c for c in cities if re.search(r'\b' + re.escape(c) + r'\b', q)]

df['destinations'] = df['query'].apply(find_destinations)
df['has_destination'] = df['destinations'].apply(lambda d: len(d) > 0)
df['n_destinations']  = df['destinations'].apply(len)

dest_total = df['has_destination'].sum()
print(f"Queries mentioning at least one destination : {dest_total} ({dest_total/len(df)*100:.1f}%)")
print(f"Queries with multiple destinations          : {(df['n_destinations'] > 1).sum()} ({(df['n_destinations']>1).sum()/len(df)*100:.1f}%)\n")

print("--- Destination mentions by category ---")
dest_by_cat = df.groupby('category').agg(
    Total=('query', 'count'),
    With_Destination=('has_destination', 'sum')
).reset_index()
dest_by_cat['% with Destination'] = (dest_by_cat['With_Destination'] / dest_by_cat['Total'] * 100).round(1)
print(dest_by_cat.to_string(index=False))

print("\n--- Top 30 most mentioned destinations ---")
all_dests = [d for dests in df['destinations'] for d in dests]
dest_freq = Counter(all_dests)
dest_df   = pd.DataFrame(dest_freq.most_common(30), columns=['Destination', 'Mentions'])
print(dest_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D. TEMPORAL SIGNALS IN QUERIES")
print("=" * 70)

temporal_patterns = {
    'Month':        r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b',
    'Season':       r'\b(summer|winter|spring|autumn|fall|monsoon|peak season|off.?season|shoulder season)\b',
    'Year':         r'\b(202[3-9]|203[0-9])\b',
    'Day/Week':     r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|weekday|next week|this week)\b',
    'Urgency':      r'\b(last.?minute|urgent|asap|same.?day|tonight|today|tomorrow|this weekend|immediately|last minute)\b',
    'Relative':     r'\b(next month|this month|next year|this year|upcoming|soon|in \d+ days|in \d+ weeks)\b',
    'Duration':     r'\b(\d+\s*(day|days|night|nights|week|weeks|month|months)\b)',
}

for label, pattern in temporal_patterns.items():
    df[f'temp_{label}'] = df['query_lower'].str.contains(pattern, regex=True)

df['has_temporal'] = df[[f'temp_{k}' for k in temporal_patterns]].any(axis=1)

print(f"Queries with any temporal signal : {df['has_temporal'].sum()} ({df['has_temporal'].sum()/len(df)*100:.1f}%)\n")
print("--- Breakdown by temporal type ---")
temp_summary = []
for label in temporal_patterns:
    col = f'temp_{label}'
    n   = df[col].sum()
    temp_summary.append({'Temporal Type': label, 'Count': n, '% of All Queries': round(n/len(df)*100,1)})
print(pd.DataFrame(temp_summary).to_string(index=False))

print("\n--- Temporal signals by category ---")
temp_by_cat = df.groupby('category')['has_temporal'].agg(['sum','count']).reset_index()
temp_by_cat.columns = ['Category', 'With Temporal', 'Total']
temp_by_cat['% with Temporal'] = (temp_by_cat['With Temporal'] / temp_by_cat['Total'] * 100).round(1)
print(temp_by_cat.to_string(index=False))

print("\n--- Sample temporal queries ---")
print(df[df['has_temporal']][['query','category']].sample(15, random_state=42).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("E. PRICE TIER — BUDGET vs LUXURY")
print("=" * 70)

budget_pat  = r'\b(budget|cheap|cheapest|affordable|low.?cost|economy|backpack|hostel|discount|deal|deals|saver|value|inexpensive|low budget|frugal|price comparison)\b'
luxury_pat  = r'\b(luxury|luxurious|premium|first.?class|business.?class|5.?star|five.?star|exclusive|high.?end|upscale|deluxe|suite|boutique hotel|resort|private|vip|opulent|lavish)\b'
mid_pat     = r'\b(mid.?range|moderate|standard|comfortable|3.?star|three.?star|4.?star|four.?star|decent)\b'

df['is_budget']  = df['query_lower'].str.contains(budget_pat,  regex=True)
df['is_luxury']  = df['query_lower'].str.contains(luxury_pat,  regex=True)
df['is_midrange'] = df['query_lower'].str.contains(mid_pat,    regex=True)
df['has_price_tier'] = df['is_budget'] | df['is_luxury'] | df['is_midrange']

total_priced = df['has_price_tier'].sum()
print(f"Queries with a price-tier signal : {total_priced} ({total_priced/len(df)*100:.1f}%)\n")

tier_summary = [
    {'Tier': 'Budget / Cheap',   'Count': int(df['is_budget'].sum()),   '% of All': round(df['is_budget'].sum()/len(df)*100,1)},
    {'Tier': 'Luxury / Premium', 'Count': int(df['is_luxury'].sum()),   '% of All': round(df['is_luxury'].sum()/len(df)*100,1)},
    {'Tier': 'Mid-range',        'Count': int(df['is_midrange'].sum()), '% of All': round(df['is_midrange'].sum()/len(df)*100,1)},
]
print(pd.DataFrame(tier_summary).to_string(index=False))

print("\n--- Price tier by category ---")
for tier, col in [('Budget', 'is_budget'), ('Luxury', 'is_luxury'), ('Mid-range', 'is_midrange')]:
    cat_dist = df[df[col]]['category'].value_counts()
    print(f"\n{tier}:")
    print(cat_dist.to_string())

print("\n--- Sample budget queries ---")
print(df[df['is_budget']][['query','category']].head(8).to_string(index=False))
print("\n--- Sample luxury queries ---")
print(df[df['is_luxury']][['query','category']].head(8).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("F. OTHER NOTABLE THEMES")
print("=" * 70)

themes = {
    'Solo Travel':         r'\b(solo|alone|solo traveler|solo female|single traveler|travelling alone)\b',
    'Family / Kids':       r'\b(family|families|kids|children|child|toddler|infant|baby|family.?friendly)\b',
    'Couple / Honeymoon':  r'\b(couple|couples|honeymoon|romantic|anniversary|partner|spouse)\b',
    'Group Travel':        r'\b(group|groups|group travel|group tour|bachelor|bachelorette|friends)\b',
    'Female Solo':         r'\b(female solo|solo female|women solo|women traveling|woman traveling|solo woman)\b',
    'Pet Friendly':        r'\b(pet|pets|dog|dogs|cat|cats|pet.?friendly|with pets)\b',
    'Accessibility':       r'\b(wheelchair|accessible|disability|disabled|mobility|handicap)\b',
    'Sustainable / Eco':   r'\b(sustainable|eco|green travel|eco.?friendly|responsible travel|carbon|offset)\b',
    'Digital Nomad':       r'\b(digital nomad|remote work|work remotely|co.?working|work from|nomad)\b',
    'Visa / Docs':         r'\b(visa|passport|entry requirements|immigration|customs|work permit|residency)\b',
    'Health / Safety':     r'\b(vaccination|vaccine|travel insurance|safety|safe to travel|health|medical|covid|pcr)\b',
    'All-Inclusive':       r'\b(all.?inclusive|all inclusive|package deal|package tour|resort package)\b',
    'Loyalty / Points':    r'\b(miles|points|rewards|frequent flyer|loyalty|lounge access|status match|elite)\b',
    'Multi-city / Route':  r'\b(multi.?city|multi city|road trip|itinerary|route|stopover|layover)\b',
    'Rental Car':          r'\b(car rental|rent a car|vehicle rental|van rental|campervan|motorhome|suv rental|minivan)\b',
}

theme_results = []
for theme, pattern in themes.items():
    n = df['query_lower'].str.contains(pattern, regex=True).sum()
    theme_results.append({'Theme': theme, 'Count': n, '% of All Queries': round(n/len(df)*100,1)})

theme_df = pd.DataFrame(theme_results).sort_values('Count', ascending=False)
print(theme_df.to_string(index=False))

print("\n--- Sample queries per theme (top 5 themes) ---")
for theme, pattern in list(themes.items())[:5]:
    mask = df['query_lower'].str.contains(pattern, regex=True)
    print(f"\n{theme} ({mask.sum()} queries):")
    print(df[mask][['query','category']].head(5).to_string(index=False))

print("\n--- Done ---")
