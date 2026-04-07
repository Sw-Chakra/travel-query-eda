import pandas as pd
import re
from collections import defaultdict

df = pd.read_csv("/Users/swagatachakraborty/Claude_workspace/teleport/data/travel_search_queries.csv")

print("=" * 70)
print("1. BASIC OVERVIEW")
print("=" * 70)
print(f"Total queries       : {len(df)}")
print(f"Total categories    : {df['category'].nunique()}")
print(f"Categories          : {sorted(df['category'].unique())}")
print(f"Missing values      :\n{df.isnull().sum()}\n")

# ── 2. Queries per category ──────────────────────────────────────────────────
print("=" * 70)
print("2. QUERY COUNT PER CATEGORY")
print("=" * 70)
cat_counts = df['category'].value_counts().reset_index()
cat_counts.columns = ['Category', 'Count']
cat_counts['% Share'] = (cat_counts['Count'] / len(df) * 100).round(1)
print(cat_counts.to_string(index=False))

# ── 3. Query length stats per category ──────────────────────────────────────
print("\n" + "=" * 70)
print("3. QUERY LENGTH (in characters) PER CATEGORY")
print("=" * 70)
df['query_len'] = df['query'].str.len()
df['word_count'] = df['query'].str.split().str.len()

len_stats = df.groupby('category')['query_len'].agg(
    Avg=lambda x: round(x.mean(), 1),
    Min='min',
    Max='max',
    Median='median'
).reset_index().rename(columns={'category': 'Category'})
print(len_stats.to_string(index=False))

print("\n--- Same stats by WORD COUNT ---")
wc_stats = df.groupby('category')['word_count'].agg(
    Avg=lambda x: round(x.mean(), 1),
    Min='min',
    Max='max',
    Median='median'
).reset_index().rename(columns={'category': 'Category'})
print(wc_stats.to_string(index=False))

# ── 4. Multi-category overlap detection ──────────────────────────────────────
print("\n" + "=" * 70)
print("4. QUERIES THAT LIKELY SPAN MULTIPLE CATEGORIES (Heuristic)")
print("=" * 70)

category_signals = {
    'Flight':      r'\b(flight|flights|fly|airline|airways|airfare|plane|depart|arrive|layover|nonstop|round.?trip|one.?way|economy|business.?class|first.?class)\b',
    'Stay':        r'\b(hotel|hotels|resort|resorts|hostel|motel|airbnb|accommodation|stay|room|suite|lodge|villa|rental|inn|bed and breakfast|b&b|checkout|check.?in)\b',
    'Things to Do':r'\b(things to do|attraction|tour|tours|activity|activities|sightseeing|festival|museum|beach|hike|hiking|adventure|experience|event|nightlife|restaurant|food|cuisine|spa|theme park|cruise)\b',
    'Home':        r'\b(travel tips|travel guide|travel blog|travel insurance|visa|passport|currency|budget travel|solo travel|family vacation|honeymoon|packing|travel rewards|vaccination|best time to visit|travel advisory|cost of living|expat|moving to|relocat)\b',
    'Railways':    r'\b(train|trains|rail|railway|railways|high.?speed rail|bullet train|amtrak|eurail|subway|metro|transit|tram)\b',
}

def detect_categories(query):
    q = query.lower()
    matched = [cat for cat, pattern in category_signals.items()
               if re.search(pattern, q)]
    return matched

df['detected_cats'] = df['query'].apply(detect_categories)
df['n_detected']    = df['detected_cats'].apply(len)

# Overlap = detected 2+ categories OR detected category differs from assigned
df['multi_cat'] = df['n_detected'] >= 2
df['wrong_cat']  = df.apply(
    lambda r: len(r['detected_cats']) > 0 and r['category'] not in r['detected_cats'], axis=1
)
df['overlap_flag'] = df['multi_cat'] | df['wrong_cat']

overlap_df = df[df['overlap_flag']].copy()
print(f"Estimated overlapping queries : {len(overlap_df)} ({len(overlap_df)/len(df)*100:.1f}% of total)\n")
print("Top 20 sample overlapping queries:")
sample_overlap = overlap_df[['query', 'category', 'detected_cats']].head(20)
sample_overlap.columns = ['Query', 'Assigned Category', 'Likely Also Fits']
print(sample_overlap.to_string(index=False))

# ── 5. Vague query detection ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. VAGUE / AMBIGUOUS QUERIES")
print("=" * 70)
print("(Assumption: vague = very short query OR matches generic travel terms with no specific destination/action)\n")

vague_patterns = r'^(travel|trip|vacation|holiday|explore|visit|tourism|journey|getaway|weekend|escape)(\s+\w+)?$'
df['is_vague'] = (
    (df['word_count'] <= 3) &
    (df['query'].str.lower().str.match(vague_patterns) |
     (df['n_detected'] == 0))          # no category signal detected at all
)

vague_df = df[df['is_vague']].copy()
print(f"Estimated vague queries : {len(vague_df)} ({len(vague_df)/len(df)*100:.1f}% of total)\n")
print("Top 20 sample vague queries:")
print(vague_df[['query', 'category']].head(20).to_string(index=False))

# ── 6. Non-English term detection ────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. QUERIES WITH NON-ENGLISH TERMS")
print("=" * 70)
print("(Assumption: flagging queries containing non-ASCII characters as a proxy for non-English terms)\n")

df['has_non_ascii'] = df['query'].apply(lambda q: bool(re.search(r'[^\x00-\x7F]', str(q))))
non_eng = df[df['has_non_ascii']]
print(f"Queries with non-ASCII characters : {len(non_eng)} ({len(non_eng)/len(df)*100:.1f}% of total)\n")
if len(non_eng) > 0:
    print("Sample non-ASCII queries:")
    print(non_eng[['query', 'category']].head(20).to_string(index=False))
else:
    print("No non-ASCII characters found. Trying Unicode range check for CJK / Cyrillic / Arabic...")
    df['has_unicode_script'] = df['query'].apply(
        lambda q: bool(re.search(
            r'[\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]',
            str(q)
        ))
    )
    uni_df = df[df['has_unicode_script']]
    print(f"Queries with non-Latin scripts : {len(uni_df)}")

# ── 7. Other key insights ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. OTHER KEY INSIGHTS")
print("=" * 70)

# 7a. Most common destinations
print("\n--- Top 20 Most Mentioned Destinations (by word frequency in queries) ---")
stop = {'to','from','in','at','for','the','a','an','and','or','of','with','near',
        'best','cheap','luxury','flights','hotels','travel','how','what','when',
        'where','is','are','do','i','my','on','by','per','vs','that','this',
        'its','it','about','can','get','find','book','top','good','great','new',
        'things','trip','guide','tips','plan','rent','rent','car','day','night',
        'stay','rent','around','between','into','out','off','up','down','back',
        'round','one','way','class','package','packages','high','speed','airline',
        'train','bus','via','within','cost','price','cheap','budget','mid','range',
        'mid-range','airfare','nonstop','book','booking'}
all_words = []
for q in df['query'].str.lower():
    words = re.findall(r'\b[a-z][a-z]+\b', q)
    all_words.extend([w for w in words if w not in stop and len(w) > 3])
from collections import Counter
word_freq = Counter(all_words)
print(pd.DataFrame(word_freq.most_common(20), columns=['Term', 'Frequency']).to_string(index=False))

# 7b. Longest and shortest queries
print("\n--- Top 5 Longest Queries ---")
print(df.nlargest(5, 'query_len')[['query', 'category', 'query_len']].to_string(index=False))
print("\n--- Top 5 Shortest Queries ---")
print(df.nsmallest(5, 'query_len')[['query', 'category', 'query_len']].to_string(index=False))

# 7c. Query patterns / common prefixes
print("\n--- Most Common Query Starters (first 2 words) ---")
df['first_two'] = df['query'].str.lower().str.split().str[:2].str.join(' ')
print(df['first_two'].value_counts().head(15).reset_index().rename(
    columns={'index': 'Prefix', 'first_two': 'Count', 'count': 'Count'}
).to_string(index=False))

print("\n--- Done ---")
