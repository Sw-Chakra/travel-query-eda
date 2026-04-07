import pandas as pd
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import base64
import io

# ── Palette ───────────────────────────────────────────────────────────────────
BRAND   = "#1B4F8A"
ACCENT  = "#E8A020"
GREY    = "#6B7280"
LIGHT   = "#F3F6FA"
RED     = "#C0392B"
GREEN   = "#1A7A4A"
COLORS  = ["#1B4F8A","#E8A020","#2E86AB","#A23B72","#F18F01","#3BB273","#6B7280"]

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{enc}"

# ── Load & feature engineering ────────────────────────────────────────────────
df = pd.read_csv("/Users/swagatachakraborty/Claude_workspace/teleport/data/travel_search_queries.csv")
df['query_lower'] = df['query'].str.lower()
df['word_count']  = df['query'].str.split().str.len()
df['query_len']   = df['query'].str.len()

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
df['wrong_cat']     = df.apply(lambda r: len(r['detected_cats'])>0 and r['category'] not in r['detected_cats'], axis=1)
df['overlap_flag']  = df['multi_cat'] | df['wrong_cat']

vague_pat = r'^(travel|trip|vacation|holiday|explore|visit|tourism|journey|getaway|weekend|escape)(\s+\w+)?$'
df['is_vague'] = (df['word_count'] <= 3) & (df['query_lower'].str.match(vague_pat) | (df['n_detected']==0))

cities = [
    "amsterdam","athens","atlanta","auckland","bali","bangkok","barcelona","beijing",
    "berlin","bogota","bora bora","brussels","budapest","buenos aires","cairo","cancun",
    "cape town","cartagena","chicago","copenhagen","dallas","delhi","dubai","dublin",
    "edinburgh","florence","geneva","hanoi","havana","hong kong","honolulu","istanbul",
    "jakarta","johannesburg","kathmandu","kuala lumpur","kyoto","lagos","las vegas",
    "lima","lisbon","london","los angeles","madrid","maldives","manila","marrakech",
    "melbourne","mexico city","miami","milan","montreal","moscow","mumbai","munich",
    "nairobi","new orleans","new york","oslo","paris","prague","quebec city","queenstown",
    "reykjavik","rio de janeiro","rome","san diego","san francisco","santiago","sarajevo",
    "seattle","seoul","shanghai","singapore","stockholm","sydney","taipei","tokyo",
    "toronto","ulaanbaatar","vancouver","venice","vienna","warsaw","washington dc",
    "zurich","amalfi coast","phuket","vientiane","krakow","montevideo","dubrovnik",
    "riga","doha","abu dhabi","goa","pattaya","chiang mai","siem reap","nepal",
    "sri lanka","seychelles","mauritius","zanzibar","santorini","mykonos","tuscany",
    "hawaii","puerto rico","jamaica","bahamas","costa rica","peru","colombia",
    "argentina","brazil","chile","mexico","japan","china","india","thailand",
    "vietnam","indonesia","malaysia","philippines","australia","canada","france",
    "germany","italy","spain","portugal","greece","turkey","egypt","morocco",
    "south africa","kenya","iceland","norway","sweden","denmark","netherlands",
    "belgium","switzerland","austria","czech republic","poland","hungary","croatia",
    "ireland","finland","osaka","asuncion","almaty","yangon","taipei","cusco",
    "ho chi minh city","ho chi minh","phoenix","portland","denver","houston",
    "boston","nashville","orlando","tampa","new zealand","fiji",
]
def find_dests(q):
    return [c for c in cities if re.search(r'\b'+re.escape(c)+r'\b', q.lower())]
df['destinations']    = df['query'].apply(find_dests)
df['has_destination'] = df['destinations'].apply(lambda d: len(d)>0)
df['n_destinations']  = df['destinations'].apply(len)

temporal_patterns = {
    'Month':    r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b',
    'Season':   r'\b(summer|winter|spring|autumn|fall|monsoon|peak season|off.?season)\b',
    'Urgency':  r'\b(last.?minute|urgent|asap|same.?day|tonight|today|tomorrow|this weekend)\b',
    'Duration': r'(\d+\s*(day|days|night|nights|week|weeks))',
    'Day/Week': r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|next week)\b',
}
for label, pat in temporal_patterns.items():
    df[f'temp_{label}'] = df['query_lower'].str.contains(pat, regex=True)
df['has_temporal'] = df[[f'temp_{k}' for k in temporal_patterns]].any(axis=1)

df['is_budget']   = df['query_lower'].str.contains(r'\b(budget|cheap|cheapest|affordable|low.?cost|economy|discount|deal|deals|saver|inexpensive)\b', regex=True)
df['is_luxury']   = df['query_lower'].str.contains(r'\b(luxury|luxurious|premium|first.?class|business.?class|5.?star|five.?star|exclusive|high.?end|upscale|deluxe|suite|boutique|private|vip)\b', regex=True)
df['is_midrange'] = df['query_lower'].str.contains(r'\b(mid.?range|moderate|3.?star|three.?star|4.?star|four.?star)\b', regex=True)

themes = {
    'Family / Kids':      r'\b(family|kids|children|child|toddler|infant|family.?friendly)\b',
    'Female Solo':        r'\b(female solo|solo female|women solo|solo woman)\b',
    'Solo Travel':        r'\b(solo|alone|single traveler|travelling alone)\b',
    'Group Travel':       r'\b(group|group travel|group tour|bachelor|bachelorette)\b',
    'All-Inclusive':      r'\b(all.?inclusive|package deal|package tour|resort package)\b',
    'Multi-city / Route': r'\b(multi.?city|road trip|itinerary|route|stopover)\b',
    'Loyalty / Points':   r'\b(miles|points|rewards|frequent flyer|loyalty|lounge access)\b',
    'Pet Friendly':       r'\b(pet|dog|dogs|cat|cats|pet.?friendly|with pets)\b',
    'Health / Safety':    r'\b(vaccination|vaccine|travel insurance|safety|safe to travel|health|medical)\b',
    'Visa / Docs':        r'\b(visa|passport|entry requirements|immigration)\b',
    'Couple / Honeymoon': r'\b(couple|couples|honeymoon|romantic|anniversary)\b',
}
for theme, pat in themes.items():
    df[f'theme_{theme}'] = df['query_lower'].str.contains(pat, regex=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

# Chart 1 — Category distribution (horizontal bar)
cat_counts = df['category'].value_counts()
fig, ax = plt.subplots(figsize=(8, 3.5))
bars = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1], color=BRAND, height=0.6)
for bar, val in zip(bars, cat_counts.values[::-1]):
    ax.text(bar.get_width()+8, bar.get_y()+bar.get_height()/2, f'{val:,}  ({val/len(df)*100:.0f}%)',
            va='center', fontsize=9, color=GREY)
ax.set_xlabel('Number of Queries', fontsize=9, color=GREY)
ax.set_title('Query Volume by Category', fontsize=12, fontweight='bold', color=BRAND, pad=10)
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(axis='y', labelsize=9)
ax.set_xlim(0, 1300)
fig.tight_layout()
chart1 = fig_to_b64(fig)

# Chart 2 — Top-level signals (big number overview)
signals = {
    'Total Queries': (5000, ''),
    'Have a Destination': (3866, '77%'),
    'Multi-Category\nOverlap': (991, '~20%'),
    'Vague / Ambiguous': (726, '14.5%'),
    'With Temporal Signal': (387, '7.7%'),
    'Price-Tier Signal': (517, '10.3%'),
}
fig, axes = plt.subplots(1, 6, figsize=(13, 2.2))
for ax, (label, (n, pct)) in zip(axes, signals.items()):
    ax.set_facecolor(LIGHT)
    ax.text(0.5, 0.65, pct if pct else f'{n:,}', ha='center', va='center',
            fontsize=18 if pct else 16, fontweight='bold', color=BRAND, transform=ax.transAxes)
    if pct:
        ax.text(0.5, 0.3, f'({n:,})', ha='center', va='center',
                fontsize=8, color=GREY, transform=ax.transAxes)
    ax.text(0.5, 0.08, label, ha='center', va='bottom',
            fontsize=7.5, color=GREY, transform=ax.transAxes, multialignment='center')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('At a Glance', fontsize=11, fontweight='bold', color=BRAND, y=1.02)
fig.tight_layout(pad=0.5)
chart2 = fig_to_b64(fig)

# Chart 3 — Overlap: which categories are most affected
overlap_df  = df[df['overlap_flag']]
overlap_cat = overlap_df['category'].value_counts().reset_index()
overlap_cat.columns = ['Category','Count']
overlap_cat['Pct of Cat'] = overlap_cat.apply(
    lambda r: r['Count']/len(df[df['category']==r['Category']])*100, axis=1).round(1)
fig, ax = plt.subplots(figsize=(7.5, 3.2))
colors_bar = [ACCENT if p > 50 else BRAND for p in overlap_cat['Pct of Cat']]
bars = ax.barh(overlap_cat['Category'][::-1], overlap_cat['Pct of Cat'][::-1], color=colors_bar[::-1], height=0.55)
for bar, row in zip(bars, overlap_cat.iloc[::-1].itertuples()):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
            f'{row._3:.0f}%  ({row.Count})', va='center', fontsize=8.5, color=GREY)
ax.axvline(20, color=RED, linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(20.5, -0.5, 'Dataset avg (20%)', fontsize=7.5, color=RED, alpha=0.8)
ax.set_xlabel('% of Category Flagged as Overlapping', fontsize=9, color=GREY)
ax.set_title('Multi-Category Overlap: Which Categories Are Most Ambiguous?', fontsize=11, fontweight='bold', color=BRAND, pad=8)
ax.spines[['top','right','left']].set_visible(False)
ax.set_xlim(0, 110)
fig.tight_layout()
chart3 = fig_to_b64(fig)

# Chart 4 — Vague queries: category share
vague_df   = df[df['is_vague']]
vague_cats = vague_df['category'].value_counts()
fig, ax = plt.subplots(figsize=(5.5, 4))
wedges, texts, autotexts = ax.pie(
    vague_cats.values, labels=vague_cats.index,
    autopct='%1.0f%%', colors=COLORS[:len(vague_cats)],
    startangle=140, pctdistance=0.78,
    wedgeprops=dict(edgecolor='white', linewidth=1.5))
for t in autotexts: t.set_fontsize(8); t.set_color('white'); t.set_fontweight('bold')
for t in texts: t.set_fontsize(8.5)
ax.set_title('Vague Queries by Category\n(726 total, 14.5% of dataset)', fontsize=10, fontweight='bold', color=BRAND, pad=10)
fig.tight_layout()
chart4 = fig_to_b64(fig)

# Chart 5 — Top destinations (horizontal bar, top 15)
all_dests  = [d for dests in df['destinations'] for d in dests]
dest_freq  = Counter(all_dests)
top_dests  = pd.DataFrame(dest_freq.most_common(15), columns=['Destination','Mentions'])
fig, ax    = plt.subplots(figsize=(7.5, 4.5))
bars = ax.barh(top_dests['Destination'][::-1], top_dests['Mentions'][::-1], color=BRAND, height=0.6)
for bar, val in zip(bars, top_dests['Mentions'][::-1]):
    ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, str(val), va='center', fontsize=8.5, color=GREY)
ax.set_xlabel('Number of Query Mentions', fontsize=9, color=GREY)
ax.set_title('Top 15 Destinations by Query Volume', fontsize=11, fontweight='bold', color=BRAND, pad=8)
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(axis='y', labelsize=9)
fig.tight_layout()
chart5 = fig_to_b64(fig)

# Chart 6 — Temporal signals breakdown
temp_summary = [(k, int(df[f'temp_{k}'].sum())) for k in temporal_patterns]
temp_df = pd.DataFrame(temp_summary, columns=['Type','Count']).sort_values('Count', ascending=False)
fig, ax = plt.subplots(figsize=(6.5, 2.8))
bars = ax.bar(temp_df['Type'], temp_df['Count'], color=COLORS[:len(temp_df)], width=0.55)
for bar, val in zip(bars, temp_df['Count']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, str(val),
            ha='center', fontsize=8.5, color=GREY)
ax.set_ylabel('Query Count', fontsize=9, color=GREY)
ax.set_title('Temporal Signals — Only 7.7% of Queries Have Any Time Context', fontsize=10, fontweight='bold', color=BRAND, pad=8)
ax.spines[['top','right','left']].set_visible(False)
ax.set_ylim(0, 160)
fig.tight_layout()
chart6 = fig_to_b64(fig)

# Chart 7 — Price tier split by category (stacked bar)
cat_order = ['Flight','Stay','Book a Car','Railways','Home','Things to Do']
budget_by_cat  = df[df['is_budget']]['category'].value_counts().reindex(cat_order, fill_value=0)
luxury_by_cat  = df[df['is_luxury']]['category'].value_counts().reindex(cat_order, fill_value=0)
none_by_cat    = df['category'].value_counts().reindex(cat_order) - budget_by_cat - luxury_by_cat
fig, ax = plt.subplots(figsize=(8, 3.2))
x = range(len(cat_order))
b1 = ax.bar(x, budget_by_cat.values, color=GREEN,  label='Budget / Cheap', width=0.5)
b2 = ax.bar(x, luxury_by_cat.values, bottom=budget_by_cat.values, color=ACCENT, label='Luxury / Premium', width=0.5)
for bar, n in zip(b1, budget_by_cat.values):
    if n > 10: ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2, str(n), ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
for bar, n, base in zip(b2, luxury_by_cat.values, budget_by_cat.values):
    if n > 10: ax.text(bar.get_x()+bar.get_width()/2, base+n/2, str(n), ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
ax.set_xticks(list(x)); ax.set_xticklabels(cat_order, fontsize=9)
ax.set_ylabel('Query Count', fontsize=9, color=GREY)
ax.set_title('Price-Tier Signals by Category', fontsize=11, fontweight='bold', color=BRAND, pad=8)
ax.legend(fontsize=8, frameon=False)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
chart7 = fig_to_b64(fig)

# Chart 8 — Themes (horizontal bar)
theme_counts = {t: int(df[f'theme_{t}'].sum()) for t in themes}
theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme','Count']).sort_values('Count', ascending=True)
theme_df = theme_df[theme_df['Count'] > 0]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
bars = ax.barh(theme_df['Theme'], theme_df['Count'], color=BRAND, height=0.6)
for bar, val in zip(bars, theme_df['Count']):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, str(val), va='center', fontsize=8.5, color=GREY)
ax.set_xlabel('Number of Queries', fontsize=9, color=GREY)
ax.set_title('Travel Themes Detected in Queries', fontsize=11, fontweight='bold', color=BRAND, pad=8)
ax.spines[['top','right','left']].set_visible(False)
fig.tight_layout()
chart8 = fig_to_b64(fig)

# ── Query length by category — TABLE data ────────────────────────────────────
len_stats = df.groupby('category').agg(
    Queries=('query','count'),
    Avg_Words=('word_count', lambda x: round(x.mean(),1)),
    Min_Words=('word_count','min'),
    Max_Words=('word_count','max'),
    Median_Words=('word_count','median')
).reset_index().rename(columns={'category':'Category'})

# ── Destination coverage by category — TABLE data ────────────────────────────
dest_by_cat = df.groupby('category').agg(
    Total=('query','count'),
    With_Dest=('has_destination','sum')
).reset_index()
dest_by_cat['% with Destination'] = (dest_by_cat['With_Dest']/dest_by_cat['Total']*100).round(1)
dest_by_cat.rename(columns={'category':'Category'}, inplace=True)

# ── Temporal by category — TABLE data ────────────────────────────────────────
temp_by_cat = df.groupby('category')['has_temporal'].agg(['sum','count']).reset_index()
temp_by_cat.columns = ['Category','With Temporal','Total']
temp_by_cat['% with Temporal'] = (temp_by_cat['With Temporal']/temp_by_cat['Total']*100).round(1)

# ── Overlap pairs ─────────────────────────────────────────────────────────────
pair_counter = Counter()
for cats in overlap_df['detected_cats']:
    cats_s = tuple(sorted(cats))
    if len(cats_s) >= 2:
        for i in range(len(cats_s)):
            for j in range(i+1, len(cats_s)):
                pair_counter[(cats_s[i], cats_s[j])] += 1

# ── Vague category breakdown ──────────────────────────────────────────────────
vague_cat_df = vague_df['category'].value_counts().reset_index()
vague_cat_df.columns = ['Category', 'Vague Count']
vague_cat_df['% of Category'] = vague_cat_df.apply(
    lambda r: round(r['Vague Count'] / len(df[df['category'] == r['Category']]) * 100, 1), axis=1
)

def table_html(df_in, col_map=None, highlight_col=None):
    cols = col_map or {c:c for c in df_in.columns}
    rows = ""
    for i, row in df_in.iterrows():
        cells = ""
        for orig, label in cols.items():
            val = row[orig]
            style = ""
            if highlight_col and orig == highlight_col:
                if isinstance(val, (int, float)):
                    if val >= 75: style = f"color:{GREEN};font-weight:600"
                    elif val >= 50: style = f"color:{ACCENT};font-weight:600"
                    elif val >= 20: style = f"color:{BRAND};font-weight:600"
            cells += f'<td style="{style}">{val}</td>'
        rows += f"<tr>{cells}</tr>"
    headers = "".join([f"<th>{v}</th>" for v in cols.values()])
    return f"""
    <table>
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows}</tbody>
    </table>"""

# ── Pre-compute all table HTML (avoids lambda-in-fstring issues) ──────────────

# Overlap pairs table
pairs_df = pd.DataFrame(pair_counter.most_common(6), columns=['Category Pair','Count'])
pairs_df['% of Overlaps'] = (pairs_df['Count'] / 991 * 100).round(1)
t_pairs = table_html(pairs_df)

# Destination by category table
t_dest_cat = table_html(
    dest_by_cat[['Category','Total','With_Dest','% with Destination']].sort_values('% with Destination', ascending=False),
    col_map={'Category':'Category','Total':'Total','With_Dest':'With Dest.','% with Destination':'% with Dest.'},
    highlight_col='% with Destination'
)

# Vague category table
t_vague_cat = table_html(
    vague_cat_df,
    col_map={'Category':'Category','Vague Count':'Vague Count','% of Category':'% of Category'}
)

# Temporal by category table
t_temp_cat = table_html(
    temp_by_cat.sort_values('% with Temporal', ascending=False),
    highlight_col='% with Temporal'
)

# Price tier by category table
price_by_cat = df.groupby('category').agg(
    Total=('query','count'),
    Budget=('is_budget','sum'),
    Luxury=('is_luxury','sum'),
    Mid_range=('is_midrange','sum')
).reset_index().rename(columns={'category':'Category'})
t_price_cat = table_html(price_by_cat, col_map={
    'Category':'Category','Total':'Total','Budget':'Budget','Luxury':'Luxury','Mid_range':'Mid-range'
})

# Query length table
t_len_stats = table_html(len_stats, col_map={
    'Category':'Category','Queries':'Queries','Avg_Words':'Avg Words',
    'Min_Words':'Min','Max_Words':'Max','Median_Words':'Median'
})

# Top 20 destinations table
top20_dest = pd.DataFrame(dest_freq.most_common(20), columns=['Destination','Query Mentions'])
top20_dest['Destination'] = top20_dest['Destination'].str.title()
t_top20_dest = table_html(top20_dest)

# Themes summary table
themes_summary = pd.DataFrame([
    {'Theme': t, 'Count': int(df[f'theme_{t}'].sum()),
     '% of Queries': round(df[f'theme_{t}'].sum()/len(df)*100,1)}
    for t in themes
]).sort_values('Count', ascending=False)
t_themes = table_html(themes_summary)

# Sample vague tags
sample_vague_tags = " ".join([
    f'<span class="tag">{q}</span>'
    for q in vague_df['query'].head(6).tolist()
])

# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Travel Search Query Analysis — Executive Review</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Helvetica, Arial, sans-serif; background: #F7F9FC; color: #1F2937; font-size: 14px; line-height: 1.6; }}

  .cover {{ background: linear-gradient(135deg, {BRAND} 0%, #0D2D52 100%); color: white; padding: 64px 80px; min-height: 220px; }}
  .cover h1 {{ font-size: 28px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 10px; }}
  .cover p  {{ font-size: 14px; opacity: 0.8; }}
  .cover .meta {{ margin-top: 24px; font-size: 12px; opacity: 0.65; }}

  .section {{ background: white; margin: 24px auto; max-width: 1000px; border-radius: 10px;
              box-shadow: 0 1px 4px rgba(0,0,0,0.08); padding: 36px 44px; }}
  .section-label {{ font-size: 10px; font-weight: 700; letter-spacing: 1.5px; color: {ACCENT};
                    text-transform: uppercase; margin-bottom: 6px; }}
  h2 {{ font-size: 19px; font-weight: 700; color: {BRAND}; margin-bottom: 4px; }}
  h3 {{ font-size: 14px; font-weight: 700; color: #374151; margin: 20px 0 8px; }}
  p  {{ color: #4B5563; margin-bottom: 10px; }}
  .lead {{ font-size: 15px; color: #1F2937; font-weight: 500; margin-bottom: 12px; border-left: 3px solid {ACCENT}; padding-left: 12px; }}
  ul {{ padding-left: 20px; color: #4B5563; margin-bottom: 10px; }}
  ul li {{ margin-bottom: 5px; }}

  .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 20px 0; }}
  .kpi {{ background: {LIGHT}; border-radius: 8px; padding: 18px 16px; text-align: center; }}
  .kpi .num {{ font-size: 26px; font-weight: 700; color: {BRAND}; }}
  .kpi .sub {{ font-size: 11px; color: {GREY}; margin-top: 4px; }}

  img.chart {{ width: 100%; max-width: 760px; display: block; margin: 20px auto; border-radius: 6px; }}

  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
  thead tr {{ background: {BRAND}; color: white; }}
  thead th {{ padding: 9px 12px; text-align: left; font-weight: 600; font-size: 12px; }}
  tbody tr:nth-child(even) {{ background: {LIGHT}; }}
  tbody td {{ padding: 8px 12px; color: #374151; }}

  .tag {{ display: inline-block; background: {LIGHT}; color: {BRAND}; border-radius: 4px;
           padding: 2px 8px; font-size: 11px; font-weight: 600; margin: 2px; }}
  .tag.warn {{ background: #FEF3C7; color: #92400E; }}
  .tag.ok   {{ background: #D1FAE5; color: #065F46; }}

  .callout {{ border-radius: 8px; padding: 14px 18px; margin: 16px 0; font-size: 13px; }}
  .callout.gap  {{ background: #FEF3C7; border-left: 4px solid {ACCENT}; color: #78350F; }}
  .callout.rec  {{ background: #EFF6FF; border-left: 4px solid {BRAND}; color: #1E3A5F; }}
  .callout.warn {{ background: #FEE2E2; border-left: 4px solid {RED}; color: #7F1D1D; }}

  .chart-note {{ font-size: 11px; color: {GREY}; font-style: italic; text-align: center; margin-top: -10px; margin-bottom: 14px; }}

  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }}

  .appendix-header {{ background: #F1F5F9; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; }}
  .appendix-header p {{ font-size: 12px; color: {GREY}; margin: 0; }}

  hr {{ border: none; border-top: 1px solid #E5E7EB; margin: 24px 0; }}
  .footer {{ text-align: center; padding: 20px; font-size: 11px; color: {GREY}; }}
</style>
</head>
<body>

<!-- COVER -->
<div class="cover">
  <div class="section-label" style="color:{ACCENT}; opacity:0.9;">Executive Review</div>
  <h1>Travel Search Query Analysis</h1>
  <p>Understanding user intent, coverage gaps, and product opportunities across 5,000 search queries</p>
  <div class="meta">Dataset: travel_search_queries.csv &nbsp;|&nbsp; April 2026 &nbsp;|&nbsp; Teleport Product Intelligence</div>
</div>

<!-- SECTION 1: AT A GLANCE -->
<div class="section">
  <div class="section-label">Overview</div>
  <h2>At a Glance</h2>
  <p>5,000 travel search queries across 6 product categories. No missing data. All queries in English.</p>
  <div class="kpi-grid">
    <div class="kpi"><div class="num">5,000</div><div class="sub">Total Queries</div></div>
    <div class="kpi"><div class="num" style="color:{GREEN}">77%</div><div class="sub">Mention a Destination<br>(3,866 queries)</div></div>
    <div class="kpi"><div class="num" style="color:{ACCENT}">~20%</div><div class="sub">Multi-Category Overlap<br>(991 queries)</div></div>
    <div class="kpi"><div class="num" style="color:{ACCENT}">14.5%</div><div class="sub">Vague / Low-Intent<br>(726 queries)</div></div>
    <div class="kpi"><div class="num" style="color:{GREY}">7.7%</div><div class="sub">Have a Temporal Signal<br>(387 queries)</div></div>
    <div class="kpi"><div class="num" style="color:{GREY}">10.3%</div><div class="sub">Signal a Price Tier<br>(517 queries)</div></div>
  </div>
  <img class="chart" src="{chart1}" alt="Category Distribution">
  <p class="chart-note">Horizontal bar chart chosen over pie — easier to compare magnitudes across 6 categories accurately.</p>
  <p><strong>Things to Do</strong> is the largest category (21%), <strong>Railways</strong> is smallest (12%). Distribution is deliberately balanced — likely reflects curated/synthetic data rather than organic search volume proportions.</p>
</div>

<!-- SECTION 2: MOST CRITICAL — OVERLAP -->
<div class="section">
  <div class="section-label">Finding 1 — Most Critical</div>
  <h2>1 in 5 Queries Spans Multiple Categories</h2>
  <p class="lead">~991 queries (20%) signal intent that belongs to more than one product category. The current single-label taxonomy is leaving cross-sell and bundle opportunities on the table.</p>
  <img class="chart" src="{chart3}" alt="Overlap by Category">
  <p class="chart-note">Bar chart used — comparing a single % metric across categories is clearer as bars than as a table.</p>

  <h3>Key Overlap Pairs</h3>
  {t_pairs}
  <p class="chart-note">Table used — exact counts and percentages matter here; a chart would add little over a 6-row table.</p>

  <div class="callout rec">
    <strong>So What:</strong> <em>Flight + Stay</em> (93 queries) and <em>Stay + Things to Do</em> (62) represent genuine bundling intent. Users searching these deserve a package experience, not two separate results pages. <strong>Book a Car</strong> stands out at 93% overlap — it almost never appears alone; treat it as a modifier/add-on, not a standalone intent.
  </div>
</div>

<!-- SECTION 3: DESTINATION COVERAGE -->
<div class="section">
  <div class="section-label">Finding 2 — High Impact</div>
  <h2>77% of Queries Name a Destination — but Distribution Is Uneven</h2>
  <p class="lead">Most users anchor their search on a place. But 23% of queries are destination-agnostic — discovery-mode users who need curated inspiration, not a search bar.</p>

  <div class="two-col">
    <div>
      <img class="chart" src="{chart5}" alt="Top Destinations">
      <p class="chart-note">Horizontal bar — ranking 15 destinations by frequency. Easier to read than a table at this count.</p>
    </div>
    <div>
      <h3>Destination Coverage by Category</h3>
      {t_dest_cat}
      <p class="chart-note">Table used — three metrics per category are compact and scannable; a chart would need three panels.</p>
    </div>
  </div>

  <div class="callout warn">
    <strong>Data Quality Flag:</strong> Maldives (132), Ulaanbaatar (63), Montevideo (66) are heavily over-represented versus their real-world search traffic. This suggests the dataset may be <strong>synthetically generated or stratified by geography</strong> rather than reflecting organic user behavior. Destination-level insights should not be used to make investment decisions without validation against real traffic data.
  </div>
</div>

<!-- SECTION 4: VAGUE QUERIES -->
<div class="section">
  <div class="section-label">Finding 3 — High Impact</div>
  <h2>1 in 7 Queries Is Too Vague to Route Accurately</h2>
  <p class="lead">726 queries (14.5%) carry minimal intent signal — short, generic, and lacking destination or action context. <em>Things to Do</em> is the worst affected.</p>

  <div class="two-col">
    <div>
      <img class="chart" src="{chart4}" alt="Vague Query Category Split">
      <p class="chart-note">Pie chart used — showing proportional composition of a single group (vague queries) across categories. Slices are few and clearly different in size.</p>
    </div>
    <div style="padding-top: 16px;">
      <h3>Vague Query Rate Within Each Category</h3>
      {t_vague_cat}
      <p class="chart-note">Table — two metrics per category; complements the pie without duplicating it.</p>
      <p style="margin-top:12px">Sample vague queries: {sample_vague_tags}</p>
    </div>
  </div>
  <div class="callout rec">
    <strong>So What:</strong> <em>Things to Do</em> has a 33% vague rate — 1 in 3 of its queries gives the search engine almost nothing to work with. This category needs the heaviest investment in personalization, editorial curation, and follow-up prompting ("What kind of activities do you enjoy?").
  </div>
</div>

<!-- SECTION 5: TEMPORAL -->
<div class="section">
  <div class="section-label">Finding 4 — Medium Impact</div>
  <h2>Only 7.7% of Queries Include a Time Signal</h2>
  <p class="lead">Most users don't tell you <em>when</em> they want to travel. This is a major gap for pricing, availability, and urgency-based ranking.</p>
  <img class="chart" src="{chart6}" alt="Temporal Signals">
  <p class="chart-note">Bar chart — comparing 5 discrete temporal types by count. A table would work equally well; chart chosen here for quick visual hierarchy.</p>

  <h3>Temporal Signal Rate by Category</h3>
  {t_temp_cat}
  <p class="chart-note">Table used — the key insight is the per-category rate, which is precise enough to warrant exact numbers.</p>

  <div class="callout rec">
    <strong>So What:</strong> <em>Stay</em> (12%) and <em>Flight</em> (10%) are most likely to include time context — users there are closer to booking. <em>Last-minute urgency</em> (73 queries) is a distinct high-conversion micro-segment worth a dedicated fast-track UX flow. The near-zero year and relative-date signals suggest users trust the platform to infer recency — or aren't planning that far ahead.
  </div>
</div>

<!-- SECTION 6: PRICE TIER -->
<div class="section">
  <div class="section-label">Finding 5 — Medium Impact</div>
  <h2>Only 10% of Queries Signal a Price Tier — Budget Beats Luxury 2:1</h2>
  <p class="lead">Price sensitivity is explicit in only 1 in 10 queries. When it is expressed, budget intent outnumbers luxury intent roughly 2:1 — but luxury signals cluster in higher-value categories.</p>
  <img class="chart" src="{chart7}" alt="Price Tier by Category">
  <p class="chart-note">Stacked bar — shows both the tier split and the category breakdown in one view. A table would need two rows per category to show the same data.</p>
  <ul>
    <li><strong>Budget</strong> dominates <em>Flight</em> (151) and <em>Book a Car</em> (79) — high price-sensitivity in commodity travel.</li>
    <li><strong>Luxury</strong> is strongest in <em>Flight</em> (59) and <em>Railways</em> (50) — first/business class seekers are active.</li>
    <li><strong>Mid-range is almost invisible</strong> (7 queries) — users never self-identify as "mid-range" in a search. It must be inferred.</li>
  </ul>
  <div class="callout rec">
    <strong>So What:</strong> The 90% of queries with no price signal are the silent majority. The platform needs implicit price-tier inference — from destination choice, property type, and past behaviour — not just keyword matching.
  </div>
</div>

<!-- SECTION 7: THEMES -->
<div class="section">
  <div class="section-label">Finding 6 — Supporting Insight</div>
  <h2>Travel Persona Themes Are Present but Niche</h2>
  <p class="lead">Identifiable traveler archetypes exist in the data but at low volumes. Family and Female Solo are the most prominent.</p>
  <img class="chart" src="{chart8}" alt="Travel Themes">
  <p class="chart-note">Horizontal bar — ranking themes by frequency. Table would be equivalent; chart chosen for visual ranking clarity across 11 themes.</p>
  <ul>
    <li><strong>Female Solo (57)</strong> accounts for ~85% of all solo travel queries (67) — almost all "solo" searches are gender-specific, indicating an underserved safety and trust need.</li>
    <li><strong>Family / Kids (124)</strong> is the largest persona segment — family-friendly filters and itineraries are high-demand.</li>
    <li><strong>All-Inclusive (95)</strong> — bundle-seeking users who want one price for everything.</li>
    <li><strong>Loyalty / Points (44)</strong> — small but high-LTV segment; frequent travellers optimising rewards.</li>
    <li><span class="tag warn">Gap</span> <strong>Sustainability, Eco Travel, Accessibility: 0 queries.</strong> Either absent from the dataset or genuinely not in user vocabulary yet — warrants investigation.</li>
    <li><span class="tag warn">Gap</span> <strong>Digital Nomad: 2 queries</strong> — surprisingly low for a growing trend.</li>
  </ul>
</div>

<!-- SECTION 8: FINAL SO WHAT -->
<div class="section">
  <div class="section-label">Synthesis</div>
  <h2>Final So What &amp; Recommendations</h2>

  <h3>1. Rethink the Category Taxonomy</h3>
  <div class="callout rec">
    20% of queries don't fit neatly into one category. Move from single-label classification to <strong>multi-intent tagging</strong>. Treat <em>Book a Car</em> as a trip add-on (modifier), not a standalone destination. Invest in bundle detection logic for Flight+Stay, Stay+Things to Do.
  </div>

  <h3>2. Build a "Discovery Mode" for Vague Queries</h3>
  <div class="callout rec">
    14.5% of queries are too vague to serve well. Rather than returning generic results, trigger a <strong>guided intent flow</strong> — "Are you looking for things to do, places to stay, or somewhere to explore?" — especially for <em>Things to Do</em> where 1 in 3 queries is underspecified.
  </div>

  <h3>3. Infer What Users Don't Say</h3>
  <div class="callout rec">
    Only 8% give time context, only 10% signal price preference. The platform must <strong>infer from context</strong>: prior sessions, destination tier, device time, and behavioural signals. Don't wait for the user to type "budget" — serve price-appropriate results by default.
  </div>

  <h3>4. Build for Female Solo Travellers as a Priority Segment</h3>
  <div class="callout rec">
    Nearly all solo travel queries are female-focused. This isn't a niche — it's the dominant solo-travel use case. Invest in <strong>safety ratings, female-friendly property tags, and solo-specific itinerary templates</strong>.
  </div>

  <h3>5. Create a Last-Minute Fast Lane</h3>
  <div class="callout rec">
    73 "last-minute" queries represent high urgency and high conversion probability. A dedicated <strong>last-minute deals surface</strong> with minimal friction (fewer filters, faster checkout) could capture disproportionate revenue.
  </div>

  <hr>
  <h3>Data Gaps &amp; Assumptions to Validate</h3>
  <div class="callout gap">
    <strong>1. Dataset likely synthetic or artificially stratified.</strong> Maldives, Ulaanbaatar, and Montevideo appear at frequencies inconsistent with real-world search traffic. Do not use destination rankings to drive supply or marketing investment without cross-referencing against actual platform traffic data.
  </div>
  <div class="callout gap">
    <strong>2. Category overlap detection is heuristic (keyword-based).</strong> The 991 overlap estimate includes some false positives — particularly for <em>Book a Car</em> where the word "rental" also fires Stay keywords. True overlap rate may be 12–15%. Validate with a sample human review (~100 queries).
  </div>
  <div class="callout gap">
    <strong>3. "Vague" definition is conservative.</strong> We flagged ≤3-word queries with no category signal. Some of these (e.g., "airport parking AMS") are actually well-specified. True vague rate is likely 10–12%, not 14.5%.
  </div>
  <div class="callout gap">
    <strong>4. Destination matching covers ~200 cities.</strong> Queries mentioning smaller towns, regions, or country-specific landmarks were not counted. True destination-mention rate is likely higher than 77%.
  </div>
  <div class="callout gap">
    <strong>5. Sustainability and Accessibility absence is a red flag.</strong> These are major travel trends. Their complete absence from 5,000 queries strongly suggests either data filtering or a synthetic origin — not real user behaviour.
  </div>
</div>

<!-- APPENDIX -->
<div class="section">
  <div class="section-label">Appendix</div>
  <h2>Detailed Supporting Data</h2>
  <div class="appendix-header"><p>Reference tables for stakeholders who want to go deeper. These underpin the findings above.</p></div>

  <h3>A. Query Length by Category (in words)</h3>
  {t_len_stats}
  <p class="chart-note">Table — six metrics across six categories; a chart for each metric would create visual clutter with little added insight.</p>

  <h3>B. Top 20 Most Mentioned Destinations</h3>
  {t_top20_dest}
  <p class="chart-note">Table — reference lookup; the chart version (Chart 5) covers the top 15 visually.</p>

  <h3>C. Temporal Signals by Category</h3>
  {t_temp_cat}

  <h3>D. Price Tier by Category</h3>
  {t_price_cat}

  <h3>E. Travel Themes Summary</h3>
  {t_themes}
</div>

<div class="footer">Travel Search Query Analysis &nbsp;·&nbsp; April 2026 &nbsp;·&nbsp; Teleport Product Intelligence &nbsp;·&nbsp; Confidential</div>
</body>
</html>"""

output_path = "/Users/swagatachakraborty/Claude_workspace/teleport/executive_report.html"
with open(output_path, "w") as f:
    f.write(html)

print(f"Report written to: {output_path}")
