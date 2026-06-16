import os
import re
import glob
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import chromadb
import pandas as pd
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader


OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
CSV_PATH        = "nccu_restaurants_cleaned.csv"   
PDF_DIR         = "restaurant_pdfs"                
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "nccu_restaurants_pdf"           
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"
TOP_K           = 10
TOP_K_TIME      = 50

client        = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_or_create_collection(name=COLLECTION_NAME)



def build_database_from_pdfs():
    print(f"📂 Documents from {PDF_DIR}/, metadata from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    pdf_files = sorted(glob.glob(f"{PDF_DIR}/*.pdf"))
    print(f"   → {len(pdf_files)} PDFs, {len(df)} CSV rows")

    documents, metadatas, ids = [], [], []
    for path in pdf_files:
        idx = int(os.path.basename(path).split("_")[0])   
        row = df.iloc[idx]

        pages = PyPDFLoader(path).load()
        text = "\n".join(p.page_content for p in pages).replace("\x00", "")  

        documents.append(text)
        metadatas.append({k: (str(v) if pd.notna(v) else "") for k, v in row.to_dict().items()})
        ids.append(str(idx))

    print(f"🔢 Embedding {len(documents)} restaurants with {EMBEDDING_MODEL} ...")
    BATCH = 50
    embeddings = []
    for i in range(0, len(documents), BATCH):
        chunk = documents[i:i + BATCH]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
        embeddings.extend([d.embedding for d in resp.data])
        print(f"   → embedded {min(i + BATCH, len(documents))}/{len(documents)}")

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"✅ Built ChromaDB from PDFs with {collection.count()} restaurants.")


if collection.count() == 0:
    build_database_from_pdfs()
else:
    print(f"✅ Loaded existing ChromaDB with {collection.count()} restaurants.")



DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

TIME_KEYWORDS_RE = re.compile(
    r"""\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)\b
        |\b(?:morning|noon|afternoon|evening|night|midnight|late|early|breakfast|brunch|lunch|dinner|supper)\b
        |\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|weekday|today|tomorrow)\b
        |(?:早上|早餐|中午|午餐|下午|晚上|晚餐|宵夜|半夜|凌晨|深夜|今天|明天|週末|平日)
        |\d{1,2}\s*(?:點|時)
        |(?:星期[一二三四五六日天]|週[一二三四五六日天])""",
    re.IGNORECASE | re.VERBOSE,
)

def query_mentions_time(query: str) -> bool:
    return bool(TIME_KEYWORDS_RE.search(query))


def parse_time_from_query(query: str):
    """Extract target hour (0-23) from query. Returns None if not found."""
    q = query.lower()
    
    m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)', q)
    if m:
        h = int(m.group(1))
        ampm = m.group(3).replace('.', '')
        if ampm == 'pm' and h != 12:
            h += 12
        elif ampm == 'am' and h == 12:
            h = 0
        return h
    if 'midnight' in q or '半夜' in q or '凌晨' in q:
        return 0
    if 'noon' in q or '中午' in q:
        return 12
    
    m = re.search(r'(\d{1,2})\s*(?:點|時)', q)
    if m:
        h = int(m.group(1))
        if ('晚上' in q or '下午' in q or 'pm' in q) and h < 12:
            h += 12
        return h
    return None


def parse_day_from_query(query: str):
    """Return day index 0-6 (Mon=0). None if not found."""
    q = query.lower()
    for i, d in enumerate(DAYS):
        if d in q:
            return i
    cn_days = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
    for char, idx in cn_days.items():
        if f"星期{char}" in q or f"週{char}" in q:
            return idx
    if 'today' in q or '今天' in q:
        return datetime.now(ZoneInfo("Asia/Taipei")).weekday()
    if 'tomorrow' in q or '明天' in q:
        return (datetime.now(ZoneInfo("Asia/Taipei")).weekday() + 1) % 7
    return None


def is_open_at(hours_str: str, day_idx: int, target_hour: int) -> bool:
    """Check if a restaurant is open at target_hour on day_idx based on its hours string."""
    if not hours_str or pd.isna(hours_str):
        return False
    day_name = DAYS[day_idx]
   
    pattern = rf'{day_name}:\s*([^|]+)'
    m = re.search(pattern, hours_str, re.IGNORECASE)
    if not m:
        return False
    today_hours = m.group(1).strip()
    if 'closed' in today_hours.lower():
        return False

    
    range_re = re.compile(
        r'(\d{1,2}):(\d{2})\s*(AM|PM)\s*[–\-]\s*(\d{1,2}):(\d{2})\s*(AM|PM)',
        re.IGNORECASE,
    )
    for match in range_re.finditer(today_hours):
        oh, om, op, ch, cm, cp = match.groups()
        oh, ch = int(oh), int(ch)
        op, cp = op.upper(), cp.upper()
        open_24  = (oh % 12) + (12 if op == 'PM' else 0)
        close_24 = (ch % 12) + (12 if cp == 'PM' else 0)
      
        if close_24 <= open_24:
            if target_hour >= open_24 or target_hour < close_24:
                return True
        else:
            if open_24 <= target_hour < close_24:
                return True
    return False


def get_hours_for_day(hours_str: str, day_idx: int) -> str:
    """Return the hours string for a specific day."""
    if not hours_str:
        return "N/A"
    day_name = DAYS[day_idx]
    m = re.search(rf'{day_name}:\s*([^|]+)', hours_str, re.IGNORECASE)
    return m.group(1).strip() if m else "N/A"


SYSTEM_PROMPT = """You are a friendly restaurant recommender for students and visitors near National Chengchi University (NCCU) in Taipei.

Your job:
- Recommend 2-4 restaurants from the retrieved context that best fit the user's needs (cuisine, budget, group size, vibe, time).
- Be warm, conversational, and concise.

IMPORTANT — Time pre-filtering:
The retrieved restaurants have already been filtered to ONLY include places that are OPEN at the time the user asked about (if they specified one). You do NOT need to re-check opening hours for time eligibility — just trust the list. Focus on choosing the best matches based on the user's other preferences (cuisine, budget, group size, vibe).

If the retrieved list is empty or labelled "No restaurants are open at the requested time", tell the user honestly that nothing matches, and suggest 24-hour convenience stores or a different time.

Output format for each pick:
- Name (numbered)
- Short description matching the user's needs
- 📍 Address
- 🕐 Hours for the relevant day
- 💰 Price range
- ✨ Highlight / signature dish

Respond in English unless the user writes in another language.
"""


def get_current_time_info():
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    return now.strftime("%A, %Y-%m-%d %H:%M (Taipei time)")


CUISINE_EXPANSIONS = {
    "korean":     "Korean cuisine, Korean BBQ, bibimbap, kimchi, tteokbokki",
    "japanese":   "Japanese cuisine, sushi, ramen, izakaya, donburi, tempura",
    "italian":    "Italian cuisine, pasta, pizza, risotto",
    "thai":       "Thai cuisine, pad thai, tom yum, curry",
    "chinese":    "Chinese cuisine, stir-fry, dim sum, hot pot",
    "vietnamese": "Vietnamese cuisine, pho, banh mi, spring rolls",
    "western":    "Western food, burgers, steak, sandwiches",
    "cafe":       "cafe, coffee shop, brunch, light meals, study-friendly",
    "dessert":    "dessert, sweets, ice cream, cake, bubble tea",
    "vegetarian": "vegetarian, vegan, plant-based, salad",
    "snack":      "snack, light food, izakaya, bar food, skewers, small plates",
    "night":      "late night, izakaya, bar, open late, supper",
    "late":       "late night, izakaya, bar, open until late, supper",
    "midnight":   "late night, izakaya, bar, open past midnight, after hours",
    "drink":      "bar, izakaya, pub, drinks, beer, cocktails",
}

def expand_query(query: str) -> str:
    q_lower = query.lower()
    expansions = [v for k, v in CUISINE_EXPANSIONS.items() if k in q_lower]
    if expansions:
        return f"{query}  [related: {'; '.join(expansions)}]"
    return query



def retrieve(query: str, k: int):
    expanded = expand_query(query)
    embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=expanded,
    ).data[0].embedding
    actual_k = min(k, collection.count())
    results = collection.query(
        query_embeddings=[embedding],
        n_results=actual_k,
    )
    
    return results["documents"][0], results["metadatas"][0], results["distances"][0]


def gradio_chat(message, history):
    is_time_query = query_mentions_time(message)
    k = TOP_K_TIME if is_time_query else TOP_K
    retrieved_docs, retrieved_metas, distances = retrieve(message, k=k)

    target_hour = parse_time_from_query(message) if is_time_query else None
    target_day  = parse_day_from_query(message)
    if target_hour is not None and target_day is None:
        target_day = datetime.now(ZoneInfo("Asia/Taipei")).weekday()

    print(f"\n🔍 Query: {message}")
    print(f"🔍 Time query: {is_time_query}  | target_hour={target_hour}  target_day={target_day}")


    pairs = list(zip(retrieved_docs, retrieved_metas))
    if target_hour is not None and target_day is not None:
        pairs = [
            (doc, meta) for doc, meta in zip(retrieved_docs, retrieved_metas)
            if is_open_at(meta.get("opening_hours", ""), target_day, target_hour)
        ]
        print(f"⏰ Time-filtered: {len(pairs)} restaurants open at "
              f"{target_hour}:00 on {DAYS[target_day]} (out of {len(retrieved_docs)} retrieved)")
        for doc, meta in pairs[:10]:
            print(f"   ✅ {meta.get('english_name', doc[:40])}")

    filtered_docs = [doc for doc, meta in pairs]


    if target_hour is not None and target_day is not None:
        if filtered_docs:
            context = (
                f"[Pre-filtered: only restaurants verified OPEN at "
                f"{target_hour}:00 on {DAYS[target_day].capitalize()}]\n\n"
                + "\n\n".join(filtered_docs)
            )
        else:
            context = (
                f"No restaurants are open at {target_hour}:00 on "
                f"{DAYS[target_day].capitalize()}. "
                "The following were considered but all are closed at that time:\n\n"
                + "\n\n".join(retrieved_docs[:5])
            )
    else:
        context = "\n\n".join(filtered_docs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        if isinstance(turn, dict):
            messages.append({"role": turn["role"], "content": turn["content"]})
        else:
            user_msg, bot_msg = turn
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

    messages.append({
        "role": "user",
        "content": (
            f"[Current time] {get_current_time_info()}\n\n"
            f"Restaurant data (for reference):\n{context}\n\n"
            f"User request: {message}"
        ),
    })

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content



demo = gr.ChatInterface(
    fn=gradio_chat,
    title="🍴 NCCU Restaurant Recommender",
    description=(
        "Ask me to recommend restaurants near National Chengchi University! "
        "I can suggest places by cuisine, budget, vibe, group size, or time."
    ),
)

if __name__ == "__main__":
    demo.launch()
