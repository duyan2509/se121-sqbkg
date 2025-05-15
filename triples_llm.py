import json
import re
import time
from collections import defaultdict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env để lấy API key
load_dotenv()

# Khởi tạo Gemini Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# --- Nhóm câu theo article_id ---
def group_sentences_by_article(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = defaultdict(list)
    for entry in data:
        article_id = entry.get("reference", {}).get("id") or entry.get("reference", {}).get("article_id")
        if article_id:
            grouped[article_id].append(entry["sentence"])
    return grouped

# --- Lấy concepts theo id ---
def get_concepts_by_id(data, target_id):
    for item in data:
        if item['id'] == target_id:
            return item.get("concepts", [])
    return []

# --- Prompt trích xuất triples ---
def build_prompt(article_id, sentences, concepts):
    return f"""
Bạn là một công cụ trích xuất thông tin từ văn bản luật. Dưới đây là thông tin của một điều luật, bao gồm danh sách các câu (sentences) và các khái niệm (concepts) được trích xuất từ điều luật đó.

Yêu cầu:
- Dựa vào ngữ cảnh trong các câu, hãy xác định các mối quan hệ (relation) giữa các khái niệm.
- Chỉ tạo các cặp subject (s) và object (o) nếu cả hai đều nằm trong mảng concepts.
- Mỗi bộ ba (triple) cần có cấu trúc: {{ "subject": ..., "relation": ..., "oject": ... }}.
- Trả về danh sách các triples dưới dạng JSON chuẩn.
- Relation phải là các cụm từ trong văn bản
- Ưu tiên tạo bộ ba chủ động, nếu relation đang bị động, đổi vị trí của subject và object, sau đó chuyển relation sang chủ động 

Dữ liệu đầu vào:

{{
  "article_id": "{article_id}",
  "sentences": {json.dumps(sentences, ensure_ascii=False)},
  "concepts": {json.dumps(concepts, ensure_ascii=False)}
}}

Trả về kết quả JSON như sau:

{{ 
  "{article_id}": {{
    "triples": [
      {{
        "subject": "...",
        "relation": "...",
        "object": "..."
      }}
    ]
  }}
}}
"""

# --- Gọi LLM và xử lý output ---
def extract_triples_from_llm(article_id, sentences, concepts):
    prompt = build_prompt(article_id, sentences, concepts)
    response = llm.invoke(prompt)
    content = response.content.strip()
    cleaned = re.sub(r"^```json\s*|\s*```$", "", content)
    try:
        data = json.loads(cleaned)
        return data.get(article_id, {}).get("triples", [])
    except Exception as e:
        raise ValueError(f"Lỗi xử lý LLM output cho {article_id}: {e}\nNội dung:\n{cleaned}")

# --- Chạy toàn bộ ---
def extract_all_triples(sentences_file, concepts_file, output_file="triples_llm.json", rpm=15):
    grouped_sentences = group_sentences_by_article(sentences_file)

    with open(concepts_file, "r", encoding="utf-8") as f:
        concepts_data = json.load(f)

    result = []
    timestamps = []

    for i, (article_id, sents) in enumerate(grouped_sentences.items()):
        print(f"⏳ Đang xử lý {article_id} ({i + 1}/{len(grouped_sentences)})")
        concepts = get_concepts_by_id(concepts_data, article_id)

        try:
            triples = extract_triples_from_llm(article_id, sents, concepts)
            result.append({
                "id": article_id,
                "triples": triples
            })
        except Exception as e:
            print(f"❌ Lỗi ở {article_id}: {e}")
            result.append({
                "id": article_id,
                "error": str(e)
            })

        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(result, out_f, ensure_ascii=False, indent=2)

        timestamps = [t for t in timestamps if time.time() - t < 60]
        timestamps.append(time.time())
        if len(timestamps) >= rpm:
            wait = 60 - (time.time() - timestamps[0])
            print(f"⏳ Đạt giới hạn {rpm} yêu cầu/phút. Đợi {wait:.2f} giây...")
            time.sleep(wait)

    print(f"\n✅ done. extracted {len(result)} . result saved at : {output_file}")
    return result

if __name__ == "__main__":
    extract_all_triples(
        sentences_file="sentences.json",
        concepts_file="concepts_llm.json",
        output_file="triples_llm.json",
        rpm=15
    )
