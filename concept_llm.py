import json
import re
import time
from collections import defaultdict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load biến môi trường từ .env
load_dotenv()

# Khởi tạo LLM Gemini Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# --- Nhóm các câu theo article_id ---
def group_sentences_by_article(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = defaultdict(list)
    for entry in data:
        article_id = entry.get("reference", {}).get("id") or entry.get("reference", {}).get("article_id")
        if article_id:
            grouped[article_id].append(entry["sentence"])
    return grouped

# --- Prompt để trích xuất danh từ ---
def extract_concepts_from_text(article_id, text):
    prompt = f"""
    Bạn là một công cụ xử lý ngôn ngữ tự nhiên cho tiếng Việt, chuyên dùng để trích xuất khái niệm từ văn bản pháp luật.

    Nhiệm vụ:
    - Phân tích đoạn văn bản dưới đây và **trích xuất tất cả các danh từ hoặc cụm danh từ chính** (gọi là concept), bao gồm: đối tượng pháp lý, hành vi, tổ chức, quy định, sự vật, hiện tượng,...

    Quy tắc:
    - Mỗi concept phải là danh từ hoặc cụm danh từ, được giữ nguyên văn từ văn bản.
    - Trả về dưới dạng danh sách **không trùng lặp**, tất cả đều **viết thường**.
    - Không bao gồm động từ, trạng từ, tính từ không cần thiết nếu không phải là một phần của danh từ/cụm danh từ.

    Ví dụ:
    1. Văn bản: "Chất ma túy là chất gây nghiện, chất hướng thần được quy định trong danh mục chất ma túy do Chính phủ ban hành."
       → concepts: ["chất ma túy", "chất gây nghiện", "chất hướng thần", "danh mục chất ma túy", "chính phủ"]

    2. Văn bản: "Chất gây nghiện là chất kích thích hoặc ức chế thần kinh, dễ gây tình trạng nghiện đối với người sử dụng."
       → concepts: ["chất gây nghiện", "chất kích thích", "ức chế thần kinh", "tình trạng nghiện", "người sử dụng"]

    Bây giờ, hãy trích xuất concepts từ văn bản sau:

    Văn bản:
    {text}

    Trả về kết quả dưới dạng JSON như sau:

    {{
      "concepts": ["..."]
    }}
    """
    print(prompt)
    response = llm.invoke(prompt)
    response_text = response.content
    cleaned = re.sub(r"^```json\s*|\s*```$", "", response_text.strip())
    print(cleaned)
    try:
        data = json.loads(cleaned)
        return data.get("concepts", [])
    except Exception as e:
        raise ValueError(f"Lỗi xử lý phản hồi: {str(e)}")

# --- Hàm chạy toàn bộ tập văn bản ---
def extract_all_concepts(grouped_by_article, output_path="extracted_concepts.json", rpm=15):
    results = []
    timestamps = []

    for i, (article_id, sentences) in enumerate(grouped_by_article.items()):
        print(f"Đang xử lý {article_id} ({i + 1}/{len(grouped_by_article)})")
        text = " ".join(sentences)

        try:
            concepts = extract_concepts_from_text(article_id, text)
            result = {
                "id": article_id,
                "concepts": concepts
            }
        except Exception as e:
            result = {
                "id": article_id,
                "error": str(e)
            }

        results.append(result)

        # Lưu tạm thời sau mỗi lần
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, ensure_ascii=False, indent=2)

        # Giới hạn tốc độ gọi API
        timestamps = [t for t in timestamps if time.time() - t < 60]
        timestamps.append(time.time())
        if len(timestamps) >= rpm:
            wait = 60 - (time.time() - timestamps[0])
            print(f"Đạt giới hạn API, chờ {wait:.2f} giây...")
            time.sleep(wait)

    print(f"\nĐã xử lý {len(results)} article. Kết quả lưu tại: {output_path}")
    return results


if __name__ == "__main__":
    grouped = group_sentences_by_article("sentences.json")
    extract_all_concepts(grouped, output_path="concepts_llm.json", rpm=15)
