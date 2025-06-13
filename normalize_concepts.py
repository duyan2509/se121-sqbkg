import json
import re
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


def normalize_concepts(text, raw_concepts):
    prompt = f"""
    Bạn là một chuyên gia pháp luật. Nhiệm vụ của bạn là chuẩn hóa danh sách keyphrase pháp lý.

    Dưới đây là:
    - Văn bản luật (một điều)
    - Danh sách keyphrase ứng viên từ hệ thống phân tích cú pháp

    Yêu cầu:
    1. Chuẩn hóa lại danh sách keyphrase thành các cụm danh từ pháp lý rõ nghĩa.
    2. Viết theo định dạng bình thường, không viết hoa.
    3. Loại bỏ những cụm không mang ý nghĩa pháp lý rõ ràng.
    4. Giữ lại và bổ sung các thuật ngữ đơn lẻ quan trọng thể hiện chủ thể pháp luật, cơ quan nhà nước, tổ chức, ví dụ như "chính phủ", "bộ", "tòa án", "cơ quan".
    5. Bổ sung keyphrase quan trọng nếu bị thiếu, dựa trên nội dung văn bản.

    Trả về kết quả duy nhất là một danh sách JSON Python các chuỗi.

    ### Văn bản:
    {text}

    ### Keyphrase ứng viên:
    {raw_concepts}
    """

    print(prompt)
    response = llm.invoke(prompt)
    response_text = response.content
    cleaned = re.sub(r"^```json\s*|\s*```$", "", response_text.strip())
    print(cleaned)
    try:
        return json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"Lỗi xử lý phản hồi: {str(e)}\nNội dung trả về: {response_text}")


def normalize_all_concepts(input_path: str, output_path: str = "normalized_concepts.json", rpm: int = 15):
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    results = []
    timestamps = []

    for i, article in enumerate(articles):
        article_id = article["article_id"]
        article_number = article.get("article_number", "unknown")
        article_text = article["text"]
        raw_concepts = [c["name"] for c in article.get("concepts", [])]

        if not raw_concepts:
            continue

        print(f"🔍 Chuẩn hóa concepts từ Điều {article_number} ({i + 1}/{len(articles)})")

        try:
            normalized = normalize_concepts(article_text, raw_concepts)

            result = {
                "article_id": article_id,
                "article_number": article_number,
                "normalized_concepts": normalized
            }
        except Exception as e:
            result = {
                "article_id": article_id,
                "article_number": article_number,
                "error": str(e)
            }

        results.append(result)

        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, ensure_ascii=False, indent=2)

        # Rate limiting
        timestamps = [t for t in timestamps if time.time() - t < 60]
        timestamps.append(time.time())
        if len(timestamps) >= rpm:
            wait = 60 - (time.time() - timestamps[0])
            print(f"⏳ Đạt giới hạn API, chờ {wait:.2f} giây...")
            time.sleep(wait)

    print(f"✅ Đã xử lý {len(results)} điều. Kết quả lưu tại: {output_path}")
    return results


if __name__ == "__main__":
    normalize_all_concepts("article_concepts_vncore.json", "normalized_concepts.json", rpm=15)
