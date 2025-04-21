import json
import re
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import ArticleConcepts, Concept

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


def chunk_legal_document(text: str) -> list:
    """
    Chia văn bản pháp luật thành các chunk theo từng điều.

    Args:
        text: Văn bản pháp luật

    Returns:
        Danh sách các chunk, mỗi chunk là một điều
    """
    # Mẫu regex để phát hiện điều (có thể điều chỉnh dựa trên định dạng văn bản)
    article_pattern = r"Điều\s+\d+\.?(.+?)(?=Điều\s+\d+|$)"

    # Tìm tất cả các điều trong văn bản
    articles = re.findall(article_pattern, text, re.DOTALL)

    # Nếu không tìm thấy điều nào, thử phát hiện theo cách khác
    if not articles:
        # Thử tìm theo định dạng "Điều X. [Tiêu đề]"
        article_pattern = r"(Điều\s+\d+\.?[^\n]+(?:\n(?!Điều\s+\d+).+)*)"
        articles = re.findall(article_pattern, text, re.DOTALL)

    # Nếu vẫn không tìm thấy, chia văn bản thành các đoạn
    if not articles:
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

    # Thêm tiêu đề "Điều X" vào nội dung
    full_articles = []
    article_pattern = r"(Điều\s+\d+\.?)(.+?)(?=Điều\s+\d+|$)"
    matches = re.findall(article_pattern, text, re.DOTALL)

    for match in matches:
        article_title = match[0].strip()
        article_content = match[1].strip()
        full_articles.append(f"{article_title} {article_content}")

    return full_articles


def extract_article_number(article_text: str) -> str:
    """
    Trích xuất số điều từ text của điều

    Args:
        article_text: Văn bản của điều

    Returns:
        Số điều dạng chuỗi (ví dụ: "1", "2", v.v.)
    """
    match = re.search(r"Điều\s+(\d+)", article_text)
    if match:
        return match.group(1)
    return "unknown"


def extract_concepts(text: str) -> ArticleConcepts:
    """
    Extract legal concepts from article text using the Gemini model.

    Args:
        text: The article text to analyze

    Returns:
        concepts in article object containing concepts, article id
    """
    prompt = f"""
    Bạn là một chuyên gia pháp luật có khả năng phân tích văn bản luật và trích xuất các khái niệm pháp lý quan trọng.
    
    Hãy đọc đoạn văn bản luật sau và trích xuất các **khái niệm pháp lý (concepts)** có trong đó. Mỗi khái niệm có cấu trúc sau:

    - **name**: tên khái niệm (ngắn gọn, rõ ràng)
    - **meaning**: mô tả ngắn gọn ý nghĩa khái niệm trong ngữ cảnh luật
    - **attrs**: danh sách thuộc tính (các đặc điểm chính hoặc yếu tố liên quan đến khái niệm)
    - **keyphrases**: danh sách cụm từ khóa (được nhắc đến trực tiếp trong văn bản, liên quan đến khái niệm)
    - **similar**: các cách diễn đạt khác nhau của khái niệm, đồng nghĩa hoặc thường dùng

    Hãy trả kết quả dưới định dạng JSON theo mẫu sau:

    {{
      "title": "Điều X. [Tên điều luật]",
      "concepts": [
        {{
          "name": "Tên khái niệm",
          "meaning": "Giải thích ngắn về khái niệm trong bối cảnh pháp luật",
          "attrs": ["thuộc tính 1", "thuộc tính 2", "..."],
          "keyphrases": ["cụm từ khóa 1", "cụm từ khóa 2", "..."],
          "similar": ["cách gọi khác", "cụm từ tương đương", "..."]
        }},
        ...
      ]
    }}

    văn bản:{text}
    """


    response = llm.invoke(prompt)
    response_text = response.content
    print("===== RESPONSE TEXT =====")
    cleaned = re.sub(r"^```json\s*|\s*```$", "", response_text.strip())
    print(cleaned)
    print("=========================")
    try:
        data = json.loads(cleaned)
        return ArticleConcepts(**data)
    except Exception as e:
        raise ValueError(f"Lỗi khi xử lý phản hồi: {str(e)}")

class RateLimiter:
    """Lớp kiểm soát tốc độ gửi request để không vượt quá giới hạn của API"""

    def __init__(self, requests_per_minute=15):
        self.requests_per_minute = requests_per_minute
        self.time_frame = 60  # 60 seconds in a minute
        self.request_times = []

    def wait_if_needed(self):
        """Chờ nếu cần thiết để đảm bảo không vượt quá giới hạn tốc độ"""
        current_time = time.time()

        self.request_times = [t for t in self.request_times if current_time - t < self.time_frame]

        if len(self.request_times) >= self.requests_per_minute:
            # Tính thời gian cần chờ
            oldest_timestamp = self.request_times[0]
            wait_time = self.time_frame - (current_time - oldest_timestamp)

            if wait_time > 0:
                print(f"Đạt giới hạn API ({self.requests_per_minute} request/phút). Chờ {wait_time:.2f} giây...")
                time.sleep(wait_time)

        self.request_times.append(time.time())


def process_legal_document(document_path: str, output_path: str = "local_concepts.json", max_articles: int = None,
                           requests_per_minute: int = 15):
    """
    Process a legal document by chunking it into articles and extracting triplets

    Args:
        document_path: Path to the legal document file
        output_path: Path where results will be saved
        max_articles: Maximum number of articles to process (optional)
        requests_per_minute: Maximum API requests per minute
    """
    rate_limiter = RateLimiter(requests_per_minute)

    with open(document_path, "r", encoding="utf-8") as f:
        legal_text = f.read()

    articles = chunk_legal_document(legal_text)

    if max_articles and max_articles < len(articles):
        articles = articles[:max_articles]

    results = []

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("[\n")

        for i, article_text in enumerate(articles):
            # Trích xuất số điều
            article_number = extract_article_number(article_text)
            article_id = f"article_{article_number}"

            try:
                print(f"Đang xử lý điều {article_number} ({i + 1}/{len(articles)})")

                rate_limiter.wait_if_needed()

                extraction = extract_concepts(article_text)

                result = {
                    "article_id": article_id,
                    "article_number": article_number,
                    "text": article_text,
                    "concepts": [concept.model_dump() for concept in extraction.concepts],
                }
            except Exception as e:
                result = {
                    "article_id": article_id,
                    "article_number": article_number,
                    "text": article_text,
                    "error": str(e)
                }

            results.append(result)

            f.write(json.dumps(result, ensure_ascii=False, indent=2))

            if i < len(articles) - 1:
                f.write(",\n")
            else:
                f.write("\n")

            f.flush()

        f.write("]\n")

    print(f"Đã xử lý {len(results)} điều. Kết quả được lưu tại: {output_path}")
    return results


def main():
    document_path = "data/processed/Luat-Phong-chong-ma-tuy-2021-445185.txt"
    output_path = "local_concepts.json"
    max_articles = None
    requests_per_minute = 15

    process_legal_document(document_path, output_path, max_articles, requests_per_minute)


if __name__ == "__main__":
    main()