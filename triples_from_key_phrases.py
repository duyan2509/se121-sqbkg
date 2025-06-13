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


def getRelations(article_id):
    with open("sentence_with_cr_vncore.json", "r", encoding="utf-8") as f:
        sentence_with_cr = json.load(f)

    relations = set()
    for entry in sentence_with_cr:
        if (entry.get("reference", {}).get("id") == article_id
                or entry.get("reference", {}).get("article_id") == article_id):
            relations.update(entry.get("relations", []))

    return list(relations)


def extract_structured_relations(text, concepts, relations):
    prompt = f"""
    Bạn là một chuyên gia pháp luật có nhiệm vụ phân tích văn bản luật và trích xuất các **quan hệ pháp lý** giữa các khái niệm đã cho.

    Bạn được cung cấp:
    - Văn bản luật
    - Danh sách các khái niệm (concepts) ứng viên đã xác định từ trước  

    Hãy trích xuất các quan hệ dưới dạng:

    R = (Name, ConckeyS, ConckeyO)

    Trong đó:
    - Name: tên quan hệ
    - ConckeyS: tên của concept đóng vai trò **chủ thể (S)** — tức là bên **thực hiện**, **ban hành**, hoặc **tác động**, phải nằm trong danh sách khái niệm được xác định từ trước
    - ConckeyO: tên của concept đóng vai trò **đối tượng (O)** — tức là bên hoặc sự việc **bị tác động**, **bị quy định**, hoặc **chịu sự điều chỉnh**, phải nằm trong danh sách khái niệm được xác định từ trước

    **Lưu ý quan trọng**:
    - Nếu câu có nhiều mệnh đề phức tạp, tách thành các câu đơn.
    - Phải lựa chọn đúng vai trò chủ thể và đối tượng dựa trên ngữ nghĩa pháp lý trong câu.
    - Nếu không chắc, hãy ưu tiên bên **tác động** làm chủ thể và bên **bị tác động** làm đối tượng.
    - Không được thêm concept mới hoặc loại bỏ concepts đã cho; 
    - Một concept có thể vừa là ConckeyS trong quan hệ này, vừa là ConckeyO trong quan hệ khác.
    - Các triples có thể nối với nhau để tạo thành chuỗi quan hệ nhiều bước giữa các khái niệm (multi-hop knowledge graph).
    - Trích xuất đầy đủ các triples để mô tả tất cả các liên kết giữa các concepts.
    - Các triples này được trích xuất để tạo đồ thị cho nhiệm vụ hỏi đáp 

    Trả về kết quả dưới dạng JSON sau:

    {{
      "relations": [
        {{
          "Name": "tên quan hệ",
          "ConckeyS": "tên concept chủ thể",
          "ConckeyO": "tên concept đối tượng"
        }},
        ...
      ]
    }}

    Văn bản:
    {text}
    Concepts:
    {[c for c in concepts]}
    """

    print(f"prompt: {prompt}")
    response = llm.invoke(prompt)
    response_text = response.content
    cleaned = re.sub(r"^```json\s*|\s*```$", "", response_text.strip())
    try:
        data = json.loads(cleaned)
        return data.get("relations", [])
    except Exception as e:
        raise ValueError(f"Lỗi xử lý phản hồi: {str(e)}")
def get_article_texts(article_concepts_path: str) -> dict:
    """
    Đọc file article_concepts_vncore.json và trả về dict
    mapping từ article_id sang text (văn bản luật của điều).
    """
    with open(article_concepts_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    article_text_map = {}
    for article in articles:
        article_id = article.get("article_id")
        text = article.get("text", "")
        if article_id:
            article_text_map[article_id] = text

    return article_text_map


def extract_all_structured_relations(local_concepts_path: str, output_path: str = "structured_relations.json", rpm: int = 15):
    with open(local_concepts_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    results = []
    timestamps = []
    article_texts = get_article_texts("article_concepts_vncore.json")
    for i, article in enumerate(articles):
        article_id = article["article_id"]
        article_number = article.get("article_number", "unknown")

        article_text = article_texts.get(article_id, None)
        concepts = article.get("normalized_concepts", [])

        relations = getRelations(article_id)

        if not concepts or not relations:
            continue

        print(f"Đang xử lý quan hệ từ Điều {article_number} ({i + 1}/{len(articles)})")

        try:
            structured_relations = extract_structured_relations(article_text, concepts, relations)

            result = {
                "article_id": article_id,
                "article_number": article_number,
                "relations": structured_relations
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
            print(f"Đạt giới hạn API, chờ {wait:.2f} giây...")
            time.sleep(wait)

    print(f"Đã xử lý {len(results)} điều. Kết quả lưu tại: {output_path}")
    return results


if __name__ == "__main__":
    extract_all_structured_relations("normalized_concepts.json", "structured_relations_vncore.json", rpm=15)
