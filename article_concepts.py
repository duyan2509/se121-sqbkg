import json
import re


def chunk_legal_document(text: str) -> list:
    article_pattern = r"Điều\s+\d+\.?(.+?)(?=Điều\s+\d+|$)"
    articles = re.findall(article_pattern, text, re.DOTALL)

    if not articles:
        article_pattern = r"(Điều\s+\d+\.?[^\n]+(?:\n(?!Điều\s+\d+).+)*)"
        articles = re.findall(article_pattern, text, re.DOTALL)

    if not articles:
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

    full_articles = []
    article_pattern = r"(Điều\s+\d+\.?)(.+?)(?=Điều\s+\d+|$)"
    matches = re.findall(article_pattern, text, re.DOTALL)

    for match in matches:
        article_title = match[0].strip()
        article_content = match[1].strip()
        full_articles.append(f"{article_title} {article_content}")

    return full_articles


def extract_article_number(article_text: str) -> str:
    match = re.search(r"Điều\s+(\d+)", article_text)
    if match:
        return match.group(1)
    return "unknown"


def getKeyPhrases(article_id) -> list:
    with open("sentence_with_cr_vncore.json", "r", encoding="utf-8") as f:
        sentence_with_cr = json.load(f)

    keyphrases = set()

    for entry in sentence_with_cr:
        if (entry.get("reference", {}).get("id") == article_id
                or entry.get("reference", {}).get("article_id") == article_id):
            keyphrases.update(entry.get("concepts", []))

    return list(keyphrases)


def process_legal_document_without_llm(document_path: str, output_path: str = "local_concepts.json", max_articles: int = None):
    with open(document_path, "r", encoding="utf-8") as f:
        legal_text = f.read()

    articles = chunk_legal_document(legal_text)

    if max_articles and max_articles < len(articles):
        articles = articles[:max_articles]

    results = []

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("[\n")

        for i, article_text in enumerate(articles):
            article_number = extract_article_number(article_text)
            article_id = f"article_{article_number}"
            keyphrases = getKeyPhrases(article_id)

            result = {
                "article_id": article_id,
                "article_number": article_number,
                "text": article_text,
                "concepts": [{"name": kp, "meaning": "Không định nghĩa"} for kp in keyphrases]
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
    output_path = "article_concepts_vncore.json"
    max_articles = None

    process_legal_document_without_llm(document_path, output_path, max_articles)


if __name__ == "__main__":
    main()
