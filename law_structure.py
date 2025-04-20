import re
import json

with open("data/processed/Luat-Phong-chong-ma-tuy-2021-445185.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines()]

law = {
    "id": "73/2021/QH14",
    "title": "LUẬT PHÒNG, CHỐNG MA TÚY",
    "date": "30-03-2021",
    "source": "Quốc hội",
    "chapters": []
}

chapter_pattern = re.compile(r"^Chương\s+([IVXLCDM]+)\s*(.*)", re.IGNORECASE)
section_pattern = re.compile(r"^Mục\s+(\d+)\s*(.*)", re.IGNORECASE)
article_pattern = re.compile(r"^Điều\s+(\d+)[\.\:]\s*(.*)", re.IGNORECASE)
clause_pattern = re.compile(r"^(\d+)\.\s+(.*)")
point_pattern = re.compile(r"^([a-z])\)\s+(.*)")

def capitalize_title(text):
    words = text.split()
    for i, word in enumerate(words):
        if re.fullmatch(r'[IVXLCDM]+', word, re.IGNORECASE):
            words[i] = word.upper()
        else:
            words[i] = word.capitalize()
    return ' '.join(words)

i = 0
current_chapter = None
current_section = None
current_article = None
current_clause = None
current_point = None

while i < len(lines):
    line = lines[i]
    if not line:
        i += 1
        continue

    chapter_match = chapter_pattern.match(line)
    if chapter_match:
        roman = chapter_match.group(1).upper()
        title = chapter_match.group(2).strip()
        full_title = f"Chương {roman} {title}".strip()

        current_chapter = {
            "id": f"chapter_{roman}",
            "name": capitalize_title(full_title),
            "number": roman,
            "articles": []
        }
        law["chapters"].append(current_chapter)
        current_section = None
        current_article = None
        current_clause = None
        current_point = None
        i += 1
        continue

    section_match = section_pattern.match(line)
    if section_match and current_chapter:
        section_number = section_match.group(1)
        section_title = section_match.group(2).strip()
        current_section = f"Mục {section_number} {section_title}".strip()
        current_section = capitalize_title(current_section)
        i += 1
        continue

    article_match = article_pattern.match(line)
    if article_match and current_chapter:
        article_number = article_match.group(1)
        article_title = article_match.group(2).strip()
        current_article = {
            "id": f"article_{article_number}",
            "parent": current_section,
            "name": capitalize_title(article_title),
            "number": int(article_number),
            "content": "",
            "clauses": [],
            "references": []
        }
        current_chapter["articles"].append(current_article)
        current_clause = None
        current_point = None
        i += 1
        continue

    clause_match = clause_pattern.match(line)
    if clause_match and current_article:
        clause_number = clause_match.group(1)
        clause_text = clause_match.group(2).strip()
        current_clause = {
            "id": f"clause_{current_article['number']}_{clause_number}",
            "number": int(clause_number),
            "content": clause_text,
            "points": []
        }
        current_article["clauses"].append(current_clause)
        current_point = None
        i += 1
        continue

    point_match = point_pattern.match(line)
    if point_match and current_clause:
        point_number = point_match.group(1)
        point_text = point_match.group(2).strip()
        current_point = {
            "id": f"point_{current_article['number']}_{current_clause['number']}_{point_number}",
            "number": point_number,
            "content": point_text
        }
        current_clause["points"].append(current_point)
        i += 1
        continue

    reference_match = re.search(r"(?:Điều\s+(\d+)|khoản\s+(\d+)\s+Điều\s+(\d+))", line)
    if reference_match and current_article:
        if reference_match.group(1):  # Điều X
            ref_article = reference_match.group(1)
            if int(ref_article) != current_article["number"]:
                ref_id = f"article_{ref_article}"
                if ref_id not in current_article["references"]:
                    current_article["references"].append(ref_id)
        elif reference_match.group(2) and reference_match.group(3):  # khoản X Điều Y
            ref_article = reference_match.group(3)
            if int(ref_article) != current_article["number"]:
                ref_id = f"article_{ref_article}"
                if ref_id not in current_article["references"]:
                    current_article["references"].append(ref_id)

    if current_point:
        current_point["content"] += " " + line
    elif current_clause:
        current_clause["content"] += " " + line
    elif current_article:
        current_article["content"] += " " + line

    i += 1

with open("law_structure_from_txt.json", "w", encoding="utf-8") as f:
    json.dump({"law": law}, f, ensure_ascii=False, indent=4)

print("converted from TXT to JSON complete.")
