import json
import re


def split_sentences(text):
    # split sentence base on .!?; excep . in Dieu 1., Chuong 1.
    sentences = re.split(r'(?<![0-9])[.!?;]\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    merged_sentences = []
    i = 0
    while i < len(sentences):
        if sentences[i].endswith(':') and i + 1 < len(sentences):
            merged_sentences.append(sentences[i] + ' ' + sentences[i + 1])
            i += 2
        else:
            merged_sentences.append(sentences[i])
            i += 1

    return merged_sentences


def extract_sentences(data, reference_type, reference_id, parent_ids=None):
    sentences = []
    content = data.get('content', '')
    if content:
        for sentence in split_sentences(content):
            reference = {
                'type': reference_type,
                'id': reference_id
            }
            if parent_ids:
                reference.update(parent_ids)
            sentences.append({
                'sentence': sentence,
                'reference': reference
            })
    return sentences


def process_law_structure(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    law_data = data['law']
    all_sentences = []

    for chapter in law_data['chapters']:
        chapter_id = chapter['id']
        if 'sections' in chapter:
            for section in chapter['sections']:
                section_id = section['id']
                for article in section['articles']:
                    article_id = article['id']
                    all_sentences.extend(extract_sentences(article, 'Article', article_id,
                                                           {'chapter_id': chapter_id, 'section_id': section_id}))
                    for clause in article['clauses']:
                        clause_id = clause['id']
                        all_sentences.extend(extract_sentences(clause, 'Clause', clause_id,
                                                               {'chapter_id': chapter_id, 'section_id': section_id,
                                                                'article_id': article_id}))
                        for point in clause.get('points', []):
                            point_id = point['id']
                            all_sentences.extend(extract_sentences(point, 'Point', point_id,
                                                                   {'chapter_id': chapter_id, 'section_id': section_id,
                                                                    'article_id': article_id, 'clause_id': clause_id}))
        else:
            for article in chapter['articles']:
                article_id = article['id']
                all_sentences.extend(extract_sentences(article, 'Article', article_id, {'chapter_id': chapter_id}))
                for clause in article['clauses']:
                    clause_id = clause['id']
                    all_sentences.extend(extract_sentences(clause, 'Clause', clause_id,
                                                           {'chapter_id': chapter_id, 'article_id': article_id}))
                    for point in clause.get('points', []):
                        point_id = point['id']
                        all_sentences.extend(extract_sentences(point, 'Point', point_id,
                                                               {'chapter_id': chapter_id, 'article_id': article_id,
                                                                'clause_id': clause_id}))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_sentences, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu {len(all_sentences)} câu vào file {output_file}")


input_file = 'law_structure_from_txt.json'
output_file = 'sentences.json'
process_law_structure(input_file, output_file)