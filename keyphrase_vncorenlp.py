import json
from typing import Tuple, List

import requests
from py_vncorenlp import VnCoreNLP
from dotenv import load_dotenv
import os

load_dotenv()
vncorenlp_dir = os.getenv("VNCORENLP_DIR")

# Khởi tạo annotator
rdr = VnCoreNLP(
    save_dir=vncorenlp_dir,
    annotators=["wseg", "pos", "ner", "parse"],
    max_heap_size='-Xmx2g'
)

def extract_noun_chunks_from_dependency(dep_data):
    sentence = dep_data[0]
    noun_tags = {'N', 'Np', 'Nc', 'Nu'}

    noun_chunks = []
    current_chunk = []

    for tok in sentence:
        if tok['posTag'] in noun_tags:
            current_chunk.append(tok['wordForm'])
        else:
            if current_chunk:
                noun_chunks.append(" ".join(current_chunk))
                current_chunk = []

    if current_chunk:
        noun_chunks.append(" ".join(current_chunk))

    return list(set(noun_chunks))  # Loại trùng

def extract_noun_phrases_from_dependency(dep_data):
    sentence = dep_data[0]
    tokens = {tok['index']: tok for tok in sentence}
    children_map = {}

    # Xây dựng bản đồ các từ phụ thuộc
    for tok in sentence:
        head = tok['head']
        if head not in children_map:
            children_map[head] = []
        children_map[head].append(tok['index'])

    def collect_full_phrase_indices(index):
        indices = [index]
        children = children_map.get(index, [])
        for child in children:
            rel = tokens[child]['depLabel']
            if rel in ('amod', 'nmod', 'compound', 'det', 'sub', 'vmod'):
                indices += collect_full_phrase_indices(child)
        return indices

    noun_phrases = set()
    visited = set()

    for tok in sentence:
        if tok['posTag'].startswith('N') or tok['nerLabel'] != 'O':
            indices = collect_full_phrase_indices(tok['index'])
            indices = sorted(set(indices))
            phrase_tokens = [tokens[i]['wordForm'] for i in indices]

            phrase_text = " ".join(phrase_tokens)

            if not any(phrase_text in np for np in noun_phrases):
                noun_phrases.add(phrase_text)

            visited.update(indices)



    return sorted(list(noun_phrases), key=lambda x: -len(x))


def extract_main_verbs(dep_data, noun_phrases):
    sentence = dep_data[0]

    np_token_indices = set()
    for np in noun_phrases:
        np_tokens = np.split()
        for i, tok in enumerate(sentence):
            if tok['wordForm'] in np_tokens:
                np_token_indices.add(tok['index'])

    auxiliary_verbs = {'được', 'bị', 'phải', 'đang', 'sẽ', 'đã', 'cần'}

    main_verbs = []
    for tok in sentence:
        if tok['posTag'].startswith('V') and tok['index'] not in np_token_indices:
            if tok['wordForm'].lower() not in auxiliary_verbs:
                main_verbs.append(tok['wordForm'])

    for tok in sentence:
        if tok['depLabel'] == 'root' and tok['index'] not in np_token_indices:
            if tok['posTag'].startswith('V') or tok['posTag'].startswith('C'):
                if tok['wordForm'].lower() not in auxiliary_verbs:
                    if tok['wordForm'] not in main_verbs:
                        main_verbs.insert(0, tok['wordForm'])

    return main_verbs


def clean (sentence):
    sentence = sentence.lower()
    sentence = sentence.replace("phòng, chống","phòng chống")
    return sentence

def process_sentences(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    stopwords = load_stopwords()
    stopwords.remove("người")
    results = []
    for entry in data:
        sentence = clean(entry["sentence"])
        annotated = rdr.annotate_text(sentence)
        noun_phrases = extract_noun_chunks_from_dependency(annotated)
        verb_phrases = extract_main_verbs(annotated, noun_phrases)
        noun_phrases = remove_stopwords(noun_phrases, stopwords)
        verb_phrases = remove_stopwords(verb_phrases, stopwords)
        result = {
            "sentence": sentence,
            "concepts": noun_phrases,
            "relations": verb_phrases,
            "reference": entry.get("reference", {})
        }
        print(result)
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def load_stopwords():
    url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"
    response = requests.get(url)
    stopwords = set()
    if response.status_code == 200:
        for line in response.text.splitlines():
            stopwords.add(line.strip())
    return stopwords

def remove_stopwords(words, stopwords):
    return [word for word in words if word.lower() not in stopwords]


current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir,  "sentences.json")
output_path = os.path.join(current_dir, "sentence_with_cr_vncore.json")

if __name__ == "__main__":
    print(input_path)
    print(output_path)

    process_sentences(input_path, output_path)