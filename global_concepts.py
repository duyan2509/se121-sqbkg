import json
from tqdm import tqdm
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from underthesea import word_tokenize
from transformers import AutoModel, AutoTokenizer
import torch

# Load PhoBERT model & tokenizer (chỉ tải 1 lần)
tokenizer_phobert = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
model_phobert = AutoModel.from_pretrained("vinai/phobert-base")
model_phobert.eval()

# Ngưỡng tương đồng
SIM_THRESHOLD = 0.95


def get_phobert_embedding(text: str) -> np.ndarray:
    input_text = word_tokenize(text, format="text")
    encoded_input = tokenizer_phobert(input_text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        output = model_phobert(**encoded_input)

    attention_mask = encoded_input['attention_mask']
    token_embeddings = output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return (sum_embeddings / sum_mask).squeeze().numpy()


def load_local_concepts(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_concepts = []
    for article in data:
        concepts = article.get("concepts", [])
        article_id = article.get("article_id")
        for concept in concepts:
            if "attrs" not in concept or not isinstance(concept["attrs"], list):
                concept["attrs"] = []
            if "keyphrases" not in concept or not isinstance(concept["keyphrases"], list):
                concept["keyphrases"] = []
            concept["reference"] = [article_id]
            all_concepts.append(concept)

    print(f"📄 Đã load {len(all_concepts)} concept.")
    return all_concepts


def compute_embeddings(concepts: List[Dict]) -> List[Dict]:
    print("📘 Đang tính embedding PhoBERT cho name * 2 + meaning + keyphrases...")
    for concept in tqdm(concepts):
        name = concept["name"]
        meaning = concept["meaning"]
        keyphrases = " ".join(concept.get("keyphrases", []))
        text = name * 2 + ": " + meaning + " " + keyphrases
        concept["embedding"] = get_phobert_embedding(text).tolist()
    return concepts


def merge_concepts(concepts: List[Dict], debug: bool = True) -> List[Dict]:
    if debug:
        print("🔍 Đang hợp nhất các concept dựa trên độ tương đồng embedding...")

    merged = []
    visited = set()

    embeddings = np.array([c["embedding"] for c in concepts])
    sim_matrix = cosine_similarity(embeddings)

    for i, concept in enumerate(concepts):
        if i in visited:
            continue

        group = [concept]
        visited.add(i)

        for j in range(i + 1, len(concepts)):
            if j in visited:
                continue

            sim = sim_matrix[i][j]
            if sim >= SIM_THRESHOLD:
                group.append(concepts[j])
                visited.add(j)

        if debug and len(group) > 1:
            names = [c["name"] for c in group]
            print(f"🔗 Hợp nhất {len(group)} concept tại vị trí {i}: {names}")

        merged_concept = merge_group(group)
        merged.append(merged_concept)

    return merged


def merge_group(group: List[Dict]) -> Dict:
    base = group[0].copy()
    for c in group[1:]:
        base["attrs"] = list(set(base.get("attrs", []) + c.get("attrs", [])))
        base["keyphrases"] = list(set(base.get("keyphrases", []) + c.get("keyphrases", [])))
        base["similar"] = list(set(base.get("similar", []) + c.get("similar", [])))
        base["reference"] = list(set(base.get("reference", []) + c.get("reference", [])))

    base.pop("embedding", None)
    return base


def save_global_concepts(concepts: List[Dict], path: str = "global_concepts.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã lưu {len(concepts)} khái niệm hợp nhất vào {path}")


def main():
    concepts = load_local_concepts("local_concepts.json")
    concepts = compute_embeddings(concepts)
    merged_concepts = merge_concepts(concepts)
    save_global_concepts(merged_concepts)


if __name__ == "__main__":
    main()
