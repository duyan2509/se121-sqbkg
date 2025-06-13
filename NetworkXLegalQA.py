import json
import os
import re
import numpy as np
from typing import List, Dict, Any
import torch
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from transformers import AutoModel, AutoTokenizer
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pickle
from py_vncorenlp import VnCoreNLP
from networkx.algorithms import isomorphism

load_dotenv()

# Environment variables
vncorenlp_dir = os.getenv("VNCORENLP_DIR")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert = AutoModel.from_pretrained("vinai/phobert-base")

rdrsegmenter = VnCoreNLP(
    save_dir=vncorenlp_dir,
    annotators=["wseg", "pos", "ner", "parse"],
    max_heap_size='-Xmx2g'
)


class NetworkXLegalQA:
    def __init__(self, graph_path='/graphs/legal_graph_with_embeddings.pkl'):
        self.graph = None
        self.embeddings_cache = {}
        self.graph_embeddings = {}
        self.load_graph(graph_path)
        if os.path.exists("embeddings_cache.pkl"):
            with open("embeddings_cache.pkl", "rb") as f:
                self.embeddings_cache = pickle.load(f)
        if not self.graph_embeddings:
            self.create_graph_embeddings()

    def load_graph(self, graph_path: str):
        """Load NetworkX graph from pickle file"""
        try:
            with open(graph_path, 'rb') as f:
                graph_data = pickle.load(f)

            if isinstance(graph_data, dict):
                self.graph = graph_data['graph']
                self.embeddings_cache = graph_data.get('embedding_cache', {})
                for node, data in self.graph.nodes(data=True):
                    if 'embedding' in data:
                        self.graph_embeddings[f"node:{node}"] = data['embedding']
            else:
                self.graph = graph_data

            print(f"Graph loaded with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except FileNotFoundError:
            print(f"Graph file {graph_path} not found. Creating empty graph.")
            self.graph = nx.MultiDiGraph()

    def save_embeddings(self):
        """Save embeddings cache"""
        with open("embeddings_cache.pkl", "wb") as f:
            pickle.dump(self.embeddings_cache, f)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using PhoBERT"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
        with torch.no_grad():
            outputs = phobert(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy().flatten()
        self.embeddings_cache[text] = embedding
        return embedding
    def create_graph_embeddings(self):
        """Create embeddings for all nodes and relationships in graph"""
        print("Creating graph embeddings...")
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'entity':
                if 'embedding' not in data:
                    embedding = self.get_embedding(node)
                    self.graph.nodes[node]['embedding'] = embedding
                    self.graph_embeddings[f"node:{node}"] = embedding
                else:
                    self.graph_embeddings[f"node:{node}"] = data['embedding']

        for source, target, edge_data in self.graph.edges(data=True):
            relation_type = edge_data.get('relation_type', 'RELATES')
            if relation_type != 'CONTAINS':
                rel_text = f"{source} {relation_type} {target}"
                rel_key = f"rel:{source}|{relation_type}|{target}"
                if rel_key not in self.graph_embeddings:
                    self.graph_embeddings[rel_key] = self.get_embedding(rel_text)

        self.save_embeddings()
        print(f"Created embeddings for {len(self.graph_embeddings)} graph elements")

    def extract_noun_chunks(self, text: str) -> List[str]:
        """Extract noun chunks using VnCoreNLP dependency parsing"""
        try:
            annotated = rdrsegmenter.annotate_text(text)
            sentence = annotated[0]
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

            return list(set(noun_chunks))
        except Exception as e:
            print(f"Error in noun chunk extraction: {e}")
            return text.split()

    def load_stopwords(self) -> List[str]:
        """Load Vietnamese stopwords"""
        return [
            'và', 'hoặc', 'là', 'của', 'cho', 'trong', 'từ', 'với',
            'theo', 'về', 'trên', 'dưới', 'cùng', 'tại'
        ]

    def extract_concepts(self, noun_chunks: List[str], question: str = "") -> List[str]:
        """Extract legal concepts using LLM"""
        prompt = f"""
        Bạn là một chuyên gia pháp luật Việt Nam. Nhiệm vụ của bạn là trích xuất các khái niệm pháp lý quan trọng từ các cụm danh từ được cung cấp.

        CÂU HỎI GỐC: {question}
        DANH SÁCH CỤM DANH TỪ: {noun_chunks}

        YÊU CẦU:
        1. Dựa vào ngữ cảnh câu hỏi để xác định khái niệm pháp lý.
        2. Loại bỏ cụm từ không mang tính pháp lý (như thời gian, địa điểm chung).
        3. Chuẩn hóa: viết thường, loại bỏ từ không cần thiết.
        4. Trả về JSON trong thẻ <LEGAL_CONCEPTS>:
        <LEGAL_CONCEPTS>
        ["khái niệm 1", "khái niệm 2", ...]
        </LEGAL_CONCEPTS>
        """
        try:
            response = llm.invoke(prompt)
            response_content = response.content.strip()
            pattern = r'<LEGAL_CONCEPTS>\s*(\[.*?\])\s*</LEGAL_CONCEPTS>'
            match = re.search(pattern, response_content, re.DOTALL)
            if match:
                concepts = json.loads(match.group(1))
                if isinstance(concepts, list) and all(isinstance(item, str) for item in concepts):
                    return [concept.strip().lower() for concept in concepts if concept.strip()]
                raise ValueError("Invalid JSON format")
            raise ValueError("No <LEGAL_CONCEPTS> tags found")
        except Exception as e:
            print(f"Error extracting concepts: {e}")
            return []

    def extract_relations(self, question: str, concepts: List[str]) -> List[Dict]:
        """Extract relationships using dependency parsing and LLM"""
        relations = []
        # Dependency parsing
        try:
            annotated = rdrsegmenter.annotate_text(question)
            sentence = annotated[0]
            for i, token in enumerate(sentence):
                if token['posTag'] in {'V', 'VB'}:
                    head_idx = token['head'] - 1  # Use 'head' instead of 'depLabel']['head']
                    if head_idx >= 0 and head_idx < len(sentence):
                        subject = None
                        obj = None
                        for j, dep_token in enumerate(sentence):
                            if dep_token['head'] == i + 1:
                                if dep_token['depLabel'] in {'nsubj', 'nsubj:pass'}:  # Use 'depLabel' directly
                                    subject = dep_token['wordForm']
                                elif dep_token['depLabel'] in {'obj', 'iobj'}:
                                    obj = dep_token['wordForm']
                        if subject and obj:
                            subject = next((c for c in concepts if subject.lower() in c.lower()), subject)
                            obj = next((c for c in concepts if obj.lower() in c.lower()), obj)
                            if subject in concepts and obj in concepts:
                                relations.append({
                                    "Name": token['wordForm'].lower(),
                                    "ConckeyS": subject,
                                    "ConckeyO": obj
                                })
        except Exception as e:
            print(f"Error in dependency parsing: {e}")

        # LLM-based extraction (unchanged)
        prompt = f"""
        Bạn là một chuyên gia pháp luật. Trích xuất quan hệ pháp lý từ câu hỏi và khái niệm cho trước.
        Câu hỏi: {question}
        Khái niệm: {concepts}
        Trả về JSON:
        {{
          "relations": [
            {{"Name": "tên quan hệ", "ConckeyS": "chủ thể", "ConckeyO": "đối tượng"}},
            ...
          ]
        }}
        Chỉ trích xuất quan hệ giữa các khái niệm đã cho.
        """
        try:
            response = llm.invoke(prompt)
            response_text = response.content.strip()
            cleaned = re.sub(r"^```json\s*|\s*```$", "", response_text)
            llm_relations = json.loads(cleaned).get("relations", [])
            for rel in llm_relations:
                if (rel["ConckeyS"] in concepts and rel["ConckeyO"] in concepts and
                        not any(r["Name"] == rel["Name"] and r["ConckeyS"] == rel["ConckeyS"] and
                                r["ConckeyO"] == rel["ConckeyO"] for r in relations)):
                    relations.append(rel)
        except Exception as e:
            print(f"Error in LLM relation extraction: {e}")

        return relations
    def create_question_graph(self, question: str, concepts: List[str], relations: List[Dict]) -> Dict:
        """Create a structured question graph with enhanced embeddings"""
        question_graph = nx.DiGraph()
        for concept in concepts:
            embedding = self.get_embedding(concept)
            question_graph.add_node(concept, node_type="concept", embedding=embedding)
        for relation in relations:
            subject = relation["ConckeyS"]
            rel_name = relation["Name"]
            obj = relation["ConckeyO"]
            if subject in concepts and obj in concepts:
                rel_text = f"{question} | {subject} {rel_name} {obj}"
                rel_embedding = self.get_embedding(rel_text)
                question_graph.add_edge(subject, obj, relation_type=rel_name, embedding=rel_embedding)
        return {
            "graph": question_graph,
            "concepts": concepts,
            "relations": relations
        }

    def match_graph_by_similarity(self, question_graph: Dict, threshold: float = 0.3, top_k: int = 5) -> Dict:
        q_graph = question_graph["graph"]
        matching_nodes = []
        matching_relations = []

        # Node similarity matching with lower threshold for better coverage
        node_mapping = {}
        for q_node, q_data in q_graph.nodes(data=True):
            q_emb = q_data["embedding"]
            similarities = []
            for g_node, g_data in self.graph.nodes(data=True):
                if "embedding" in g_data:
                    sim = cosine_similarity([q_emb], [g_data["embedding"]])[0][0]
                    similarities.append((g_node, float(sim)))  # Convert to Python float
            # Sort by similarity and take top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            print(f"Debug: Top similarities for '{q_node}': {similarities[:3]}")
            for g_node, sim in similarities[:top_k]:
                if sim > threshold:
                    node_mapping[q_node] = {"graph_node": g_node, "similarity": sim}
                    matching_nodes.append({
                        "query_node": q_node,
                        "graph_node": g_node,
                        "similarity": sim
                    })

        # Single-concept case - expand search to include related nodes
        if not q_graph.edges() and matching_nodes:
            matched_concepts = [node["graph_node"] for node in matching_nodes]
            subgraph = nx.DiGraph()
            
            # Add matched nodes and their immediate neighbors
            for node in matched_concepts:
                if node in self.graph:
                    subgraph.add_node(node, **self.graph.nodes[node])
                    # Add outgoing edges
                    for u, v, edge_data in self.graph.edges(node, data=True):
                        if u in self.graph and v in self.graph:
                            subgraph.add_node(v, **self.graph.nodes[v])
                            subgraph.add_edge(u, v, **edge_data)
                    # Add incoming edges
                    for u, v, edge_data in self.graph.in_edges(node, data=True):
                        if u in self.graph and v in self.graph:
                            subgraph.add_node(u, **self.graph.nodes[u])
                            subgraph.add_edge(u, v, **edge_data)

            articles = []
            for node in matched_concepts:
                for source, target, edge_data in self.graph.edges(data=True):
                    if (target == node and edge_data.get("relation_type") == "CONTAINS" and
                            self.graph.nodes[source].get("node_type") == "article"):
                        articles.append({
                            "article_id": source,
                            "article_number": self.graph.nodes[source].get("number", "unknown"),
                            "content": self.graph.nodes[source].get("content", "")
                        })
            articles = [dict(t) for t in {tuple(d.items()) for d in articles}]

            return {
                "nodes": list(subgraph.nodes),
                "edges": [
                    {"subject": u, "relation": d.get("relation_type"), "object": v}
                    for u, v, d in subgraph.edges(data=True)
                ],
                "articles": articles,
                "matched_nodes": matching_nodes,
                "matched_relations": matching_relations
            }

        # Existing relation similarity matching (for cases with relations)
        for u, v, data in q_graph.edges(data=True):
            q_emb = data["embedding"]
            best_match = None
            best_score = 0
            for g_u, g_v, g_data in self.graph.edges(data=True):
                if "embedding" in g_data and g_data["relation_type"] != "CONTAINS":
                    sim = cosine_similarity([q_emb], [g_data["embedding"]])[0][0]
                    if sim > best_score and sim > threshold:
                        best_score = float(sim)  # Convert to Python float
                        best_match = (g_u, g_v, g_data["relation_type"])
                if best_match:
                    matching_relations.append({
                        "query_relation": {"subject": u, "relation": data["relation_type"], "object": v},
                        "graph_relation": {"subject": best_match[0], "relation": best_match[2],
                                           "object": best_match[1]},
                        "similarity": best_score
                    })

        # Subgraph isomorphism (for cases with relations)
        matcher = isomorphism.DiGraphMatcher(self.graph, q_graph,
                                             node_match=lambda n1, n2: n1.get("node_type") == n2.get("node_type"),
                                             edge_match=lambda e1, e2: e1.get("relation_type") == e2.get(
                                                 "relation_type"))
        if matcher.subgraph_is_isomorphic():
            subgraph_mapping = matcher.mapping
            matching_nodes = [n for n in matching_nodes if n["query_node"] in subgraph_mapping]
            matching_relations = [
                r for r in matching_relations
                if r["query_relation"]["subject"] in subgraph_mapping and
                   r["query_relation"]["object"] in subgraph_mapping
            ]
        else:
            print("No isomorphic subgraph found; relying on similarity matches")

        return self.fetch_matched_subgraph(matching_nodes, matching_relations)

    def fetch_matched_subgraph(self, matching_nodes: List[Dict], matching_relations: List[Dict]) -> Dict:
        """Fetch a coherent subgraph based on matched nodes and relations"""
        matched_concepts = [node["graph_node"] for node in matching_nodes]
        subgraph = nx.DiGraph()
        for node in matched_concepts:
            if node in self.graph:
                subgraph.add_node(node, **self.graph.nodes[node])
        for rel in matching_relations:
            g_rel = rel["graph_relation"]
            if (g_rel["subject"] in matched_concepts and
                    g_rel["object"] in matched_concepts and
                    self.graph.has_edge(g_rel["subject"], g_rel["object"])):
                edge_data = self.graph.get_edge_data(g_rel["subject"], g_rel["object"])
                subgraph.add_edge(g_rel["subject"], g_rel["object"], **edge_data)

        articles = []
        for node in matched_concepts:
            for source, target, edge_data in self.graph.edges(data=True):
                if (target == node and edge_data.get("relation_type") == "CONTAINS" and
                        self.graph.nodes[source].get("node_type") == "article"):
                    articles.append({
                        "article_id": source,
                        "article_number": self.graph.nodes[source].get("number", "unknown"),
                        "content": self.graph.nodes[source].get("content", "")
                    })
        articles = [dict(t) for t in {tuple(d.items()) for d in articles}]

        return {
            "nodes": list(subgraph.nodes),
            "edges": [
                {"subject": u, "relation": d.get("relation_type"), "object": v}
                for u, v, d in subgraph.edges(data=True)
            ],
            "articles": articles,
            "matched_nodes": matching_nodes,
            "matched_relations": matching_relations
        }

    def generate_answer(self, question: str, subgraph: Dict) -> str:
        """Generate answer with enhanced context"""
        context = {
            "nodes": subgraph["nodes"],
            "edges": subgraph["edges"],
            "articles": [
                {
                    "article_id": a["article_id"],
                    "number": a["article_number"],
                    "content": a.get("content", "N/A")
                } for a in subgraph["articles"]
            ],
            "matched_nodes": subgraph["matched_nodes"],
            "matched_relations": subgraph["matched_relations"]
        }
        context_json = json.dumps(context, ensure_ascii=False, indent=2)

        prompt = f"""
        Bạn là một chuyên gia pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên đồ thị tri thức pháp luật.

        Câu hỏi: {question}
        Thông tin đồ thị:
        {context_json}

        Yêu cầu:
        1. Sử dụng thông tin từ nodes, edges và articles để trả lời.
        2. Trích dẫn số điều luật (article_number) nếu có.
        3. Nếu nội dung điều luật (content) có sẵn, tóm tắt hoặc trích dẫn.
        4. Trả lời rõ ràng, chính xác, đầy đủ.
        5. Nếu thiếu thông tin, nêu rõ hạn chế.
        Trả lời súc tích, dễ hiểu.
        """
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Lỗi khi sinh câu trả lời: {e}"

    def normalize_concepts_to_graph(self, concepts: List[str], threshold: float = 0.6) -> List[str]:
        """Normalize extracted concepts to most similar concepts in the base graph"""
        normalized_concepts = []
        
        # Get all entity nodes from the graph
        graph_entities = self.inspect_graph_entities()
        
        print(f"Debug: Available entities in graph: {len(graph_entities)}")
        if len(graph_entities) > 0:
            print(f"Debug: Sample entities: {graph_entities[:10]}")
        
        for concept in concepts:
            best_match = None
            best_similarity = 0
            
            # First try exact matching
            if concept in graph_entities:
                best_match = concept
                best_similarity = 1.0
                print(f"Exact match found: '{concept}'")
            else:
                # Try partial matching (substring matching)
                for entity in graph_entities:
                    # Check if concept is contained in entity or vice versa
                    if concept.lower() in entity.lower() or entity.lower() in concept.lower():
                        similarity = len(set(concept.lower().split()) & set(entity.lower().split())) / len(set(concept.lower().split()) | set(entity.lower().split()))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = entity
                
                # If no good partial match, try embedding similarity
                if not best_match or best_similarity < 0.3:
                    try:
                        # Get embedding for the concept
                        concept_embedding = self.get_embedding(concept)
                        
                        # Find the most similar entity in the graph
                        for entity in graph_entities:
                            if f"node:{entity}" in self.graph_embeddings:
                                entity_embedding = self.graph_embeddings[f"node:{entity}"]
                                similarity = float(cosine_similarity([concept_embedding], [entity_embedding])[0][0])  # Convert to Python float
                                
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = entity
                    except Exception as e:
                        print(f"Error computing embedding similarity for '{concept}': {e}")
            
            # Only use the normalized concept if similarity is above threshold
            if best_match and best_similarity >= threshold:
                normalized_concepts.append(best_match)
                print(f"Normalized '{concept}' -> '{best_match}' (similarity: {best_similarity:.3f})")
            else:
                # Keep original concept if no good match found
                normalized_concepts.append(concept)
                print(f"Kept original concept '{concept}' (best similarity: {best_similarity:.3f})")
                if best_match:
                    print(f"  Best candidate was: '{best_match}'")
                
                # Show related entities for debugging
                related_entities = self.inspect_graph_entities(concept)
                if related_entities:
                    print(f"  Related entities found: {related_entities[:5]}")
                else:
                    print(f"  No related entities found for '{concept}'")
        
        return normalized_concepts

    def safe_json_serialize(self, obj):
        """Safely serialize object for JSON output"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.safe_json_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.safe_json_serialize(item) for item in obj]
        else:
            return obj

    def question_answering_pipeline(self, question: str) -> str:
        """Complete pipeline for question answering"""
        try:
            noun_chunks = self.extract_noun_chunks(question)
            print(f"p1: Noun chunks: {noun_chunks}")
            concepts = self.extract_concepts(noun_chunks, question)
            print(f"p2: Concepts: {concepts}")
            
            # Normalize concepts to graph entities
            normalized_concepts = self.normalize_concepts_to_graph(concepts)
            print(f"p2.5: Normalized concepts: {normalized_concepts}")
            
            relations = self.extract_relations(question, normalized_concepts)
            print(f"p3: Relations: {relations}")
            question_graph = self.create_question_graph(question, normalized_concepts, relations)
            print(f"p4: Question graph nodes: {list(question_graph['graph'].nodes)}")
            print(f"p4: Question graph edges: {list(question_graph['graph'].edges)}")
            matching_subgraph = self.match_graph_by_similarity(question_graph, 0.3, 5)
            print(f"p5: Matched subgraph: {json.dumps(self.safe_json_serialize(matching_subgraph), ensure_ascii=False, indent=2)}")
            answer = self.generate_answer(question, matching_subgraph)
            self.save_embeddings()
            return answer
        except Exception as e:
            return f"Lỗi trong quá trình xử lý câu hỏi: {e}"

    def inspect_graph_entities(self, search_term: str = None) -> List[str]:
        """Inspect entities in the graph, optionally filtered by search term"""
        entities = []
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'entity':
                if search_term is None or search_term.lower() in node.lower():
                    entities.append(node)
        return entities

    def get_graph_statistics(self):
        """Print graph statistics"""
        print("\n=== Thống kê đồ thị ===")
        print(f"Tổng số nodes: {self.graph.number_of_nodes()}")
        print(f"Tổng số edges: {self.graph.number_of_edges()}")
        node_types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        for node_type, count in node_types.items():
            print(f"{node_type}: {count}")
        print(f"Cached embeddings: {len(self.embeddings_cache)}")
        print(f"Graph embeddings: {len(self.graph_embeddings)}")
        
        # Show sample entities
        entities = self.inspect_graph_entities()
        if entities:
            print(f"\nSample entities ({len(entities)} total):")
            for i, entity in enumerate(entities[:20]):
                print(f"  {i+1}. {entity}")
            if len(entities) > 20:
                print(f"  ... and {len(entities) - 20} more")


def main():
    #abs_path = os.path.join(os.path.dirname(__file__), 'legal_graph_with_embeddings.pkl')
    abs_path = os.path.join(os.path.dirname(__file__), 'legal_graph_underthesea.pkl')

    qa_system = NetworkXLegalQA(abs_path)
    qa_system.get_graph_statistics()
    try:
        while True:
            question = input("\nNhập câu hỏi của bạn (hoặc 'q' để thoát): ")
            if question.lower() == 'q':
                break
            print("\nĐang xử lý câu hỏi...")
            answer = qa_system.question_answering_pipeline(question)
            print("\n===== Câu trả lời =====")
            print(answer)
    except KeyboardInterrupt:
        print("\nThoát chương trình...")
    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    main()