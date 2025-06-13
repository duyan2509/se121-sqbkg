import json
import networkx as nx
import pickle
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from dotenv import load_dotenv
import os

load_dotenv()


class NetworkXGraphLoader:
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # MultiDiGraph để hỗ trợ multiple edges giữa nodes

        # Load PhoBERT model và tokenizer
        print("Loading PhoBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.model = AutoModel.from_pretrained("vinai/phobert-base")
        self.model.eval()  # Set to evaluation mode

        # Cache cho embeddings để tránh tính lại
        self.embedding_cache = {}

    def get_embedding(self, text):
        """Tính embedding cho text sử dụng PhoBERT"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", max_length=256,
                                truncation=True, padding=True)

        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Lấy embedding của [CLS] token (first token)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        # Cache kết quả
        self.embedding_cache[text] = embedding
        return embedding

    def load_data(self, data):
        """Load dữ liệu từ JSON vào NetworkX graph"""
        print("Starting to load data into NetworkX graph...")

        for article in data:
            # Kiểm tra các key bắt buộc
            if 'article_id' not in article:
                print("Bỏ qua article không có article_id")
                continue

            article_id = article['article_id']
            article_number = article.get('article_number', 'unknown')

            print(f"Đang xử lý {article_id}...")

            # Tạo node cho article với embedding
            article_embedding = self.get_embedding(f"Article {article_number}")
            self.graph.add_node(
                article_id,
                node_type='article',
                number=article_number,
                embedding=article_embedding
            )

            # Kiểm tra xem có relations không
            if 'relations' not in article:
                print(f"Article {article_id} không có relations, bỏ qua...")
                continue

            # Tạo relationships
            for relation in article['relations']:
                rel_name = relation['Name']

                # Xử lý tên relationship an toàn hơn
                rel_name_clean = rel_name.replace(' ', '_')
                rel_name_clean = rel_name_clean.replace(',', '')
                rel_name_clean = rel_name_clean.replace('.', '')
                rel_name_clean = rel_name_clean.replace('-', '_')
                rel_name_clean = rel_name_clean.replace('(', '')
                rel_name_clean = rel_name_clean.replace(')', '')
                rel_name_clean = rel_name_clean.upper()

                # Xử lý ConckeyS - có thể là string hoặc array
                subjects = relation['ConckeyS']
                if isinstance(subjects, str):
                    subjects = [subjects]

                # Xử lý ConckeyO - có thể là string hoặc array
                objects = relation['ConckeyO']
                if isinstance(objects, str):
                    objects = [objects]

                # Tạo relationships từ mỗi subject tới mỗi object
                for subject in subjects:
                    for obj in objects:
                        # Tạo entity nodes với embeddings nếu chưa tồn tại
                        if not self.graph.has_node(subject):
                            subject_embedding = self.get_embedding(subject)
                            self.graph.add_node(
                                subject,
                                node_type='entity',
                                embedding=subject_embedding
                            )

                        if not self.graph.has_node(obj):
                            obj_embedding = self.get_embedding(obj)
                            self.graph.add_node(
                                obj,
                                node_type='entity',
                                embedding=obj_embedding
                            )

                        # Tạo relationship giữa subject và object
                        self.graph.add_edge(
                            subject, obj,
                            relation_type=rel_name,
                            relation_clean=rel_name_clean
                        )

                        # Tạo relationships từ article tới entities
                        self.graph.add_edge(
                            article_id, subject,
                            relation_type='CONTAINS'
                        )
                        self.graph.add_edge(
                            article_id, obj,
                            relation_type='CONTAINS'
                        )

        print("Data loaded successfully!")

    def save_graph(self, filename='graph_with_embeddings.pkl', graphml_file='graph.graphml'):
        """Lưu graph vào pickle và GraphML (không lưu embedding vào GraphML để nhẹ và dễ visualize)"""
        import copy

        # node - edge - embedding on pickle
        graph_data = {
            'graph': self.graph,
            'embedding_cache': self.embedding_cache
        }

        with open(filename, 'wb') as f:
            pickle.dump(graph_data, f)
        print(f"✅ Graph saved to {filename} (pickle format)")
        # graphml for visualize
        graph_copy = copy.deepcopy(self.graph)

        for node, data in graph_copy.nodes(data=True):
            if 'embedding' in data:
                del data['embedding']

        for u, v, k, data in graph_copy.edges(data=True, keys=True):
            if 'embedding' in data:
                del data['embedding']

        try:
            nx.write_graphml(graph_copy, graphml_file)
            print(f"GraphML saved to {graphml_file} (without embeddings)")
        except Exception as e:
            print(f" Could not save GraphML: {e}")

    def load_graph(self, filename='graph_with_embeddings.pkl'):
        """Load graph và embeddings từ file"""
        with open(filename, 'rb') as f:
            graph_data = pickle.load(f)

        self.graph = graph_data['graph']
        self.embedding_cache = graph_data['embedding_cache']
        print(f"Graph loaded from {filename}")

    def get_statistics(self):
        """In thống kê về graph"""
        print("\n=== Graph Statistics ===")
        print(f"Total nodes: {self.graph.number_of_nodes()}")
        print(f"Total edges: {self.graph.number_of_edges()}")

        # Thống kê theo loại node
        node_types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        for node_type, count in node_types.items():
            print(f"{node_type}: {count}")

        # Thống kê về embeddings
        print(f"Cached embeddings: {len(self.embedding_cache)}")

    def find_similar_entities(self, entity_name, top_k=5):
        """Tìm các entities tương tự dựa trên embedding"""
        if entity_name not in self.graph.nodes():
            print(f"Entity '{entity_name}' not found in graph")
            return []

        target_embedding = self.graph.nodes[entity_name].get('embedding')
        if target_embedding is None:
            print(f"No embedding found for entity '{entity_name}'")
            return []

        similarities = []
        for node, data in self.graph.nodes(data=True):
            if node != entity_name and data.get('node_type') == 'entity':
                embedding = data.get('embedding')
                if embedding is not None:
                    # Tính cosine similarity
                    similarity = np.dot(target_embedding, embedding) / (
                            np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                    )
                    similarities.append((node, similarity))

        # Sắp xếp theo độ tương tự
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def main():
    # Đọc dữ liệu từ file
    with open('structured_relations_vncore.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    loader = NetworkXGraphLoader()

    try:
        # Load dữ liệu
        loader.load_data(data)

        # Lưu graph
        loader.save_graph(
            filename='legal_graph_with_embeddings.pkl',
            graphml_file='graphs/legal_graph.graphml'
        )

        # In thống kê
        loader.get_statistics()

        # Ví dụ tìm kiếm entities tương tự (nếu có data)
        # entities = list(loader.graph.nodes())
        # entity_nodes = [n for n, d in loader.graph.nodes(data=True)
        #                if d.get('node_type') == 'entity']
        # if entity_nodes:
        #     sample_entity = entity_nodes[0]
        #     print(f"\nSimilar entities to '{sample_entity}':")
        #     similar = loader.find_similar_entities(sample_entity)
        #     for entity, similarity in similar:
        #         print(f"  {entity}: {similarity:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()