from collections import defaultdict, deque

#
# class Node:
#     def __init__(self, index, wordForm, posTag, nerLabel, head, depLabel):
#         self.index = index
#         self.wordForm = wordForm
#         self.posTag = posTag
#         self.nerLabel = nerLabel
#         self.head = head
#         self.depLabel = depLabel
#         self.edges = []  # Lưu danh sách các index của các node kết nối
#
#     def add_edge(self, edge):
#         self.edges.append(edge)
#
#     def __repr__(self):
#         return f"Node(index={self.index}, word='{self.wordForm}', pos='{self.posTag}', edges={self.edges})"


def build_children_mapping(nodes):
    """Tạo mapping từ một node (theo index) đến danh sách các node con của nó."""
    children = defaultdict(list)
    for node_index, node in nodes.items():
        print(node, node.index, node.head)
        if node.head != 0:
            children[node.head].append(node.index)
    return children


def is_np_head(node, node_dict):
    """
    Kiểm tra xem node có phải là NP head hay không.
    Ta xem một danh từ là NP head nếu:
      - Nó không có cha (head == 0) hoặc
      - Cha của nó không phải là danh từ, hoặc
      - Quan hệ nối với cha không thuộc tập các nhãn mở rộng (ví dụ: nmod, amod, det, compound, loc, pob)
    """
    if node.head == 0:
        return True
    parent = node_dict[node.head]
    allowed_upward = {'nmod', 'amod', 'det', 'compound', 'loc', 'pob'}
    if parent.posTag == 'N' and node.depLabel in allowed_upward:
        return False
    return True


def extract_np_phrase(np_head_index, children, node_dict, allowed_downward=None):
    """
    Từ NP head, duyệt cây con với các nhãn phụ thuộc cho phép và thu thập các node.
    Sau đó, sắp xếp theo thứ tự xuất hiện và ghép lại thành cụm.
    """
    if allowed_downward is None:
        allowed_downward = {'nmod', 'loc', 'adv', 'vmod', 'amod', 'det', 'compound', 'pob', 'conj'}
    indices = set()

    def dfs(idx):
        indices.add(idx)
        for child in children.get(idx, []):
            if node_dict[child].depLabel in allowed_downward:
                dfs(child)

    dfs(np_head_index)
    sorted_indices = sorted(indices)
    phrase = " ".join(node_dict[idx].wordForm for idx in sorted_indices)
    return phrase


def bfs_path(graph, start, goal):
    """Tìm đường đi ngắn nhất từ start đến goal trong đồ thị không hướng."""
    visited = {start}
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        current = path[-1]
        if current == goal:
            return path
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append(new_path)
    return None


def build_dependency_graph(nodes):
    """Xây dựng đồ thị phụ thuộc không hướng từ dictionary Node."""
    graph = {}
    for node_index, node in nodes.items():
        idx = node.index
        graph.setdefault(idx, [])
        if node.head != 0:
            graph[idx].append(node.head)
            graph.setdefault(node.head, []).append(idx)
    return graph


def build_knowledge_graph_np(nodes):
    """
    Xây dựng đồ thị tri thức với các concept là cụm từ (NP) trích xuất từ dependency parsing.
    Các bước:
      1. Tạo từ điển node theo index và mapping từ node đến các node con.
      2. Với mỗi Node có posTag 'N' hoặc 'Nc', nếu là NP head (theo hàm is_np_head), trích xuất cụm NP.
      3. Xây dựng đồ thị phụ thuộc không hướng.
      4. Với mỗi cặp NP (đại diện bởi NP head), tìm đường đi ngắn nhất trên đồ thị.
         Trong đường đi (ngoại trừ đầu và cuối), các node có posTag 'V' được ghép lại làm nhãn quan hệ.
    """
    # Tạo từ điển mapping từ index đến Node
    node_dict = nodes
    children = build_children_mapping(nodes)

    # Trích xuất NP (concept) từ các Node có posTag 'N' hoặc 'Nc'
    np_phrases = {}
    for node_index, node in nodes.items():
        if (node.posTag == 'N' or node.posTag == 'Nc') and is_np_head(node, node_dict):
            phrase = extract_np_phrase(node.index, children, node_dict)
            np_phrases[node.index] = phrase

    # Xây dựng đồ thị phụ thuộc không hướng
    dep_graph = build_dependency_graph(nodes)

    edges = []
    np_head_indices = list(np_phrases.keys())
    n = len(np_head_indices)
    for i in range(n):
        for j in range(i + 1, n):
            start = np_head_indices[i]
            end = np_head_indices[j]
            path = bfs_path(dep_graph, start, end)
            if path:
                # Lấy các node trung gian (loại trừ start và end)
                intermediate = path[1:-1]
                # Trích xuất các token có posTag là 'V' làm nhãn quan hệ
                relation_tokens = [node_dict[idx].wordForm for idx in intermediate if node_dict[idx].posTag == 'V']
                if relation_tokens:
                    relation = " ".join(relation_tokens)
                    edges.append((np_phrases[start], relation, np_phrases[end]))

    nodes_np = list(set(np_phrases.values()))
    return nodes_np, edges