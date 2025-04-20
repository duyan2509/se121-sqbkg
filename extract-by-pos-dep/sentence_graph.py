class TokenNode:
    def __init__(self, index, wordForm, posTag, nerLabel, head, depLabel):
        self.index = index
        self.wordForm = wordForm # token
        self.posTag = posTag
        self.nerLabel = nerLabel
        self.head = head
        self.depLabel = depLabel
        self.edges = []

    def add_edge(self, edge):
        self.edges.append(edge)

    def __repr__(self):
        return f"Token Node(index={self.index}, wordForm='{self.wordForm}', pos='{self.posTag}', edges={self.edges})"


def sentence2graph(tokens: list[dict]) -> dict:
    """build graph for sentence"""
    tokenNodes = {token["index"]: TokenNode(token["index"], token["wordForm"], token["posTag"], token["nerLabel"], token["head"],
                                 token["depLabel"]) for token in tokens}
    for token in tokens:
        if token["head"] != 0:
            tokenNodes[token["head"]].add_edge(token["index"])
    return tokenNodes


def filter_nodes_by_pos(nodes: dict, valid_pos_tags: set = {"N", "Nc", "Np", "V"}) -> dict:
    return {
        index: node for index, node in nodes.items()
        if node.posTag in valid_pos_tags
    }


def reconnect_edges(nodes:dict, valid_nodes:dict) -> dict:
    for index, node in list(valid_nodes.items()):
        new_edges = []
        for edge in node.edges:
            if edge in valid_nodes:
                new_edges.append(edge)
            else:
                if edge in nodes:
                    for grandchild in nodes[edge].edges:
                        if grandchild in valid_nodes:
                            new_edges.append(grandchild)
        node.edges = new_edges
    return nodes

def display_sentence_tokens_graph(valid_nodes:dict):
    for index, node in valid_nodes.items():
        for edge in node.edges:
            print(f"[{node.index}-{node.wordForm}] : [{valid_nodes[edge].index}-{valid_nodes[edge].wordForm}]")

