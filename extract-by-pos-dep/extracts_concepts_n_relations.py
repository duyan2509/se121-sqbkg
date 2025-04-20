from py_vncorenlp import VnCoreNLP
from dotenv import load_dotenv
import os
import sentence_graph
import knowledge_base
load_dotenv()
vncorenlp_dir = os.getenv("VNCORENLP_DIR")
rdrsegmenter = VnCoreNLP(
    save_dir=vncorenlp_dir,
    annotators=["wseg", "pos", "ner", "parse"],
    max_heap_size='-Xmx2g'
)

text = "Phòng, chống ma túy là phòng ngừa, ngăn chặn, đấu tranh chống tội phạm và tệ nạn ma túy; kiểm soát các hoạt động hợp pháp liên quan đến ma túy."
annotation = rdrsegmenter.annotate_text(text)

if annotation:
    for sentence_id, tokens in annotation.items():
        for token in tokens:
            print(token)


        valid_pos = {"N", "Nc", "Np", "V"}
        print("\n\nedge of sentence:")
        nodes = sentence_graph.sentence2graph(tokens)
        sentence_graph.display_sentence_tokens_graph(nodes)

        print("\n\nedge processed sentence:")
        valid_nodes = sentence_graph.filter_nodes_by_pos(nodes, valid_pos)
        nodes=sentence_graph.reconnect_edges(nodes, valid_nodes)
        sentence_graph.display_sentence_tokens_graph(valid_nodes)
        print(nodes)
        nodes_np, edges = knowledge_base.build_knowledge_graph_np(nodes)

        print("Các concept (nodes):")
        print(nodes_np)
        print("\nCác mối quan hệ (edges):")
        for subj, rel, obj in edges:
            print(f"{subj} -- {rel} --> {obj}")