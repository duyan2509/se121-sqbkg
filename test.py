from py_vncorenlp import VnCoreNLP
from dotenv import load_dotenv
import os
load_dotenv()
vncorenlp_dir = os.getenv("VNCORENLP_DIR")

rdrsegmenter = VnCoreNLP(
    save_dir=vncorenlp_dir,
    annotators=["wseg", "pos", "ner", "parse"],
    max_heap_size='-Xmx2g'
)

text = "Chi phí đầu tư vào đất còn lại là chi phí hợp lý mà người sử dụng đất đã đầu tư trực tiếp vào đất phù hợp với mục đích sử dụng đất nhưng đến thời điểm Nhà nước thu hồi đất còn chưa thu hồi hết."


print(f"Văn bản đầu vào: {text}")
annotation = rdrsegmenter.annotate_text(text)
print("=== Kết quả phân tích ===")
print(annotation)
