from pydantic import BaseModel, Field
from typing import List

class Concept(BaseModel):
    """
    Đại diện cho một khái niệm trong văn bản pháp luật.
    """
    name: str = Field(
        ...,
        description="Tên định danh của khái niệm.",
    )

    meaning: str = Field(
        ...,
        description="Định nghĩa hoặc ý nghĩa chính thức của khái niệm.",
    )

    attrs: List[str] = Field(
        ...,
        description="Danh sách các thuộc tính mô tả khái niệm.",
    )

    keyphrases: List[str] = Field(
        ...,
        description="Các cụm từ khóa liên quan hoặc đại diện cho khái niệm.",
    )

    similar: List[str] = Field(
        ...,
        description="Các biểu diễn khác hoặc cách gọi tương đương của khái niệm.",
    )

class ArticleConcepts(BaseModel):
    """
    """
    title: str = Field(
        ...,
        description="Tiêu đề hoặc mã định danh của điều luật.",
    )

    concepts: List[Concept] = Field(
        ...,
        description="Danh sách các khái niệm liên quan trong điều luật."
    )
