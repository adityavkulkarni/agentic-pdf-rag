from typing import Optional
from pydantic import BaseModel, Field
from typing import List, Dict, Any


# Agentic Chunker
class ChunkID(BaseModel):
    """Extracting the chunk id"""
    chunk_id: Optional[str]

    class Config:
        extra = "forbid"


# PDF Parser
class ParsedPDF(BaseModel):
    pages_ocr: List[Any] = Field(default_factory=list)
    pages_llm: List[Any] = Field(default_factory=list)
    pages_descriptions: List = Field(default_factory=list)
    groups: Dict[Any, Any] = Field(default_factory=dict)
    summary: str = ""
    processed_images: Dict[Any, Any] = Field(default_factory=dict)
    ner: List[Any] = Field(default_factory=list)
    title: str = ""
    identifier: str = ""
    page_to_group_map: Dict[Any, Any] = Field(default_factory=dict)

class ImageData(BaseModel):
    pages: List[Any] = Field(default_factory=list)
    processed: List[Any] = Field(default_factory=list)
    processed_directory: str = ""
    group_directory: str = ""
    pages_directory: str = ""

class PDFParserResults(BaseModel):
    processed: bool = False
    file_name: str = ""
    parsed_pdf: ParsedPDF = Field(default_factory=ParsedPDF)
    custom_metadata: Dict[Any, Any] = Field(default_factory=dict)
    custom_extraction_llm_response: Dict[Any, Any] = Field(default_factory=dict)

    def to_dict(self):
        result = self.model_dump()
        result['parsed_pdf'] = self.parsed_pdf.model_dump()
        return result


class PageDescription(BaseModel):
    title: str = Field(..., description="The title of the extraction.")
    summary: str = Field(..., description="The summary of the extraction.")
    text_content: str = Field(..., description="Text present in image. Format it as paragraphs.")
    visual_elements: str = Field(..., description="Additional non-textual features like diagrams, signatures etc.")
    class Config:
        extra = "forbid"


class SummaryAndNER(BaseModel):
    summary: str = Field(..., description="The summary of the extraction.")
    ner: str = Field(..., description="The string containing named entities.")
    title: str = Field(..., description="The title of the extraction.")
    identifier: str = Field(..., description="The ID or number or any such element of the extraction.")

    class Config:
        extra = "forbid"


# PDF Chunker
class PDFChunkerResults(BaseModel):
    semantic_chunks: List[Any] = Field(default_factory=list)
    agentic_chunks: List[Any] = Field(default_factory=list)

    def to_dict(self):
        return {
            "semantic_chunks": self.semantic_chunks,
            "agentic_chunks": self.agentic_chunks,
        }


# rag
class QueryType(BaseModel):
    type: str = Field(..., description="Type of query: chunks or summary")
    files: str = Field(..., description="pipe separated list of filenames")
    class Config:
        extra = "forbid"


class SummaryResponse(BaseModel):
    files: str = Field(..., description="pipe separated list of filenames")
    augmented_query: str = Field(..., description="augmented query")
    class Config:
        extra = "forbid"

class ContextType(BaseModel):
    context_type: str = Field(..., description="document/page/hybrid")
    class Config:
        extra = "forbid"

class MultiModalResponse(BaseModel):
    summary: str = Field(..., description="Summary of the response")
    response: str = Field(..., description="Response of the query")
    markdown: str = Field(..., description="Markdown formatted response (optional)")
    class Config:
        extra = "forbid"
