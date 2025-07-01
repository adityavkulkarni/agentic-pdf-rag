import base64
import os
import re
import json
import logging
import shutil

import pdf2image
import pytesseract

from PIL import Image
from dotenv import load_dotenv

from . import image_parser
from .config_manager import config
from .models import PDFParserResults, ImageData, SummaryAndNER, PageDescription
from .openai_client import AzureOpenAIChatClient

load_dotenv()
logger = logging.getLogger(__name__)


class AgenticPDFParser:
    def __init__(self,
                 model="gpt-4o-2024-08-06",
                 openai_endpoint=None,
                 openai_api_key=None,
                 openai_api_version=None,
                 output_directory="pdf_images",
                 ):
        self.results = PDFParserResults()
        self.image_data= ImageData()

        self.pdf_path = None
        self.images = None
        self.pages_dir = None
        self.processed_dir = None
        self.groups_dir = None
        self.semantic_chunks_embeddings = None
        self.agentic_chunks_embeddings = None
        self.llm_response = None
        self.custom_extraction_llm_response = None

        self.output_directory = output_directory
        self.llm_client = AzureOpenAIChatClient(
            model=model or config.agentic_pdf_parser_model,
            api_key=openai_api_key or config.openai_api_key,
            api_endpoint=openai_endpoint or config.openai_endpoint,
            api_version=openai_api_version or config.openai_api_version
        )
        os.makedirs(self.output_directory, exist_ok=True)
        image_parser.group_index = 1

    def parse_file(self, pdf_file, file_name=None, custom_metadata=None):
        logger.info(f'Initializing AgenticPDFParser with file: {pdf_file}')
        self.pdf_path = pdf_file
        filename_with_extension = file_name if file_name else os.path.basename(pdf_file)
        logger.info('Converting PDF to images')
        self.images = pdf2image.convert_from_path(pdf_file)
        self.results.file_name = filename_with_extension
        logger.info(f'Found {len(self.images)} pages')
        filename, _ = os.path.splitext(self.results.file_name)
        if os.path.exists(os.path.join(self.output_directory, filename)):
            shutil.rmtree(os.path.join(self.output_directory, filename))
        os.makedirs(os.path.join(self.output_directory, filename), exist_ok=True)
        self.pages_dir = os.path.join(self.output_directory, filename, "pages")
        self.processed_dir = os.path.join(self.output_directory, filename, "processed")
        self.groups_dir = os.path.join(self.output_directory, filename, "groups")
        self.image_data.pages_directory = self.pages_dir
        self.image_data.processed_directory = self.processed_dir
        self.image_data.group_directory = self.groups_dir
        os.makedirs(self.pages_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.groups_dir, exist_ok=True)
        if custom_metadata:
            self.results.custom_metadata = custom_metadata

    def get_image_data(self):
        logger.info('Extracting image data')
        for i, image in enumerate(self.images):
            image_path = os.path.join(self.pages_dir, f'page_{i + 1}.jpg')
            image.save(image_path, 'JPEG')
            self.image_data.processed.append(os.path.join(self.processed_dir, f'processed_page_{i + 1}.jpg'))
            self.results.parsed_pdf.page_to_group_map[f"page_{i + 1}"] = image_parser.draw_text_group_boxes(image_path=image_path,
                                                                                                            output_path=os.path.join(self.processed_dir, f'processed_page_{i + 1}.jpg'),
                                                                                                            groups_output_dir=self.groups_dir)
        self.image_data.pages = [page for page in os.listdir(self.pages_dir)]
        self.image_data.pages.sort()
        self.results.parsed_pdf.processed_images = {
            os.path.basename(path): self._image_to_base64(path)
            for path in sorted(self.image_data.processed)
        }
        logger.debug(f"Page to group map: {self.results.parsed_pdf.page_to_group_map}")
        logger.info(f'Stored {len(self.image_data.pages)} page images')
        return self.image_data

    @staticmethod
    def _get_ocr_text(image_path):
        logger.info(f'OCR text extraction for: {image_path}')
        text = pytesseract.image_to_string(Image.open(image_path))
        text = re.sub(r'\n{2,}', '\n\n', text.strip())
        text = re.sub(r'\s*:\s*', ': ', text)
        text = re.sub(r'\s*,\s*', ', ', text)
        text = ' '.join(text.split())
        paragraphs = [p.strip() for p in text.split('\n\n')]
        text = '\n\n'.join(paragraphs)
        return text

    @staticmethod
    def _image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            encoded_string = base64.b64encode(image_data).decode('utf-8')
        return encoded_string

    def _get_page_content(self, image_path):
        logger.info(f'LLM for summary and feature extraction for page: {image_path}')
        prompt = (
            "Analyze the provided page /image and perform these tasks in sequence:\n"
            "1. **Text Extraction & Structure Reconstruction**\n"
            "   - Extract ALL text elements using AI vision models for poor-quality sections\n"
            "   - Preserve original:\n"
            "     * Hierarchical relationships (headers → subheaders → body)\n"
            "     * Visual layout (columns, tables, bullet points)\n"
            "   - Output: Structured string with layout annotations\n\n"
            "2. **Semantic Feature Extraction**\n"
            "   - Identify and catalog:\n"
            "     * **Signatures**: Location coordinates, validation status (handwritten/printed)\n"
            "     * **Visual Elements**: Diagrams/charts with type classification and caption linking\n"
            "     * **Special Fields**: Checkboxes (checked/unchecked), stamps, handwritten marginalia\n"
            "   - Output: JSON structure with element metadata and cross-references to text sections\n\n"
            "3. **Summarize and Create Description**\n"
            "   - Output: String with summary Description\n"
            "4. **Create a meaningful title"
        )
        llm_response = json.loads(
            self.llm_client.chat_completion(
                text=prompt, image_base64=self._image_to_base64(image_path), feature_model=PageDescription
            ).choices[0].message.content
        )
        logger.debug(f"LLM Response: {llm_response}")
        return llm_response

    def process_text(self):
        if self.groups_dir:
            for file in [f for f in sorted(os.listdir(self.groups_dir)) if
                         os.path.isfile(os.path.join(self.groups_dir, f))]:
                filename_with_extension = os.path.basename(os.path.join(self.groups_dir, file))
                filename, _ = os.path.splitext(filename_with_extension)
                self.results.parsed_pdf.groups[filename] = self._get_ocr_text(os.path.join(self.groups_dir, file))
            self.results.parsed_pdf.groups = dict(sorted(self.results.parsed_pdf.groups.items()))
        self.results.parsed_pdf.pages_ocr = [self._get_ocr_text(os.path.join(self.pages_dir, image_path)) for image_path in self.image_data.pages]
        self.results.parsed_pdf.pages_descriptions = [self._get_page_content(os.path.join(self.pages_dir, image_path)) for image_path in self.image_data.pages]
        self.results.parsed_pdf.pages_llm = [
            f"{page['text_content']} \n\n Visual information: {page['visual_elements']}"
            for page in self.results.parsed_pdf.pages_descriptions
        ]
        self.results.processed = True
        logger.info(f'Extracted text from {len(self.results.parsed_pdf.pages_llm)} pages')
        return self.results.parsed_pdf

    def get_summary_and_ner(self):
        try:
            logger.info(f'LLM for summary and named entities extraction')
            prompt = (
                "Given the following document, please perform the following tasks:\n"
                "- Summarize the main purpose and key points of the document."
                " The summary should be in-depth and capture the complete information.\n"
                "Summarize it so that it captures all the key information that may be used in retrieving "
                "this document. Mention all the key entities, dates, locations and other specific information.\n"
                "- Extract all named entities and return in a string with pipe separated named entities.\n"
                "- Extract a title from the document if present, else generate a title.\n"
                "- Extract an ID or document number or any such identifier.\n"
                "- Return the results in the provided format.\n\n"
                "document: \n"
                f"{'\n'.join(self.results.parsed_pdf.pages_llm)}"
            )
            self.llm_response = json.loads(
                self.llm_client.chat_completion(prompt, feature_model=SummaryAndNER).choices[0].message.content)
            logger.debug(f"LLM Response: {self.llm_response}")
            if self.results.parsed_pdf.pages_llm:
                if self.llm_response:
                    self.results.parsed_pdf.summary = self.llm_response["summary"]
                    self.results.parsed_pdf.ner = self.llm_response["ner"].split('|')
                    self.results.parsed_pdf.title = self.llm_response["title"]
                    self.results.parsed_pdf.identifier = self.llm_response["identifier"]
            return self.results.parsed_pdf
        except Exception as e:
            logger.error(f"Exception: {e}")

    def get_custom_extraction(self, custom_extraction_prompt, custom_feature_model):
        try:
            logger.info(f'LLM for custom fields extraction')
            self.custom_extraction_llm_response = json.loads(self.llm_client.chat_completion(
                custom_extraction_prompt.replace("<<document>>", '\n'.join(self.results.parsed_pdf.pages_llm)),
                feature_model=custom_feature_model
                ).choices[0].message.content
            )
            logger.debug(f"LLM Response: {self.custom_extraction_llm_response}")
            self.results.custom_extraction_llm_response = self.custom_extraction_llm_response
            return self.custom_extraction_llm_response
        except Exception as e:
            logger.error(f"Exception: {e}")

    def run_pipline(self, pdf_file, file_name=None,
                    custom_metadata=None, custom_extraction_prompt=None, custom_feature_model=None
                    ):
        self.parse_file(pdf_file, file_name, custom_metadata=custom_metadata)
        self.get_image_data()
        self.process_text()
        self.get_summary_and_ner()
        if custom_extraction_prompt:
            self.get_custom_extraction(custom_extraction_prompt, custom_feature_model)
        return self.results.to_dict()
