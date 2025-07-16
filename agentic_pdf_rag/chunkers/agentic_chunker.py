import json
import uuid
import logging

from ..config_manager import config
from ..models import ChunkID
from ..clients import AzureOpenAIChatClient


logger = logging.getLogger(__name__)


class AgenticChunker:
    def __init__(self,
                 context="",
                 generate_new_metadata_ind=True,
                 llm_client=None,
                 model=None,
                 openai_endpoint=None,
                 openai_api_key=None,
                 openai_api_version=None,
                 metadata={}
                 ):
        self.chunks = dict()
        self.id_truncate_limit = 5
        self.generate_new_metadata_ind = generate_new_metadata_ind
        self.print_logging = True
        self.context = context
        self.metadata = metadata
        self.llm_client = llm_client  if llm_client else AzureOpenAIChatClient(
            model=model or config.agentic_chunker_model,
            api_key=openai_api_key or config.openai_api_key,
            api_endpoint=openai_endpoint or config.openai_endpoint,
            api_version=openai_api_version or config.openai_api_version
        )

    def reset(self):
        self.chunks = dict()

    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)

    def add_proposition(self, proposition):
        if self.print_logging:
            logger.debug(f"Adding: '{proposition}'")
        if len(self.chunks) == 0:
            if self.print_logging:
                logger.info("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return
        chunk_id = self._find_relevant_chunk(proposition)
        if chunk_id and chunk_id in self.chunks:
            if self.print_logging:
                logger.info(f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
            return
        else:
            if self.print_logging:
                logger.info("No chunks found")
            self._create_new_chunk(proposition)

    def add_proposition_to_chunk(self, chunk_id, proposition):
        self.chunks[chunk_id]['propositions'].append(proposition)
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])
            if self.metadata.get(proposition) not in self.chunks[chunk_id]['locators']:
                self.chunks[chunk_id]['locators'].append(self.metadata.get(proposition))

    def _update_chunk_summary(self, chunk):
        prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "As a chunk steward responsible for grouping similar sentences, generate a 1-sentence summary that: \n"
                            "1. Identifies the common theme across propositions using context-aware generalization\n"
                            "2. Specifies acceptable addition criteria\n"
                            "3. Maintains natural language flow\n\n"
                            "**Instructions:**\n"
                            "- Generalize specific instances to broader categories (e.g., apples → fruits, June 16 → dates)\n"
                            "- Include both topical focus and addition guidelines\n"
                            "- Keep under 25 words\n\n"
                            "**Examples:**\n"
                            "Input: Proposition: Greg likes to eat pizza  \n"
                            "Output: 'This chunk contains food preferences with focus on Italian cuisine items.'\n\n"
                            "Input: 'Meeting scheduled June 16'  \n"
                            "Output: This chunk tracks scheduled events requiring date and participant details.\n\n"
                            "**Current Context:**\n"
                            f"{self.context}\n\n"
                            "Respond only with the new summary and nothing else."
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Chunk's propositions:\n{'\n'.join(chunk['propositions'])}\n\nCurrent chunk summary:\n{chunk['summary']}"
                    }
                ]
            }
        ]
        result = self.llm_client.chat_completion(prompt)
        try:
            summary = result.choices[0].message.content
        except Exception as e:
            print(chunk)
            summary = chunk['summary']
        return summary

    def _update_chunk_title(self, chunk):
        prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "As chunk steward responsible for grouping similar sentences, generate a 2-4 word capitalized title that:  \n"
                            "1. Abstracts specific propositions to broader categories  \n"
                            "2. Maintains Pythonic naming conventions  \n"
                            "3. Aligns with existing chunk taxonomy\n\n"
                            "**Rules:**  \n"
                            "- Generalize nouns to parent categories (e.g., pizza → Food)  \n"
                            "- Use ampersands instead of and  \n"
                            "- Omit articles (a/the)  \n"
                            "- Prioritize existing taxonomy terms for Agentic Chunking, Document Structuring, Search Systems\n\n"
                            "**Examples:**  \n"
                            "Input Summary: Contains dates and event scheduling details  \n"
                            "Output: Temporal Events  \n"
                            "Input Summary: Food preferences including Italian cuisine  \n"
                            "Output: Culinary Preferences  \n\n"
                            "**Current Context:**\n"
                            f"{self.context}\n\n"
                            "Respond only with New Title and nothing else"
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                             "text": f"Chunk's propositions:\n{'\n'.join(chunk['propositions'])}\n\nChunk summary:\n{chunk['summary']}\n\nCurrent chunk title:\n{chunk['title']}"
                    }
                ]
            }
        ]
        result = self.llm_client.chat_completion(prompt)
        try:
            title =  result.choices[0].message.content
            if title is None or title == 'None':
                title = chunk['title']
        except Exception as e:
            title = chunk['title']
        return title

    def _get_new_chunk_summary(self, proposition):
        prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.\n"
                            "You will be given a proposition which will go into a new chunk. This new chunk needs a summary.\n"
                            "Generate a 1-sentence chunk summary that:  \n"
                            "1. Abstracts specific instances to categorical themes  \n"
                            "2. A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.\n"
                            "3. Aligns with document structure patterns  \n\n"
                            "**Instructions:**  \n"
                            "- Generalize using hierarchy: Instance → Category → Domain (e.g., 'Python' → Programming → Computer Science)  \n"
                            "- Format: 'Contains [domain]-related [category] with focus on [specifics]'  \n"
                            "- Exclude proper nouns and exact dates  \n\n"
                            "**Examples:**  \n"
                            "Input: 'Customizing Python dictionaries'  \n"
                            "Output: 'Contains programming-language customization patterns focusing on data structure extensions [dict]'  \n"
                            "Input: 'June 2025 meeting schedule'  \n"
                            "Output: 'Contains temporal event coordination details requiring date/time validation [datetime]'  \n\n"
                            "**Current Context:**  \n"
                            f"{self.context}  \n\n"
                            "Respond only with summary and nothing else\n"
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Determine the summary of the new chunk that this proposition will go into:\n{proposition}"
                    }
                ]
            }
        ]
        result = self.llm_client.chat_completion(prompt)
        try:
            summary = result.choices[0].message.content
        except Exception as e:
            print(proposition)
            summary = 'summary'
        return summary

    def _get_new_chunk_title(self, summary):
        prompt = [
            {
                "role": "system",
                "content": [
                {
                    "type": "text",
                    "text": (
                        "As chunk steward responsible for grouping similar sentences, generate 2-4 word title that:  \n"
                        "1. Abstracts to categorical domains using Python class naming conventions  \n"
                        "2. Prioritizes existing taxonomy for Document Structuring, Search Systems, Data Customization \n"
                        "3. Uses ampersands for multi-concept chunks  \n"
                        "**Rules:**  \n"
                        "- Generalize hierarchy: Instance → Class → Module (e.g., 'dictionaries' → Data Structures → Python Customization)  \n"
                        "- Capitalize main concepts  \n"
                        "- Omit articles and verbs  \n"
                        "**Examples:**  \n"
                        "Input Summary: 'Python dictionary method extensions'  \n"
                        "Output: 'Data Structure Extensions'  \n"
                        "Input Summary: 'Document formatting with bounding boxes'  \n"
                        "Output: 'Document Structuring'  \n"
                        "**Current Context:**  \n"
                        f"{self.context}  \n"
                        "Respond only with the Title and nothing else."
                    )
                }
            ]
        },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Determine the title of the chunk that this summary belongs to:\n{summary}"
                    }
                ]
            }
        ]
        result = self.llm_client.chat_completion(prompt)
        try:
            title = result.choices[0].message.content
        except Exception as e:
            print(summary)
            title = 'title'
        return title

    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]  # I don't want long ids
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index': len(self.chunks),
            'locators': [self.metadata.get(proposition, {})],
        }
        if self.print_logging:
            logger.info(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

    def get_chunk_outline(self):
        """
        Get a string which represents the chunks you currently have.
        This will be empty when you first start off
        """
        chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ID: {chunk['chunk_id']}\nChunk Name: {chunk['title']}\nChunk Summary: {chunk['summary']}\n\n"""

            chunk_outline += single_chunk_string

        return chunk_outline

    def _find_relevant_chunk(self, proposition):
        current_chunk_outline = self.get_chunk_outline()
        prompt = [
            {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "As chunk steward responsible for grouping similar sentences, "
                        "analyze semantic relationships between the new proposition and existing chunks using the following criteria:\n"
                        "1. Semantic Similarity: Core meaning and key entities between the new proposition and propositions in a chunk\n"
                        "2. Intent Alignment: Purpose and objectives\n"
                        "3. Contextual Relevance: Domain-specific connections\n\n"
                        "**Steps:**\n"
                        "Compare proposition against each chunk's:\n"
                        "   - Title\n"
                        "   - Summary\n"
                        "   - Existing propositions\n"
                        "**Example 1:**\n"
                        "  - Proposition: 'Greg really likes hamburgers'\n"
                        "  - Current Chunks:\n"
                        "      - Chunk ID: 2n4l3d\n"
                        "      - Chunk Name: Places in San Francisco\n"
                        "      - Chunk Summary: Overview of the things to do with San Francisco Places\n\n"
                        "      - Chunk ID: 93833k\n"
                        "      - Chunk Name: Food Greg likes\n"
                        "      - Chunk Summary: Lists of the food and dishes that Greg likes\n"
                        "Output: 93833k\n\n"
                        "**Current Context:**  \n"
                        f"{self.context}  \n"
                        "Respond only with the chunk ID and if no chunk matches, return None"
                    )
                }
            ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"
                    }
                ]
            }
        ]
        result = self.llm_client.chat_completion(prompt, feature_model=ChunkID)
        chunk_id = json.loads(result.choices[0].message.content)["chunk_id"]

        if chunk_id and len(chunk_id) != self.id_truncate_limit:
            return None

        return chunk_id

    def get_chunks(self):
        chunks = []
        for chunk_id, chunk in self.chunks.items():
            chunks.append({
                "metadata": {
                    "chunk_id": chunk_id,
                    "title": chunk['title'],
                    "summary": chunk['summary'],
                    "agentic_chunk": "\n\n".join([x for x in chunk['propositions']]),
                    "locators": chunk["locators"],
                }
            })
        return chunks

    def pretty_print_chunks(self):
        logger.info(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            logger.info(f"Chunk #{chunk['chunk_index']}")
            logger.info(f"Chunk ID: {chunk_id}")
            logger.info(f"Summary: {chunk['summary']}")
            logger.info(f"Propositions:")
            for prop in chunk['propositions']:
                logger.info(f"    -{prop}")
            logger.info("\n\n")

    def pretty_print_chunk_outline(self):
        logger.info("Chunk Outline\n")
        logger.info(self.get_chunk_outline())


if __name__ == "__main__":
    propositions = [
        'The month is October.',
        'The year is 2023.',
        "One of the most important things that I didn't understand about the world as a child was the degree to which the returns for performance are superlinear.",
        'Teachers and coaches implicitly told us that the returns were linear.',
        "I heard a thousand times that 'You get out what you put in.'",
    ]
    ac = AgenticChunker()
    ac.add_propositions(propositions)
    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()
    print(ac.get_chunks())
