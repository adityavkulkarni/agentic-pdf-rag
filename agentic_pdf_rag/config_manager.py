import os
import logging
import configparser

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_file=None, openai_api_key=None, openai_embeddings_api_key=None, ):
        if config_file is None:
            config_file = os.path.join(os.getcwd(), "config", "config.ini")
            logger.info(f"No config_file provided. Using default: {config_file}")
        else:
            logger.info(f"Using provided config_file: {config_file}")
        config = configparser.ConfigParser()
        read_file = config.read(config_file)
        if not read_file:
            logger.error(f"Config file not found or could not be read: {config_file}")
            raise FileNotFoundError(f"Config file not found: {config_file}")
        logger.info(f"Config file loaded: {config_file}")

        try:
            models = config['models']
            self.agentic_pdf_parser_model = models.get('agentic_pdf_parser_model')
            self.agentic_chunker_model = models.get('agentic_chunker_model')
            self.openai_embedding_model = models.get('openai_embedding_model')
            self.use_qwen3 = True if models.get('use_qwen3') == "true" else False
        except KeyError as e:
            logger.error(f"Missing 'models' section or key: {e}")
            raise

        try:
            azure_openai = config['azure_openai']
            self.openai_api_key = openai_api_key or os.getenv("AZURE_OPENAI_API_KEY", None)
            if self.openai_api_key:
                logger.info("OpenAI API key loaded from  environment variable.")
            else:
                logger.error("OpenAI API key not found in arguments or environment variables.")
                raise Exception("OpenAI API key not found in arguments or environment variables.")
            self.openai_embedding_api_key = openai_embeddings_api_key or os.getenv("AZURE_OPENAI_API_KEY", None)
            if self.openai_embedding_api_key:
                logger.info("OpenAI API key loaded from  environment variable.")
            else:
                logger.error("OpenAI API key not found in arguments or environment variables.")
                raise Exception("OpenAI API key not found in arguments or environment variables.")
            self.openai_endpoint = azure_openai.get('openai_endpoint')
            self.openai_embedding_endpoint = azure_openai.get('openai_embeddings_endpoint')
            self.openai_api_version = azure_openai.get('openai_api_version')
            self.openai_embedding_api_version = azure_openai.get('openai_embeddings_api_version')
        except KeyError as e:
            logger.error(f"Missing 'azure_openai' section or key: {e}")
            raise

        try:
            directories = config['directories']
            self.output_directory = directories.get('output_directory')
        except KeyError as e:
            logger.error(f"Missing 'directories' section or key: {e}")
            raise

        try:
            database = config['database']
            self.db_name = database.get('dbname')
            self.db_user = database.get('user')
            self.db_password = database.get('password')
            self.db_host = database.get('host')
            self.db_port = database.get('port')
        except KeyError as e:
            logger.error(f"Missing 'database' section or key: {e}")
            raise

        try:
            neo4j = config['neo4j']
            self.neo4j_uri = neo4j.get('uri')
            self.neo4j_user = neo4j.get('user')
            self.neo4j_password = neo4j.get('password')
        except KeyError as e:
            logger.warning(f"Missing 'neo4j' section or key: {e}")
            logger.info(f"Not using Knowledg graphs")
            raise
        try:
            docling = config['docling']
            self.docling_url = docling.get('docling_url')
        except KeyError as e:
            logger.error(f"Missing 'docling' section or key: {e}")

        try:
            document_manager = config['document_manager']
            self.document_manager_url = document_manager.get('document_manager_url', None)
        except KeyError as e:
            logger.error(f"Missing 'docling' section or key: {e}")

        logger.info(f"Loaded config file:")
        for key, value in self.to_dict().items():
            logger.info(f"{key}: {value}")

    def to_dict(self):
        return {
            "agentic_pdf_parser_model": self.agentic_pdf_parser_model,
            "agentic_chunker_model": self.agentic_chunker_model,
            "openai_embedding_model": self.openai_embedding_model,
            "use_qwen3": self.use_qwen3,
            "openai_api_key": "<REDACTED>",
            "openai_embeddings_api_key": "<REDACTED>",
            "openai_endpoint": self.openai_endpoint,
            "openai_embeddings_endpoint": self.openai_embedding_endpoint,
            "openai_api_version": self.openai_api_version,
            "openai_embeddings_api_version": self.openai_embedding_api_version,
            "output_directory": self.output_directory,
            "db_name": self.db_name,
            "db_user": self.db_user,
            "db_password": "<REDACTED>",
            "db_host": self.db_host,
            "db_port": self.db_port,
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_user,
            "neo4j_password": self.neo4j_password,
            "docling_url": self.docling_url,
            "document_manager_url": self.document_manager_url,
        }

    def set_config(self, config):
        if config.get('openai_api_key'):
            os.environ['AZURE_OPENAI_API_KEY'] = config.get('openai_api_key')
        if "agentic_pdf_parser_model" in config:
            self.agentic_pdf_parser_model = config["agentic_pdf_parser_model"]
        if "agentic_chunker_model" in config:
            self.agentic_chunker_model = config["agentic_chunker_model"]
        if "openai_embedding_model" in config:
            self.openai_embedding_model = config["openai_embedding_model"]
        if "use_qwen3" in config:
            self.use_qwen3 = config["use_qwen3"]
        if "openai_api_key" in config:
            self.openai_api_key = config["openai_api_key"]
        if "openai_embeddings_api_key" in config:
            self.openai_embedding_api_key = config["openai_embeddings_api_key"]
        if "openai_endpoint" in config:
            self.openai_endpoint = config["openai_endpoint"]
        if "openai_embeddings_endpoint" in config:
            self.openai_embedding_endpoint = config["openai_embeddings_endpoint"]
        if "openai_api_version" in config:
            self.openai_api_version = config["openai_api_version"]
        if "openai_embeddings_api_version" in config:
            self.openai_embedding_api_version = config["openai_embeddings_api_version"]
        if "output_directory" in config:
            self.output_directory = config["output_directory"]
        if "dbname" in config:
            self.db_name = config["dbname"]
        if "user" in config:
            self.db_user = config["user"]
        if "password" in config:
            self.db_password = config["password"]
        if "host" in config:
            self.db_host = config["host"]
        if "port" in config:
            self.db_port = config["port"]
        if "neo4j_uri" in config:
            self.neo4j_uri = config["neo4j_uri"]
        if "neo4j_user" in config:
            self.neo4j_user = config["neo4j_user"]
        if "neo4j_password" in config:
            self.neo4j_password = config["neo4j_password"]
        if "docling_url" in config:
            self.docling_url = config["docling_url"]
        if "document_manager_url" in config:
            self.document_manager_url = config["document_manager_url"]

config = Config()
