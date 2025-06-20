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
        logger.info(f"Loaded config file: {'\n'.join([f'{k}: {v}' for k, v in self.to_dict().items()])}")

    def to_dict(self):
        return {
            "agentic_pdf_parser_model": self.agentic_pdf_parser_model,
            "agentic_chunker_model": self.agentic_chunker_model,
            "openai_embedding_model": self.openai_embedding_model,
            "openai_api_key": self.openai_api_key,# "<REDACTED>",
            "openai_embeddings_api_key": self.openai_embedding_api_key,#"<REDACTED>",
            "openai_endpoint": self.openai_endpoint,
            "openai_embeddings_endpoint": self.openai_embedding_endpoint,
            "openai_api_version": self.openai_api_version,
            "openai_embeddings_api_version": self.openai_embedding_api_version,
            "output_directory": self.output_directory,
            "db_name": self.db_name,
            "db_user": self.db_user,
            "db_password": self.db_password,#"<REDACTED>",
            "db_host": self.db_host,
            "db_port": self.db_port,
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

config = Config()
