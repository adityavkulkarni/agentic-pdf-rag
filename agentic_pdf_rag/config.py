import os
import logging
import configparser

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
            self.embedding_model = models.get('embedding_model')
        except KeyError as e:
            logger.error(f"Missing 'models' section or key: {e}")
            raise

        try:
            azure_openai = config['azure_openai']
            self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", None)
            if self.openai_api_key:
                logger.info("OpenAI API key loaded from  environment variable.")
            else:
                logger.error("OpenAI API key not found in arguments or environment variables.")
                raise Exception("OpenAI API key not found in arguments or environment variables.")
            self.openai_embeddings_api_key = os.getenv("AZURE_OPENAI_API_KEY_EAST2", None)
            if self.openai_embeddings_api_key:
                logger.info("OpenAI API key loaded from  environment variable.")
            else:
                logger.error("OpenAI API key not found in arguments or environment variables.")
                raise Exception("OpenAI API key not found in arguments or environment variables.")
            self.openai_endpoint = azure_openai.get('openai_endpoint')
            self.openai_embeddings_endpoint = azure_openai.get('openai_embeddings_endpoint')
            self.openai_api_version = azure_openai.get('openai_api_version')
            self.openai_embeddings_api_version = azure_openai.get('openai_embeddings_api_version')
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
            self.dbname = database.get('dbname')
            self.user = database.get('user')
            self.password = database.get('password')
            self.host = database.get('host')
            self.port = database.get('port')
        except KeyError as e:
            logger.error(f"Missing 'database' section or key: {e}")
            raise
        logger.info(f"Loaded config file: {'\n'.join([f'{k}: {v}' for k, v in self.to_dict().items()])}")

    def to_dict(self):
        return {
            "agentic_pdf_parser_model": self.agentic_pdf_parser_model,
            "agentic_chunker_model": self.agentic_chunker_model,
            "embedding_model": self.embedding_model,
            "openai_api_key": "<REDACTED>",
            "openai_embeddings_api_key": "<REDACTED>",
            "openai_endpoint": self.openai_endpoint,
            "openai_embeddings_endpoint": self.openai_embeddings_endpoint,
            "openai_api_version": self.openai_api_version,
            "openai_embeddings_api_version": self.openai_embeddings_api_version,
            "output_directory": self.output_directory,
            "dbname": self.dbname,
            "user": self.user,
            "password": "<REDACTED>",
            "host": self.host,
            "port": self.port,
        }