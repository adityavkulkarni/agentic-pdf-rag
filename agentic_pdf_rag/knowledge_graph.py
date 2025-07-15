from .neo4j_client import Neo4jClient


class KnowledgeGraph(Neo4jClient):
    def __init__(self, **kwargs):
        Neo4jClient.__init__(self, **kwargs)

    def add_cda(self, filename, summary=""):
        pass

    def add_amendment(self, filename, cda_name, summary=""):
        pass

    def update_amendment_summary(self):
        pass

    def get_cda(self, cda_name):
        pass

    def get_amendments(self, cda_name, level=None):
        pass

    def get_amendments_summary(self, cda_name):
        pass
