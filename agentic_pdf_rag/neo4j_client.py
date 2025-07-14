from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, label, properties):
        with self.driver.session() as session:
            result = session.write_transaction(self._create_node_tx, label, properties)
            return result

    @staticmethod
    def _create_node_tx(tx, label, properties):
        query = f"CREATE (n:{label} $props) RETURN n"
        result = tx.run(query, props=properties)
        return result.single()[0]

    def create_relationship(self, from_label, from_id, to_label, to_id, rel_type, rel_properties=None):
        with self.driver.session() as session:
            result = session.write_transaction(
                self._create_relationship_tx,
                from_label, from_id, to_label, to_id, rel_type, rel_properties
            )
            return result

    @staticmethod
    def _create_relationship_tx(tx, from_label, from_id, to_label, to_id, rel_type, rel_properties):
        rel_properties = rel_properties or {}
        query = (
            f"MATCH (a:{from_label} {{id: $from_id}}), (b:{to_label} {{id: $to_id}}) "
            f"CREATE (a)-[r:{rel_type} $props]->(b) RETURN r"
        )
        result = tx.run(query, from_id=from_id, to_id=to_id, props=rel_properties)
        return result.single()[0]

    def fetch_nodes(self, label, filters=None):
        with self.driver.session() as session:
            result = session.read_transaction(self._fetch_nodes_tx, label, filters)
            return [record[0] for record in result]

    @staticmethod
    def _fetch_nodes_tx(tx, label, filters):
        filters = filters or {}
        filter_str = ' AND '.join([f"n.{k} = ${k}" for k in filters.keys()])
        query = f"MATCH (n:{label})"
        if filter_str:
            query += f" WHERE {filter_str}"
        query += " RETURN n"
        result = tx.run(query, **filters)
        return result
