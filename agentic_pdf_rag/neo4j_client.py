from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, label, properties):
        with self.driver.session() as session:
            result = session.execute_write(self._create_node_tx, label, properties)
            return result

    @staticmethod
    def _create_node_tx(tx, label, properties):
        query = f"CREATE (n:{label} $props) RETURN n"
        result = tx.run(query, props=properties)
        return result.single()[0]

    def create_relationship(self, from_label, from_id, to_label, to_id, rel_type, rel_properties=None):
        with self.driver.session() as session:
            result = session.execute_write(
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
            result = session.execute_read(self._fetch_nodes_tx, label, filters)
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

    def fetch_relationships(self, from_label=None, rel_type=None, to_label=None):
        with self.driver.session() as session:
            return session.execute_read(self._fetch_relationships_tx, from_label, rel_type, to_label)

    @staticmethod
    def _fetch_relationships_tx(tx, from_label, rel_type, to_label):
        query = "MATCH (a)"
        if from_label:
            query = f"MATCH (a:{from_label})"
        query += "-[r"
        if rel_type:
            query += f":{rel_type}"
        query += "]->(b"
        if to_label:
            query += f":{to_label}"
        query += ") RETURN a, type(r), b"
        result = tx.run(query)
        return [(record["a"], record["type(r)"], record["b"]) for record in result]

    def delete_node(self, label, node_id):
        with self.driver.session() as session:
            session.run(
                f"MATCH (n:{label} {{id: $node_id}}) DETACH DELETE n",
                node_id=node_id
            )

    def delete_relationship(self, from_label, from_id, to_label, to_id, rel_type):
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (a:{from_label} {{id: $from_id}})-[r:{rel_type}]->(b:{to_label} {{id: $to_id}})
                DELETE r
                """,
                from_id=from_id,
                to_id=to_id
            )


if __name__ == "__main__":
    # Connection details
    uri = "bolt://ulhchathamster:7687"
    user = "neo4j"
    password = "contract@iq@123"

    # Initialize Neo4j client
    client = Neo4jClient(uri, user, password)

    try:
        # Create CDA node
        cda_id = str(3107)
        cda_node = client.create_node("CDA", {"id": cda_id, "name": "Sample CDA"})

        # Create two Amendment nodes
        amendment1_id = "3107.1"
        amendment2_id = "3107.2"
        amendment1 = client.create_node("Amendment", {"id": amendment1_id, "description": "First amendment"})
        amendment2 = client.create_node("Amendment", {"id": amendment2_id, "description": "Second amendment"})

        # Attach amendments to CDA node
        client.create_relationship(
            from_label="CDA",
            from_id=cda_id,
            to_label="Amendment",
            to_id=amendment1_id,
            rel_type="HAS_AMENDMENT"
        )
        """client.create_relationship(
            from_label="CDA",
            from_id=cda_id,
            to_label="Amendment",
            to_id=amendment2_id,
            rel_type="HAS_AMENDMENT"
        )

        print("CDA node and two Amendment nodes created and linked successfully.")

        # Delete a node and all its relationships
        client.delete_node("CDA", "3107")

        # Delete a specific relationship
        client.delete_relationship("CDA", "3107", "Amendment", "3107.2", "HAS_AMENDMENT")
        client.delete_relationship("Amendment", "3107.1", "Amendment", "3107.1", "HAS_AMENDMENT")

        client.delete_node("Amendment", "3107.1")
        client.delete_node("Amendment", "3107.2")"""

        with client.driver.session() as session:
            session.run(
                """
                MATCH (c:CDA {id: $cda_id})-[r:HAS_AMENDMENT]->(a:Amendment {id: $amendment1_id})
                DELETE r
                """,
                cda_id=cda_id,
                amendment1_id=amendment1_id
            )

        client.create_relationship(
            from_label="CDA",
            from_id=cda_id,
            to_label="Amendment",
            to_id=amendment2_id,
            rel_type="HAS_AMENDMENT"
        )

        client.create_relationship(
            from_label="Amendment",
            from_id=amendment2_id,
            to_label="Amendment",
            to_id=amendment1_id,
            rel_type="UPDATES"
        )

        relationships = client.fetch_relationships()
        print("\nRelationships in the database:")
        for a, rel_type, b in relationships:
            print(f"({a['labels']} {{id: {a['id']}}}) -[:{rel_type}]-> ({b['labels']} {{id: {b['id']}}})")

    finally:
        client.close()

        # 2589