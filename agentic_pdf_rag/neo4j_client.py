from datetime import date, datetime

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
            return session.execute_read(self._fetch_nodes_tx, label, filters)

    @staticmethod
    def _fetch_nodes_tx(tx, label, filters):
        filters = filters or {}
        filter_str = ' AND '.join([f"n.{k} = ${k}" for k in filters.keys()])
        query = f"MATCH (n:{label})"
        if filter_str:
            query += f" WHERE {filter_str}"
        query += " RETURN n"
        result = tx.run(query, **filters)
        return [record[0] for record in result]

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

    def fetch_graph(self):
        with self.driver.session() as session:
            return session.execute_read(self._fetch_graph_tx)

    def get_graph_dict(self):
        with self.driver.session() as session:
            nodes = []
            edges = []
            for record in session.execute_read(self._fetch_graph_tx):
                n = record["n"]
                m = record["m"]
                r = record["r"]
                if n.id not in [node["data"]["id"] for node in nodes]:
                    # nodes.append(Node(id=str(n.element_id), label=f'{str(""list(n.labels))}: {n._properties["id"]}'))
                    nodes.append({"data": {"label": str(list(n.labels)[0]), **n._properties}})
                if m.id not in [node["data"]["id"] for node in nodes]:
                    nodes.append({"data": {"label": str(list(m.labels)[0]), **m._properties}})
                # edges.append(Edge(source=str(n.element_id), target=str(m.element_id)))
                edges.append({"data": {"id": r.element_id, "label": r.type, "source": n._properties["id"],
                                       "target": m._properties["id"]}})
            return {
                "nodes": nodes,
                "edges": edges,
            }

    @staticmethod
    def _fetch_graph_tx(tx):
        return list(tx.run("MATCH (n)-[r]->(m) RETURN n, r, m"))

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
        '''# Create CDA node
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
        )'''
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

        '''with client.driver.session() as session:
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
        )'''
        """cda_node = client.create_node(
            label="CDA",
            properties={
                "id": 14,
                "name": "CDA #EP Consolidated_0_0_Original_Recorded.pdf",
                "summary": '''The document titled "University EP Energy Consolidated Drilling and Development Unit Agreement" outlines a comprehensive agreement involving the State of Texas, the Board for Lease of University Lands, and EP Energy E&P Company, L.P. The agreement consolidates multiple oil and gas leases across Crockett, Reagan, Irion, and Upton Counties in Texas. It includes several Development Unit Agreements, specifically numbered 2589, 2615, 2618, 2623, and 2624, which were amended to optimize the use of state lands for oil and gas production. The document details the terms for drilling obligations, productive acreage, and the assignment of interests. It also specifies the conditions under which the agreement can be terminated or amended. The agreement is effective from February 1, 2013, and includes provisions for continuous drilling obligations and the assignment of interests, requiring state consent for any transfer. The document is signed by representatives from EP Energy and the State of Texas, including Richard H. Little and Jerry E. Patterson, with notarization by Lisa C. Belue. The agreement is recorded in various deed records across the involved counties, with specific book and page references provided for each lease.''',
                "date": "February 1, 2013"
            }
        )
        print(cda_node)"""
        # print(client.fetch_nodes(label="CDA"))
        amendment1 = client.create_node(
            label="Amendment",
            properties={
                "id": 15,
                "name": "CDA #EP Consolidated_1_0_1st Amendment.pdf",
                "summary": '''''',
                "date": "February 1, 2013"
            }
        )
        # client.delete_node(label="CDA", node_id="14")
        relationships = client.fetch_graph()
        print(f"\nRelationships in the database: {relationships}")
        #for a, rel_type, b in relationships:
        #    print(f"({a['labels']} {{id: {a['id']}}}) -[:{rel_type}]-> ({b['labels']} {{id: {b['id']}}})")
        for record in relationships:
            n = record["n"]
            m = record["m"]
            r = record["r"]
            print(n.labels)
            print(m.labels)
            print(r.type)
    finally:
        client.close()

        # 2589