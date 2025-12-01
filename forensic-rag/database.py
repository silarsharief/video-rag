import os
import certifi
import ssl

# Global SSL Fix for Mac
os.environ['SSL_CERT_FILE'] = certifi.where()

import chromadb
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class ForensicDB:
    def __init__(self):
        # 1. Initialize Vector DB
        self.chroma_client = chromadb.PersistentClient(path="./chromadb")
        self.vector_col = self.chroma_client.get_or_create_collection(name="forensic_scenes")

        # 2. Load Env Variables
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri:
            raise ValueError("❌ ERROR: 'NEO4J_URI' missing.")
        if not user or not password:
            raise ValueError("❌ ERROR: Neo4j credentials missing.")

        # 3. Initialize Graph DB
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    # === UPDATED: Added 'mode' argument ===
    def add_scene_node(self, video_name, mode, scene_id, start_time, end_time, summary, person_ids, object_tags):
        """
        Atomic transaction to update Graph (Timeline) and Vector (Search)
        """
        # A. Vector Store: Save 'mode' in metadata for filtering
        self.vector_col.add(
            documents=[summary],
            metadatas=[{
                "video": video_name, 
                "mode": mode,  # <--- NEW: Allows filtering by Use Case
                "start": start_time, 
                "end": end_time, 
                "scene_id": scene_id
            }],
            ids=[scene_id]
        )

        # B. Graph Store: Pass 'mode' to Cypher
        with self.driver.session() as session:
            session.execute_write(
                self._create_graph_nodes, 
                video_name, mode, scene_id, start_time, end_time, summary, person_ids, object_tags
            )

    @staticmethod
    def _create_graph_nodes(tx, video_name, mode, scene_id, start, end, summary, person_ids, object_tags):
        # 1. Create Video Node with a 'type' property
        tx.run("MERGE (v:Video {name: $name}) SET v.type = $mode", name=video_name, mode=mode)
        
        # 2. Create Scene Node
        query_scene = """
        MATCH (v:Video {name: $video_name})
        MERGE (s:Scene {id: $id})
        SET s.start = $start, 
            s.end = $end, 
            s.summary = $summary,
            s.objects = $objects,
            s.mode = $mode      // <--- NEW: Store mode on Scene too
        MERGE (v)-[:HAS_SEGMENT]->(s)
        """
        tx.run(query_scene, video_name=video_name, mode=mode, id=scene_id, start=start, end=end, summary=summary, objects=object_tags)

        # 3. Create Timeline Link
        query_timeline = """
        MATCH (s:Scene {id: $id})
        MATCH (prev:Scene) WHERE prev.end <= $start AND prev.id <> $id
        WITH s, prev ORDER BY prev.end DESC LIMIT 1
        MERGE (prev)-[:NEXT]->(s)
        """
        tx.run(query_timeline, id=scene_id, start=start)

        # 4. Link Persons
        for pid in person_ids:
            query_person = """
            MATCH (s:Scene {id: $sid})
            MERGE (p:Person {id: $pid})
            MERGE (p)-[:APPEARS_IN]->(s)
            """
            tx.run(query_person, sid=scene_id, pid=pid)