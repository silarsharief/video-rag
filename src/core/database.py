"""
Database connections for Forensic RAG System
Handles ChromaDB (vector store) and Neo4j (graph database) connections.
"""
import os
import chromadb
from neo4j import GraphDatabase

# Import centralized settings
from config.settings import (
    CHROMADB_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD
)


class ForensicDB:
    """
    Manages connections to both vector and graph databases.
    Provides methods to store and link video scene data.
    """
    
    def __init__(self):
        # 1. Initialize Vector DB
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
        # Ensure cosine similarity is used for better text matching
        self.vector_col = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": CHROMA_DISTANCE_METRIC} 
        )

        # 2. Validate Env Variables
        if not NEO4J_URI:
            raise ValueError("❌ ERROR: 'NEO4J_URI' missing in .env")
        if not NEO4J_USERNAME or not NEO4J_PASSWORD:
            raise ValueError("❌ ERROR: Neo4j credentials missing in .env")

        # 3. Initialize Graph DB
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.driver.verify_connectivity()

    def close(self):
        """Close database connections."""
        self.driver.close()

    def add_scene_node(self, video_name, mode, scene_id, start_time, end_time, summary, person_ids, object_tags, ppe_detections=None):
        """
        Atomic transaction to update Graph (Timeline) and Vector (Search).
        
        Args:
            video_name (str): Name of the video file
            mode (str): Processing mode (traffic, factory, kitchen, general)
            scene_id (str): Unique scene identifier
            start_time (float): Scene start time in seconds
            end_time (float): Scene end time in seconds
            summary (str): AI-generated scene description
            person_ids (list): List of detected person IDs
            object_tags (list): List of detected object tags
            ppe_detections (dict, optional): Detailed PPE detection results for factory mode
        """
        # Prepare metadata with PPE detection results
        metadata = {
            "video_name": video_name, 
            "mode": mode,
            "start_time": start_time, 
            "end_time": end_time, 
            "scene_id": scene_id,
            "person_count": len(person_ids),
            "objects": ", ".join(object_tags) if object_tags else "none"
        }
        
        # Add PPE detection results for factory mode
        if ppe_detections and mode == "factory":
            # Format PPE detections for storage
            compliance = []
            violations = []
            for obj_name, det_data in ppe_detections.items():
                count = det_data['count']
                avg_conf = sum(det_data['confidences']) / len(det_data['confidences']) if det_data['confidences'] else 0
                if obj_name.startswith('NO-'):
                    violations.append(f"{obj_name}({count})")
                elif obj_name in ['Hardhat', 'Mask', 'Safety Vest', 'Safety Cone']:
                    compliance.append(f"{obj_name}({count})")
            
            metadata["ppe_compliance"] = ", ".join(compliance) if compliance else "none"
            metadata["ppe_violations"] = ", ".join(violations) if violations else "none"
            metadata["ppe_detection_count"] = sum(d['count'] for d in ppe_detections.values())
        
        # A. Vector Store - for semantic search
        self.vector_col.add(
            documents=[summary],
            metadatas=[metadata],
            ids=[scene_id]
        )

        # B. Graph Store - for temporal relationships
        with self.driver.session() as session:
            session.execute_write(
                self._create_graph_nodes, 
                video_name, mode, scene_id, start_time, end_time, summary, person_ids, object_tags, ppe_detections
            )

    @staticmethod
    def _create_graph_nodes(tx, video_name, mode, scene_id, start, end, summary, person_ids, object_tags, ppe_detections=None):
        """
        Static method for Neo4j transaction.
        Creates video, scene, and person nodes with relationships.
        """
        # 1. Create Video Node
        tx.run("MERGE (v:Video {name: $name}) SET v.type = $mode", name=video_name, mode=mode)
        
        # 2. Create Scene Node with PPE detection results
        if ppe_detections and mode == "factory":
            # Format PPE detections for Neo4j
            compliance = []
            violations = []
            for obj_name, det_data in ppe_detections.items():
                count = det_data['count']
                if obj_name.startswith('NO-'):
                    violations.append(f"{obj_name}({count})")
                elif obj_name in ['Hardhat', 'Mask', 'Safety Vest', 'Safety Cone']:
                    compliance.append(f"{obj_name}({count})")
            
            ppe_compliance_str = ", ".join(compliance) if compliance else "none"
            ppe_violations_str = ", ".join(violations) if violations else "none"
            
            query_scene = """
            MATCH (v:Video {name: $video_name})
            MERGE (s:Scene {id: $id})
            SET s.start = $start, 
                s.end = $end, 
                s.summary = $summary,
                s.objects = $objects,
                s.mode = $mode,
                s.ppe_compliance = $ppe_compliance,
                s.ppe_violations = $ppe_violations
            MERGE (v)-[:HAS_SEGMENT]->(s)
            """
            tx.run(query_scene, video_name=video_name, mode=mode, id=scene_id, start=start, end=end, 
                   summary=summary, objects=object_tags, ppe_compliance=ppe_compliance_str, ppe_violations=ppe_violations_str)
        else:
            query_scene = """
            MATCH (v:Video {name: $video_name})
            MERGE (s:Scene {id: $id})
            SET s.start = $start, 
                s.end = $end, 
                s.summary = $summary,
                s.objects = $objects,
                s.mode = $mode
            MERGE (v)-[:HAS_SEGMENT]->(s)
            """
            tx.run(query_scene, video_name=video_name, mode=mode, id=scene_id, start=start, end=end, summary=summary, objects=object_tags)

        # 3. Create Timeline Link (The "Next" Arrow)
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

