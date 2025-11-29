from database import ForensicDB
import google.generativeai as genai
import os

class ForensicSearch:
    def __init__(self):
        self.db = ForensicDB()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # OLD: self.model = genai.GenerativeModel('gemini-1.5-flash')
        # NEW:
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def search(self, user_query):
        # Step 1: Vector Search (The "What")
        results = self.db.vector_col.query(
            query_texts=[user_query],
            n_results=5
        )
        
        context_data = []
        scene_ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        
        # Step 2: Graph Expansion (The "Who" & "When")
        with self.db.driver.session() as session:
            for i, sid in enumerate(scene_ids):
                # Query: Find connected persons and adjacent scenes
# UPDATED CYPHER QUERY (Fixes "NoneType" and "Multiple Records" error)
                cypher = """
                MATCH (s:Scene {id: $sid})
                OPTIONAL MATCH (p:Person)-[:APPEARS_IN]->(s)
                WITH s, collect(p.id) as person_list
                OPTIONAL MATCH (prev)-[:NEXT]->(s)-[:NEXT]->(next)
                RETURN person_list as persons, prev.summary as prev_summary, next.summary as next_summary
                """
                graph_data = session.run(cypher, sid=sid).single()
                
                scene_info = {
                    "time": f"{metadatas[i]['start']:.1f}s - {metadatas[i]['end']:.1f}s",
                    "description": results['documents'][0][i],
                    "video": metadatas[i]['video'],
                    "persons_present": graph_data['persons'] if graph_data else [],
                    "context_before": graph_data['prev_summary'] if graph_data else "None",
                    "context_after": graph_data['next_summary'] if graph_data else "None"
                }
                context_data.append(scene_info)

        # Step 3: Synthesis (LLM)
        # NEUTRAL SYNTHESIS PROMPT
        prompt = f"""
        User Query: "{user_query}"
        
        Video Evidence Found:
        {context_data}
        
        Task: Answer the user's question based strictly on the evidence.
        - If asking about time, use the timestamps provided.
        - If asking about objects, count or describe them neutrally.
        - Do not assume a crime occurred unless explicitly visible.
        """
        
        response = self.model.generate_content(prompt)
        return response.text, context_data