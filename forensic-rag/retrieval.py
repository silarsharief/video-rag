from database import ForensicDB
import google.generativeai as genai
import os
import json

class ForensicSearch:
    def __init__(self):
        self.db = ForensicDB()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # OLD: self.model = genai.GenerativeModel('gemini-1.5-flash')
        # NEW:
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def search(self, user_query):
        # Step 1: Broad Search (Get Top 10 candidates)
        results = self.db.vector_col.query(
            query_texts=[user_query],
            n_results=10 
        )
        
        candidates = []
        scene_ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        
        # Build Candidate List
        with self.db.driver.session() as session:
            for i, sid in enumerate(scene_ids):
                # Fetch Graph Context
                cypher = """
                MATCH (s:Scene {id: $sid})
                OPTIONAL MATCH (p:Person)-[:APPEARS_IN]->(s)
                WITH s, collect(p.id) as person_list
                RETURN person_list as persons
                """
                graph_data = session.run(cypher, sid=sid).single()
                
                candidate = {
                    "id": i, # Index for filtering
                    "time": f"{metadatas[i]['start']:.1f}s - {metadatas[i]['end']:.1f}s",
                    "description": documents[i],
                    "video": metadatas[i]['video'],
                    "persons": graph_data['persons'] if graph_data else []
                }
                candidates.append(candidate)

        # Step 2: The "Smart Filter" (Gemini decides what is relevant)
        filtered_evidence = self._smart_filter_candidates(user_query, candidates)
        
        # Step 3: Synthesis (Final Answer based ONLY on filtered evidence)
        if not filtered_evidence:
            return "No relevant video footage found.", []

        prompt = f"""
        User Query: "{user_query}"
        
        Relevant Video Evidence:
        {filtered_evidence}
        
        Task: Synthesize a final report based ONLY on the evidence above.
        """
        response = self.model.generate_content(prompt)
        return response.text, filtered_evidence

    def _smart_filter_candidates(self, query, candidates):
        """
        Asks Gemini to pick which video clips actually match the user's request.
        """
        prompt = f"""
        User Query: "{query}"
        
        I have retrieved {len(candidates)} potential video clips. 
        Your job is to act as a Quality Filter.
        
        Candidate Clips:
        {json.dumps([{ 'id': c['id'], 'desc': c['description'] } for c in candidates], indent=2)}
        
        Task: Return a JSON list of the IDs (integers) that are TRULY relevant to the query.
        - If the query is "accident" and only ID 0 describes a crash, return [0].
        - If 5 clips show the crash, return [0, 1, 2, 3, 4].
        - If none are relevant, return [].
        
        Output JSON only:
        """
        
        try:
            response = self.model.generate_content(prompt)
            # clean json
            clean_text = response.text.strip().replace("```json", "").replace("```", "")
            relevant_indices = json.loads(clean_text)
            
            # Filter the original list
            final_list = [c for c in candidates if c['id'] in relevant_indices]
            return final_list
            
        except Exception as e:
            print(f"Filter Error: {e}")
            return candidates[:3] # Fallback to Top 3 if filter fails

        """"""
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
    """"""