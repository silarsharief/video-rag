from database import ForensicDB
import google.generativeai as genai
import os
import json
import time

class ForensicSearch:
    def __init__(self):
        self.db = ForensicDB()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Use the stable model with 5 req/min limit
        self.model = genai.GenerativeModel('gemini-flash-latest')

    def search(self, query, mode_filter=None):
        where_clause = {"mode": mode_filter} if mode_filter else None
        results = self.db.vector_col.query(query_texts=[query], n_results=10, where=where_clause)
        
        if not results['ids']: return "No data found.", []
        
        candidates = []
        with self.db.driver.session() as session:
            for i, sid in enumerate(results['ids'][0]):
                res = session.run("""
                    MATCH (s:Scene {id: $sid}) 
                    OPTIONAL MATCH (p:Person)-[:APPEARS_IN]->(s) 
                    RETURN s.objects, collect(p.id)
                """, sid=sid).single()
                
                candidates.append({
                    "id": i,
                    # === FIX: Renamed 'desc' to 'description' to match app.py ===
                    "description": results['documents'][0][i], 
                    "video": results['metadatas'][0][i]['video_name'],
                    "time": f"{results['metadatas'][0][i]['start_time']:.1f}s",
                    "mode": results['metadatas'][0][i].get('mode', 'unknown'),
                    "yolo_tags": res[0] if res and res[0] else [],
                    "persons": res[1] if res and res[1] else []
                })

        final_candidates = self._smart_filter_safe(query, candidates)
        
        try:
            # Wait 15s to prevent quota crash
            time.sleep(15)
            prompt = f"User Query: {query}\nEvidence: {final_candidates}\nSummarize."
            response = self.model.generate_content(prompt)
            return response.text, final_candidates
        except Exception as e:
            return f"⚠️ AI Summary Unavailable: {e}", final_candidates

    def _smart_filter_safe(self, query, candidates):
        # Update prompt to use 'description'
        prompt = f"Query: {query}\nClips: {json.dumps([{'id': c['id'], 'desc': c['description']} for c in candidates])}\nReturn IDs list."
        try:
            time.sleep(15) 
            response = self.model.generate_content(prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            relevant_ids = json.loads(clean_text)
            return [c for c in candidates if c['id'] in relevant_ids]
        except Exception:
            return candidates