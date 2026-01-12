"""
Forensic Search and Retrieval Module
Queries vector database and generates AI summaries of search results.
"""
import os
import json
import time
import logging
import google.generativeai as genai

# Configure logging
logger = logging.getLogger(__name__)

# Import centralized configurations
from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_SEARCH_DELAY,
    SEARCH_TOP_K,
    SEARCH_FETCH_LIMIT,
    SIMILARITY_THRESHOLD,
    MIN_RESULTS,
    ENABLE_QUERY_CACHE,
    ENABLE_METRICS
)
from config.prompts import get_search_prompt, get_query_rewrite_prompt
from src.core.database import ForensicDB
from src.core.cache import get_query_cache
from src.core.metrics import get_metrics


class ForensicSearch:
    """
    Handles semantic search across forensic video database.
    Combines vector search with graph queries and AI summarization.
    """
    
    def __init__(self):
        """Initialize search engine with database and AI model."""
        self.db = ForensicDB()
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)

    def _rewrite_query(self, query, mode):
        """
        Rewrite and enhance user query for better retrieval accuracy.
        
        This method:
        - Checks cache first to avoid redundant API calls
        - Fixes spelling and grammar errors
        - Adds context-rich terms relevant to the mode
        - Expands abbreviations and makes query more specific
        - Preserves original user intent
        
        Args:
            query (str): Original user query
            mode (str): Mode for context-specific enhancement (traffic, factory, etc.)
            
        Returns:
            str: Enhanced query (or original if rewriting fails)
        """
        # Log original query
        logger.info("="*60)
        logger.info("QUERY REWRITING")
        logger.info(f"Original Query: '{query}'")
        logger.info(f"Mode: {mode}")
        print(f"\n{'='*60}")
        print(f"🔍 QUERY REWRITING")
        print(f"{'='*60}")
        print(f"📝 Original Query: '{query}'")
        print(f"🏷️  Mode: {mode}")
        
        # Check cache first
        if ENABLE_QUERY_CACHE:
            query_cache = get_query_cache()
            cached_result = query_cache.get_rewritten_query(query, mode)
            if cached_result is not None:
                if ENABLE_METRICS:
                    get_metrics().record_cache_access(hit=True)
                print(f"💾 Using cached query (cache hit)")
                print(f"✅ Enhanced Query: '{cached_result}'")
                print(f"{'='*60}\n")
                return cached_result
            else:
                if ENABLE_METRICS:
                    get_metrics().record_cache_access(hit=False)
        
        try:
            # Rate limiting: Wait before calling Gemini
            time.sleep(GEMINI_SEARCH_DELAY)
            
            # Get mode-specific rewrite prompt
            rewrite_prompt_template = get_query_rewrite_prompt(mode)
            
            # Format prompt with original query
            prompt = rewrite_prompt_template.format(query=query)
            
            # Generate enhanced query using AI
            logger.info("Calling Gemini API for query enhancement...")
            print(f"⏳ Calling Gemini API for query enhancement...")
            response = self.model.generate_content(prompt)
            enhanced_query = response.text.strip()
            
            # Basic validation: ensure we got a meaningful response
            if enhanced_query and len(enhanced_query) > 0:
                logger.info(f"Enhanced Query: '{enhanced_query}'")
                logger.info("Query rewriting successful")
                print(f"✅ Enhanced Query: '{enhanced_query}'")
                print(f"{'='*60}\n")
                
                # Cache the successful rewrite
                if ENABLE_QUERY_CACHE:
                    query_cache = get_query_cache()
                    query_cache.put_rewritten_query(query, mode, enhanced_query)
                
                return enhanced_query
            else:
                # Fallback to original query if response is empty
                logger.warning("Empty response from AI. Using original query.")
                print(f"⚠️  Empty response from AI. Using original query.")
                print(f"{'='*60}\n")
                return query
                
        except Exception as e:
            # If rewriting fails, return original query
            logger.error(f"Query rewriting failed: {e}. Using original query.")
            print(f"❌ Query rewriting failed: {e}")
            print(f"🔄 Using original query: '{query}'")
            print(f"{'='*60}\n")
            return query

    def search(self, query, mode_filter=None):
        """
        Search for relevant video scenes based on query.
        
        Args:
            query (str): Natural language search query
            mode_filter (str, optional): Filter by mode (traffic, factory, kitchen, general)
            
        Returns:
            tuple: (AI summary string, list of evidence dictionaries)
        """
        # Step 1: Rewrite query for better retrieval accuracy
        # Determine mode for query enhancement (use filter or default to 'general')
        rewrite_mode = mode_filter if mode_filter else 'general'
        enhanced_query = self._rewrite_query(query, rewrite_mode)
        
        # Build filter clause for mode-specific search
        where_clause = {"mode": mode_filter} if mode_filter else None
        
        # Step 2: Query vector database using enhanced query
        logger.info("="*60)
        logger.info("VECTOR DATABASE SEARCH")
        logger.info(f"Using Enhanced Query: '{enhanced_query}'")
        logger.info(f"Search Filter: {where_clause}")
        logger.info(f"Fetch Limit: {SEARCH_FETCH_LIMIT}, Top K: {SEARCH_TOP_K}, Threshold: {SIMILARITY_THRESHOLD}")
        print(f"\n{'='*60}")
        print(f"🔎 VECTOR DATABASE SEARCH")
        print(f"{'='*60}")
        print(f"📊 Using Enhanced Query: '{enhanced_query}'")
        print(f"🎯 Filter: {where_clause if where_clause else 'None (searching all modes)'}")
        print(f"📈 Fetch Limit: {SEARCH_FETCH_LIMIT} | Top K: {SEARCH_TOP_K} | Threshold: {SIMILARITY_THRESHOLD}")
        print(f"⏳ Querying vector database...")
        
        # Fetch more results, then filter by threshold
        results = self.db.vector_col.query(
            query_texts=[enhanced_query], 
            n_results=SEARCH_FETCH_LIMIT, 
            where=where_clause,
            include=["documents", "metadatas", "distances"]  # Include distances for filtering
        )
        
        raw_count = len(results['ids'][0]) if results['ids'] and results['ids'][0] else 0
        logger.info(f"Raw results from vector DB: {raw_count}")
        print(f"📥 Raw results from vector DB: {raw_count}")
        
        if not results['ids'] or not results['ids'][0]:
            return "No data found.", []
        
        # Step 3: Filter results by similarity threshold
        # ChromaDB distance: lower = more similar (0.0 = perfect match)
        filtered_indices = []
        distances = results.get('distances', [[]])[0]
        
        for i, dist in enumerate(distances):
            if dist <= SIMILARITY_THRESHOLD:
                filtered_indices.append(i)
        
        # Ensure we return at least MIN_RESULTS (even if threshold not met)
        if len(filtered_indices) < MIN_RESULTS and raw_count > 0:
            sorted_indices = sorted(range(len(distances)), key=lambda x: distances[x])
            filtered_indices = sorted_indices[:MIN_RESULTS]
            logger.warning(f"Threshold not met. Returning best {MIN_RESULTS} results regardless.")
            print(f"⚠️  Threshold not met. Returning best {MIN_RESULTS} results regardless.")
        
        # Limit to top K
        filtered_indices = filtered_indices[:SEARCH_TOP_K]
        
        logger.info(f"After filtering: {len(filtered_indices)} results (threshold: {SIMILARITY_THRESHOLD})")
        print(f"✅ After filtering: {len(filtered_indices)} results pass threshold")
        
        # Log distance scores for debugging
        for idx in filtered_indices:
            print(f"   📍 Result {idx}: distance={distances[idx]:.4f}")
        
        print(f"{'='*60}\n")
        
        # Enrich results with graph data (only for filtered indices)
        candidates = []
        with self.db.driver.session() as session:
            for idx in filtered_indices:
                sid = results['ids'][0][idx]
                # Get additional data from Neo4j graph
                res = session.run("""
                    MATCH (s:Scene {id: $sid}) 
                    OPTIONAL MATCH (p:Person)-[:APPEARS_IN]->(s) 
                    RETURN s.objects, collect(p.id)
                """, sid=sid).single()
                
                candidates.append({
                    "id": idx,
                    "description": results['documents'][0][idx], 
                    "video": results['metadatas'][0][idx]['video_name'],
                    "time": f"{results['metadatas'][0][idx]['start_time']:.1f}s - {results['metadatas'][0][idx]['end_time']:.1f}s",
                    "start_time": results['metadatas'][0][idx]['start_time'],  # For video playback
                    "end_time": results['metadatas'][0][idx]['end_time'],  # For video playback
                    "mode": results['metadatas'][0][idx].get('mode', 'unknown'),
                    "distance": distances[idx],  # Include distance score
                    "yolo_tags": res[0] if res and res[0] else [],
                    "persons": res[1] if res and res[1] else []
                })

        # Use AI to filter most relevant results
        final_candidates = self._smart_filter_safe(query, candidates)
        
        # Generate AI summary of results
        try:
            # Rate limiting: Wait before calling Gemini
            time.sleep(GEMINI_SEARCH_DELAY)
            
            # Determine mode from candidates (use most common mode, or first candidate's mode)
            mode = mode_filter
            if not mode and final_candidates:
                # Get mode from first candidate if not specified
                mode = final_candidates[0].get('mode', 'general')
            
            # Get appropriate search prompt for the mode
            search_prompt_template = get_search_prompt(mode)
            
            # Format evidence for prompt (include video name, time, and description)
            evidence_text = json.dumps([
                {
                    "video": c.get('video', 'Unknown'),
                    "time": c.get('time', 'N/A'),
                    "description": c.get('description', ''),
                    "yolo_tags": c.get('yolo_tags', []),
                    "persons": c.get('persons', [])
                }
                for c in final_candidates
            ], indent=2)
            
            # Format the prompt with query and evidence
            prompt = search_prompt_template.format(
                query=query,
                evidence=evidence_text
            )
            
            response = self.model.generate_content(prompt)
            return response.text, final_candidates
        except Exception as e:
            return f"⚠️ AI Summary Unavailable: {e}", final_candidates

    def _smart_filter_safe(self, query, candidates):
        """
        Use AI to filter most relevant candidates.
        
        Args:
            query (str): User's search query
            candidates (list): List of candidate scenes
            
        Returns:
            list: Filtered list of most relevant candidates
        """
        # Prepare simplified data for AI filtering
        prompt = f"Query: {query}\nClips: {json.dumps([{'id': c['id'], 'desc': c['description']} for c in candidates])}\nReturn IDs list."
        
        try:
            # Rate limiting
            time.sleep(GEMINI_SEARCH_DELAY)
            response = self.model.generate_content(prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            relevant_ids = json.loads(clean_text)
            return [c for c in candidates if c['id'] in relevant_ids]
        except Exception:
            # Fallback: return all candidates if AI filtering fails
            return candidates

