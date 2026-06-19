"""
backend/evaluation/test_rewriter_quality.py

Standalone evaluation script for the Query Rewriter.
This script tests the rewriter against real, messy human queries from the
LMSYS-Chat-1M dataset on Hugging Face.

It performs the following:
1. Downloads a streaming sample of lmsys-chat-1m.
2. Extracts the first user turn from English conversations.
3. Filters for queries > 15 tokens (mirroring production threshold).
4. Runs the production QueryRewriter on each query.
5. Records token reduction and semantic similarity (vs the 0.85 threshold).
6. Outputs a detailed CSV and a clean summary table to standard output.
"""

import sys
import os
import asyncio
import csv

# Ensure the script can import from the backend directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()  # Load HF_TOKEN and API keys from .env

try:
    from datasets import load_dataset
except ImportError:
    print("Error: The 'datasets' library is required. Please run: pip install datasets")
    sys.exit(1)

from backend.pipeline.rewriter import QueryRewriter
from backend.utils.token_counter import count_tokens

# Number of eligible queries to evaluate.
NUM_SAMPLES = 100

async def run_evaluation():
    print(f"Initializing Query Rewriter evaluation on {NUM_SAMPLES} real queries...")
    rewriter = QueryRewriter()
    
    print("Connecting to Hugging Face to stream lmsys/lmsys-chat-1m dataset...")
    # Using streaming=True so we don't download the entire 1M dataset, just what we need.
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    
    eligible_queries = []
    
    # Collect queries
    for row in dataset:
        # Filter 1: English only
        if row.get("language", "").lower() != "english":
            continue
            
        # Filter 2: First user turn
        conv = row.get("conversation", [])
        if not conv:
            continue
        first_turn = conv[0]
        if first_turn.get("role") != "user":
            continue
            
        content = first_turn.get("content", "").strip()
        if not content:
            continue
            
        # Filter 3: Must be > 15 tokens to trigger rewriter logic
        token_count = count_tokens(content)
        if token_count > 15:
            eligible_queries.append({
                "content": content,
                "original_tokens": token_count
            })
            
        if len(eligible_queries) >= NUM_SAMPLES:
            break
            
    print(f"Successfully collected {len(eligible_queries)} eligible queries.")
    print("-" * 80)
    
    results = []
    passed_count = 0
    total_original_tokens = 0
    total_rewritten_tokens = 0
    
    for i, q_data in enumerate(eligible_queries, 1):
        original_query = q_data["content"]
        original_tk = q_data["original_tokens"]
        
        # We want to capture the candidate even if it fails.
        # But rewriter.py only returns candidate if it passes, otherwise fallback.
        # So we will capture the result and check if fallback was used.
        res = await rewriter.rewrite(original_query)
        
        # RATE LIMIT PROTECTION: Groq has a TPM (Tokens Per Minute) limit.
        # Since our prompt now preserves large text payloads, we are consuming thousands
        # of tokens quickly. We must sleep for 15 seconds to avoid rate limits.
        await asyncio.sleep(15.0)
        
        # If fallback_used is True, the rewriter discarded the candidate.
        # The result similarity_score will tell us why (either < 0.85 or not shorter).
        passed = not res.fallback_used
        if passed:
            passed_count += 1
            
        total_original_tokens += res.original_tokens
        total_rewritten_tokens += res.rewritten_tokens
        
        # Determine status reason
        if passed:
            status = "PASS"
        else:
            if res.similarity_score < 0.85:
                status = "FAIL (Low Similarity)"
            else:
                status = "FAIL (Not Shorter/Other)"
                
        results.append({
            "original_query": original_query,
            "rewritten_query": res.rewritten_query,
            "candidate_query": res.candidate_query,
            "original_tokens": res.original_tokens,
            "rewritten_tokens": res.rewritten_tokens,
            "reduction_pct": res.reduction_pct,
            "similarity_score": res.similarity_score,
            "status": status
        })
        
        print(f"[{i}/{NUM_SAMPLES}] Sim: {res.similarity_score:.4f} | Tokens: {res.original_tokens}->{res.rewritten_tokens} | {status}")

    # Write CSV
    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(output_dir, "rewriter_eval_results.csv")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    # Print Summary
    print("\n" + "="*80)
    print("REWRITER EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Queries Evaluated: {len(results)}")
    print(f"Passed Threshold (>=0.85 & shorter): {passed_count} ({passed_count/len(results)*100:.1f}%)")
    print(f"Failed Threshold: {len(results) - passed_count} ({(len(results) - passed_count)/len(results)*100:.1f}%)")
    print("-" * 80)
    print(f"Total Original Tokens:   {total_original_tokens}")
    print(f"Total Rewritten Tokens:  {total_rewritten_tokens}")
    
    overall_reduction = 0.0
    if total_original_tokens > 0:
        overall_reduction = (1.0 - total_rewritten_tokens / total_original_tokens) * 100.0
    print(f"Aggregate Token Reduction: {overall_reduction:.1f}%")
    print("="*80)
    print(f"Detailed results saved to: {csv_path}\n")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
