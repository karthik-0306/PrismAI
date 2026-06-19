import csv

with open('backend/evaluation/rewriter_eval_results.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    failures = [r for r in list(reader) if not r['status'].startswith('PASS')]

with open('backend/evaluation/failures.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total Failures: {len(failures)}\n\n")
    for r in failures:
        out.write(f"Sim: {r['similarity_score']} | Status: {r['status']} | Reduction: {r['reduction_pct']}%\n")
        out.write(f"Orig ({r['original_tokens']}): {r['original_query']}\n")
        out.write(f"Cand ({r['rewritten_tokens']}): {r['candidate_query']}\n")
        out.write("-" * 80 + "\n")
