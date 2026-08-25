import sys
import os
import argparse
import asyncio
import json
import traceback

pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--mode")
pre_args, _ = pre_parser.parse_known_args()
if pre_args.mode:
    os.environ["RAGIFY_MODE"] = pre_args.mode.lower()
else:
    os.environ.setdefault("RAGIFY_MODE", "pilot")

# Setup path to include project root (one level up from scripts/) globally
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import backend now
try:
    from app.services import rag_service, clients
except ImportError as e:
    print(f"Error importing app modules: {e}")
    sys.exit(1)

# Function to run the demo
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--mode", help="demo|pilot|prod|dev")
    parser.add_argument("--questions", default="demo/demo_questions.json", help="Path to questions JSON")
    args = parser.parse_args()

    if args.mode:
        print(f"Using RAGIFY_MODE={os.environ['RAGIFY_MODE']}")

    # Load questions
    questions_path = os.path.join(project_root, args.questions)
    if not os.path.exists(questions_path):
        print(f"Error: Questions file not found at {questions_path}")
        sys.exit(1)
        
    with open(questions_path, "r") as f:
        questions = json.load(f)

    if args.max_questions:
        questions = questions[:args.max_questions]

    print(f"Running {len(questions)} questions for tenant '{args.tenant}'...\n")

    await clients.initialize_http_client()
    clients.initialize_chroma_client()

    try:
        for q_data in questions:
            q_id = q_data.get("id", "unknown")
            q_text = q_data.get("question")
            notes = q_data.get("notes", "")

            print(f"\n=== {q_id} ===")
            print(f"QUESTION: {q_text}")
            
            try:
                # Invoke RAG Service
                result = await rag_service.query_collection(
                    tenant_id=args.tenant,
                    question=q_text,
                    top_k=4,
                    debug=1 
                )
                
                # Unpack result
                answer_gen, source_files, evidence_items, context_text, debug_info = result

                # Collect answer text
                answer_chunks = []
                async for chunk in answer_gen:
                    answer_chunks.append(chunk)
                full_answer = "".join(answer_chunks)

                # Determine Decision
                decision = "ANSWER"
                if debug_info and isinstance(debug_info, dict) and debug_info.get("refused"):
                    decision = "REFUSE"
                    if debug_info.get("refusal_reason"):
                        full_answer = f"[Refused: {debug_info['refusal_reason']}] {full_answer}"
                elif "clarif" in full_answer.lower() and len(full_answer) < 300:
                    decision = "CLARIFY"
                
                print(f"DECISION: {decision}")
                print(f"ANSWER: {full_answer.strip()}")
                
                print("SOURCES:", end="")
                # Check debug_info for selected chunks
                sources_printed = False
                if debug_info and isinstance(debug_info, dict) and "selected_chunks" in debug_info:
                    # Dedupe sources
                    seen = set()
                    for ch in debug_info["selected_chunks"]:
                         # chunk is dict with source_file, chunk_id
                         fname = ch.get("source_file", "unknown")
                         cid = ch.get("chunk_id", "?")
                         
                         # Clean up filename for display
                         if os.path.sep in fname:
                            fname = os.path.basename(fname)
                            
                         key = f"{fname}_{cid}"
                         if key not in seen:
                             print(f"\n  - {fname} (Chunk {cid})", end="")
                             seen.add(key)
                             sources_printed = True
                
                if not sources_printed and source_files:
                     for s in set(source_files):
                         fname = os.path.basename(s)
                         print(f"\n  - {fname} (File match)", end="")
                
                print(f"\nNOTES: {notes}")
                print("-" * 40)

            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
    
    finally:
        await clients.close_http_client()

if __name__ == "__main__":
    asyncio.run(main())
