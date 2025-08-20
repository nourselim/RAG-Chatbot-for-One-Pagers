import argparse
import subprocess
import sys
from pathlib import Path
from faiss_service import FaissCandidateSearch
from embed_only import OUT_DIR, EMB_NPY, META_JSONL, pct_from_cos
import os

def run_command(command, description):
    """Run a shell command and print its output."""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error running: {command}")
        print(f"Error output: {result.stderr}")
        sys.exit(1)
    print(f"✅ {description} completed successfully")
    return result

def check_file_exists(file_path):
    """Check if a file exists and is not empty."""
    return file_path.exists() and file_path.stat().st_size > 0

def build_full_pipeline():
    """Run the complete pipeline from scratch."""
    print("🔍 Detecting missing files...")
    
    # Check if we need to run convert_json_to_chunks.py
    chunks_file = Path("data/chunks.jsonl")
    if not check_file_exists(chunks_file):
        print("📦 Creating chunks from employee data...")
        run_command("python convert_json_to_chunks.py", "Converting JSON to chunks")
    
    # Check if we need to run embed_only.py embed
    if not check_file_exists(EMB_NPY) or not check_file_exists(META_JSONL):
        print("🧠 Creating embeddings...")
        run_command("python embed_only.py embed", "Creating embeddings")
    
    # Build FAISS index
    print("🏗️ Building FAISS index...")
    search_engine = FaissCandidateSearch(OUT_DIR)
    search_engine.build_index(EMB_NPY, META_JSONL)
    print("✅ FAISS index built successfully!")
    
    return search_engine

def main():
    parser = argparse.ArgumentParser(description="AI Recruiter Assistant - Find the best candidates for your project.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build the FAISS index (run full pipeline if needed)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for candidates matching a project description')
    search_parser.add_argument('query', type=str, help='The project description or requirements')
    search_parser.add_argument('--top', type=int, default=5, help='Number of top candidates to show (default: 5)')
    
    # Auto command (new) - runs everything automatically
    auto_parser = subparsers.add_parser('auto', help='Automatically run full pipeline and then search')
    auto_parser.add_argument('query', type=str, nargs='?', help='The project description to search for (optional)')
    auto_parser.add_argument('--top', type=int, default=5, help='Number of top candidates to show (default: 5)')
    
    args = parser.parse_args()
    
    # If no command provided, default to 'auto'
    if not args.command:
        args.command = 'auto'
        args.query = input("Please enter your project description: ")
        args.top = 5
    
    if args.command == 'build':
        build_full_pipeline()
        print("🎉 Build process completed!")
        
    elif args.command == 'search':
        search_engine = FaissCandidateSearch(OUT_DIR)
        search_engine.load_index()
        perform_search(search_engine, args.query, args.top)
        
    elif args.command == 'auto':
        # Run full pipeline automatically
        search_engine = build_full_pipeline()
        
        # If query was provided, perform search
        if hasattr(args, 'query') and args.query:
            perform_search(search_engine, args.query, args.top)
        else:
            print("\n🎉 Pipeline completed! You can now search using:")
            print("python main.py search \"your project description\"")
            
    else:
        parser.print_help()

def perform_search(search_engine, query, top_n):
    """Perform and display search results."""
    print(f"\n🔍 Searching for candidates matching: '{query}'\n")
    
    ranked_candidates = search_engine.search(query, top_k=50, pool_size=top_n)
    
    if not ranked_candidates:
        print("❌ No suitable candidates found.")
        return
    
    top_score = ranked_candidates[0][1][0]
    
    print("=" * 100)
    for i, (emp_id, (final_score, cos_score, meta)) in enumerate(ranked_candidates, 1):
        rel_score_pct = (final_score / top_score) * 100 if top_score > 0 else 0
        cos_pct = pct_from_cos(cos_score)
        confidence = "High" if cos_score >= 0.60 else ("Medium" if cos_score >= 0.40 else "Low")
        
        name = meta.get("employee_name") or "Name Not Available"
        title = f" - {meta.get('title')}" if meta.get("title") else ""
        email = f" - {meta.get('email')}" if meta.get("email") else ""
        chunk_source = meta.get("chunk_type", "unknown").title()
        
        print(f"{i}. {name}{title}{email}")
        print(f"   🔍 Match Confidence: {confidence} ({cos_pct:.1f}% cosine similarity)")
        print(f"   📊 Relative Score: {rel_score_pct:.0f}%")
        print(f"   📝 Best Matching Info: [{chunk_source}] {meta.get('text', '')[:120]}...")
        print("-" * 100)

if __name__ == "__main__":
    main()