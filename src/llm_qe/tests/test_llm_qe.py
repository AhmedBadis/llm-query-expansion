"""
Setup verification script - tests everything WITHOUT loading LLM.
"""
import sys
import os
from pathlib import Path

print("="*60)
print("LLM-QE Setup Test")
print("="*60)

# Test 1: Python version
print(f"\n[1/7] Python version: {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    print("❌ Need Python 3.8+")
    sys.exit(1)
print("✓ Python version OK")

# Test 2: PyTorch
try:
    import torch
    print(f"\n[2/7] PyTorch: {torch.__version__}")
    print(f"       CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"       GPU: {torch.cuda.get_device_name(0)}")
    print("✓ PyTorch installed")
except ImportError:
    print("\n[2/7] ❌ PyTorch not installed")
    print("       Run: pip install torch")
    sys.exit(1)

# Test 3: Transformers
try:
    import transformers
    print(f"\n[3/7] Transformers: {transformers.__version__}")
    print("✓ Transformers installed")
except ImportError:
    print("\n[3/7] ❌ Transformers not installed")
    print("       Run: pip install transformers")
    sys.exit(1)

# Test 4: Project structure
print(f"\n[4/7] Project structure:")
project_root = Path.cwd()
required_dirs = [
    "src/llm_qe",
    "src/retrieval",
    "src/ingest",
    "src/eval"
]

all_exist = True
for dir_path in required_dirs:
    full_path = project_root / dir_path
    exists = full_path.exists()
    status = "✓" if exists else "❌"
    print(f"       {status} {dir_path}")
    if not exists:
        all_exist = False

if not all_exist:
    print("❌ Missing directories. Make sure you're in project root.")
    sys.exit(1)

# Test 5: LLM-QE files
print(f"\n[5/7] LLM-QE files:")
llm_qe_files = [
    "src/llm_qe/__init__.py",
    "src/llm_qe/prompts.py",
    "src/llm_qe/expander.py",
    "src/llm_qe/main.py"
]

all_exist = True
for file_path in llm_qe_files:
    full_path = project_root / file_path
    exists = full_path.exists()
    status = "✓" if exists else "❌"
    print(f"       {status} {file_path}")
    if not exists:
        all_exist = False

if not all_exist:
    print("❌ Missing LLM-QE files.")
    sys.exit(1)

# Test 6: Can import llm_qe
print(f"\n[6/7] Import test:")
try:
    sys.path.insert(0, str(project_root / "src"))
    from llm_qe import ExpansionStrategy
    from llm_qe.prompts import GENERATE_ONLY_PROMPT
    print("       ✓ Can import llm_qe.ExpansionStrategy")
    print("       ✓ Can import llm_qe.prompts")
except ImportError as e:
    print(f"       ❌ Import failed: {e}")
    sys.exit(1)

# Test 7: Team code imports
print(f"\n[7/7] Team code imports:")
try:
    from ingest import load_dataset
    print("       ✓ Can import ingest.load_dataset")
except ImportError:
    print("       ⚠️  Cannot import ingest (might need data setup)")

try:
    from retrieval import run_bm25_baseline
    print("       ✓ Can import retrieval.run_bm25_baseline")
except ImportError:
    print("       ⚠️  Cannot import retrieval")

try:
    from ingest.utils import set_nltk_path
    print("       ✓ Can import ingest.utils.set_nltk_path")
except ImportError:
    print("       ⚠️  Cannot import ingest.utils")

# Final summary
print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
