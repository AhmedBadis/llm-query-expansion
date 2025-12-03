"""
Script to run all evaluation notebooks and verify they execute successfully.
"""
import sys
import subprocess
from pathlib import Path

def run_notebook(notebook_path: Path) -> bool:
    """Execute a Jupyter notebook and return True if successful."""
    print(f"\n{'='*60}")
    print(f"Running: {notebook_path}")
    print('='*60)
    
    try:
        # Use jupyter nbconvert to execute the notebook
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                str(notebook_path)
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per notebook
        )
        
        if result.returncode == 0:
            print(f"[OK] Successfully executed: {notebook_path.name}")
            return True
        else:
            print(f"[FAIL] Failed to execute: {notebook_path.name}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"[FAIL] Timeout executing: {notebook_path.name}")
        return False
    except Exception as e:
        print(f"[FAIL] Error executing {notebook_path.name}: {e}")
        return False

def main():
    """Run all evaluation notebooks."""
    project_root = Path.cwd()
    
    # List of notebooks to run
    notebooks = [
        project_root / "runner" / "test.ipynb",
        project_root / "runner" / "eval" / "baseline.ipynb",
        project_root / "runner" / "eval" / "append.ipynb",
        project_root / "runner" / "eval" / "reformulate.ipynb",
        project_root / "runner" / "eval" / "agr.ipynb",
    ]
    
    results = []
    for notebook in notebooks:
        if not notebook.exists():
            print(f"[FAIL] Notebook not found: {notebook}")
            results.append(False)
        else:
            success = run_notebook(notebook)
            results.append(success)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for notebook, success in zip(notebooks, results):
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {notebook.name}")
    
    all_passed = all(results)
    if all_passed:
        print("\n[OK] All notebooks executed successfully!")
        return 0
    else:
        print("\n[FAIL] Some notebooks failed to execute.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

