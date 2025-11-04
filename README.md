# Domain-Specific Query Expansion with LLMs

## Members
  - Ahmed Badis Lakrach
  - B. Kaan Özkan
  - Rami Baffoun
  - Rami Kallel

## Project Setup

### 1. Clone the repository
```
git clone git@gitlab.informatik.uni-bonn.de:lab-information-retrieval/domain-specific-query-expansion-with-llms.git
cd domain-specific-query-expansion-with-llms
```

### 2. Create a virtual environment
- Linux / macOS
    ```
    python3 -m venv .venv
    ```
- Windows (Git Bash)
    ```
    python -m venv .venv
    ```

### 3. Activate the virtual environment
- Linux / macOS
    ```
    source .venv/bin/activate
    ```
- Windows (Git Bash)
    ```
    source .venv/Scripts/activate
    ```

### 4. Install dependencies
```
pip install -r requirements.txt
```

### 5. Deactivate when done
```
deactivate
```

### Notes:
 - The venv/ folder is ignored by Git (see .gitignore), so each developer creates their own local environment.
 - If you add new dependencies, run:
    ```
    pip freeze > requirements.txt
    ```
    and commit the updated requirements.txt.
