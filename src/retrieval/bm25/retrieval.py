import nltk
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def tokenize(text):
    """
    Tokenizes text using NLTK, handling potential setup errors.
    """
    try:
        return nltk.word_tokenize(text.lower())
    except LookupError:
        raise RuntimeError(
            "NLTK 'punkt' resource not found. Please run 'python -m src.ingest prepare' from the project root to download it."
        )

def build_bm25(corpus):
    """
    Builds the BM25 index from the corpus.
    """
    doc_ids = list(corpus.keys())
    docs = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in doc_ids]
    
    tokenized_corpus = [tokenize(doc) for doc in tqdm(docs, desc="Tokenizing corpus")]
    
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, doc_ids

def retrieve_bm25_results(bm25, doc_ids, queries, top_k=10):
    """
    Retrieves BM25 results for all queries.
    """
    results = {}
    for qid, query_text in tqdm(queries.items(), total=len(queries), desc="Processing queries"):
        tokenized_query = tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)
        
        # Create (doc_id, score) tuples and sort
        scored_docs = sorted(
            zip(doc_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top_k results
        top_n = scored_docs[:top_k]
        
        # Store as dict {doc_id: score}
        results[qid] = {doc_id: float(score) for doc_id, score in top_n}
        
    return results

def run_bm25_baseline(corpus, queries, top_k=10):
    """
    Runs the full BM25 baseline: build index and retrieve results.
    """
    print("Building BM25 index...")
    bm25, doc_ids = build_bm25(corpus)

    print("Retrieving results...")
    results = retrieve_bm25_results(bm25, doc_ids, queries, top_k=top_k)

    return results