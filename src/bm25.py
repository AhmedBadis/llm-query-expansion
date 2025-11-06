
# BM25 Baseline on BEIR dataset (scifact)


import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import random
import nltk

# download tokenizers
nltk.download("punkt")
nltk.download("punkt_tab")


# 1. Load BEIR dataset

def load_dataset(dataset_name="scifact", data_dir="datasets"):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = beir.util.download_and_unzip(url, data_dir)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    print(f"Loaded dataset: {dataset_name}")
    print(f"Corpus size: {len(corpus)}, Queries: {len(queries)}")
    return corpus, queries, qrels


# 2. Tokenization helper

def tokenize(text):
    return nltk.word_tokenize(text.lower())



# 3. BM25 Baseline Retrieval

def run_bm25_baseline(corpus, queries, qrels, top_k=10):
    # Build corpus
    doc_ids = list(corpus.keys())
    docs = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in doc_ids]
    tokenized_corpus = [tokenize(doc) for doc in tqdm(docs, desc="Tokenizing corpus")]

    bm25 = BM25Okapi(tokenized_corpus)

    results = {}
    print("Retrieving documents...")
    for qid, query_text in tqdm(queries.items(), total=len(queries)):
        tokenized_query = tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)
        top_n = sorted(
            list(zip(doc_ids, scores)), key=lambda x: x[1], reverse=True
        )[:top_k]
        results[qid] = {doc_id: float(score) for doc_id, score in top_n}

    # Evaluate results
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, [1, 3, 5, 10,100]
    )

    print("\n Evaluation Metrics (BM25 baseline):")
    for k in sorted(ndcg.keys()):
        print(f"{k}: {ndcg[k]:.4f}")
    for k in sorted(_map.keys()):
        print(f"{k}: {_map[k]:.4f}")
    for k in sorted(recall.keys()):
        print(f"{k}: {recall[k]:.4f}")
    for k in sorted(precision.keys()):
        print(f"{k}: {precision[k]:.4f}")


    return results


# 4. Show sample retrieval results

def show_examples(corpus, queries, results, num_samples=3):
    sample_queries = random.sample(list(queries.keys()), num_samples)
    print("\n Example retrieval results:\n")
    for qid in sample_queries:
        query_text = queries[qid]
        print(f"Query: {query_text}\n")
        top_hits = list(results[qid].items())[:3]
        for rank, (doc_id, score) in enumerate(top_hits, 1):
            doc_text = corpus[doc_id]["text"]
            print(f"    Rank {rank} | Score {score:.4f}")
            print(f"   {doc_text[:250]}...\n")
        print("-" * 80)



# 5. Main

if __name__ == "__main__":
    corpus, queries, qrels = load_dataset("scifact")
    results = run_bm25_baseline(corpus, queries, qrels)
    show_examples(corpus, queries, results, num_samples=3)
