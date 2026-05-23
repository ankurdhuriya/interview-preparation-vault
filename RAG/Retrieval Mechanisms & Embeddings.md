
The retrieval stage is the backbone of RAG. It determines *what* information the LLM sees. If retrieval fails, generation fails ("Garbage In, Garbage Out"). This section covers how data is represented (embeddings), stored (VectorDBs), and searched (Algorithms & Strategies).

## Embeddings

Embeddings are vector representations of text (or other media) where semantically similar items are mapped to nearby points in a high-dimensional space.

### Dense vs. Sparse Embeddings

| Feature | **Dense Embeddings** | **Sparse Embeddings** |
| :--- | :--- | :--- |
| **Representation** | Fixed-length vector of floating-point numbers (e.g., 768 or 1536 dimensions). Most values are non-zero. | High-dimensional vector where most values are zero. Non-zero values correspond to specific terms/IDs. |
| **Underlying Tech** | Deep Neural Networks (Transformers like BERT, E5, Ada-002). Captures semantic meaning. | Statistical models (BM25, SPLADE, Lexical Search). Captures exact keyword matches. |
| **Strengths** | Semantic matching (synonyms, paraphrasing), concept association. | Exact keyword matching, proper nouns, technical codes, rare terms. |
| **Weaknesses** | May miss exact string matches; "vector collapse" for very short texts. | Fails on synonyms or semantic intent; sensitive to vocabulary mismatch. |
| **Use Case** | General semantic search, question answering. | Legal docs, code search, product SKUs, hybrid search component. |

### Embedding Model Selection
Key criteria for choosing an embedding model:
1.  **Performance Benchmarks:** Check MTEB (Massive Text Embedding Benchmark) scores for retrieval tasks. Top contenders include `text-embedding-3-large` (OpenAI), `E5-mistral` (Microsoft), and `bge-m3` (BAAI).
2.  **Dimensionality:** Higher dimensions (e.g., 1536 vs. 768) often capture more nuance but increase storage and compute costs.
3.  **Context Length:** Must support the maximum chunk size you intend to embed (e.g., 8k tokens).
4.  **Language Support:** Multilingual models (e.g., `bge-m3`, `E5`) are essential for non-English datasets.
5.  **Latency/Cost:** Local models (HuggingFace) vs. API-based (OpenAI/Azure). Local offers privacy and no per-call cost but requires GPU infrastructure.

### Fine-Tuning Embedding Models
*   **Why Fine-Tune?** Pre-trained models are generalists. Fine-tuning adapts the model to your specific domain (e.g., medical, legal, internal jargon).
*   **Process:** Train on pairs of `(query, relevant_document)` and `(query, irrelevant_document)` using contrastive loss.
*   **Impact:** Significant boost in retrieval accuracy for domain-specific terminology that generic models might treat as noise or map incorrectly.

### Quantization of Embeddings
Quantization reduces the precision of embedding vectors to save storage and speed up search.

#### Scalar Quantization (e.g., INT8)
*   **Method:** Converts 32-bit floats to 8-bit integers.
*   **Pros:** ~4x reduction in storage/memory; minimal accuracy loss (<1-2%).
*   **Cons:** Still requires significant memory compared to binary.

#### Binary Quantization (1-bit)
*   **Method:** Converts each dimension to a single bit (0 or 1) based on sign.
*   **Pros:** ~32x reduction in storage; extremely fast Hamming distance calculations.
*   **Cons:** Higher accuracy drop (5-10%); best used with re-ranking to correct errors.

#### Pros and Cons of Quantized Embeddings
*   **Pros:**
    *   Drastically reduces VectorDB storage costs.
    *   Increases throughput (more vectors fit in RAM/GPU memory).
    *   Faster ANN search due to simpler distance calculations (Hamming vs. Cosine).
*   **Cons:**
    *   Loss of precision can lead to lower recall.
    *   Requires careful evaluation to ensure accuracy drop is acceptable.
    *   Not all VectorDBs support efficient binary search natively.

### Dimensionality Reduction
*   **Techniques:** PCA (Principal Component Analysis), UMAP, or trained projection layers.
*   **Goal:** Reduce vector size (e.g., from 1536 to 256 dimensions) to save space.
*   **Vs. Quantization:** Dimensionality reduction removes *information axes*, while quantization reduces *precision per axis*.
*   **Recommendation:** Quantization is generally preferred over dimensionality reduction for RAG, as it preserves the semantic structure better with less accuracy loss.

---

## Vector Search

### Role of Vector Database (VectorDB)
A specialized database optimized for storing and searching high-dimensional vectors.
*   **Functions:** Indexing (building ANN structures), Storage (persistence + metadata), Querying (similarity search), Filtering (pre/post-filtering).
*   **Popular Options:** Pinecone, Milvus, Qdrant, Weaviate, Chroma, Pgvector (PostgreSQL extension).

### ANN (Approximate Nearest Neighbor) Algorithms
Exact nearest neighbor search is too slow for large datasets ($O(N)$). ANN algorithms approximate the result for sub-linear time complexity.

#### Common Algorithms
1.  **HNSW (Hierarchical Navigable Small World):**
    *   *Working:* Builds a multi-layer graph. Top layers have sparse connections for long-distance jumps; bottom layers have dense connections for fine-grained search.
    *   *Pros:* Extremely fast query speed, high recall, dynamic (supports inserts/deletes).
    *   *Cons:* High memory usage (stores graph pointers).
2.  **IVF (Inverted File Index):**
    *   *Working:* Clusters vectors into Voronoi cells. Search only examines the nearest clusters.
    *   *Pros:* Lower memory footprint than HNSW.
    *   *Cons:* Slower than HNSW for high-recall requirements; requires tuning number of clusters.
3.  **DiskANN:**
    *   *Working:* Optimized for SSD storage, keeping only a small index in RAM.
    *   *Pros:* Massive scale (billions of vectors) at low cost.
    *   *Cons:* Higher latency than pure RAM-based solutions.

#### Speed Factors
*   **Index Type:** HNSW is fastest for RAM-bound systems.
*   **Ef_search / ef_construction:** Hyperparameters controlling the trade-off between speed and accuracy. Higher values = slower but more accurate.
*   **Dimensionality:** Lower dimensions search faster.
*   **Dataset Size:** ANN scales logarithmically or sub-linearly, unlike linear brute force.

### Distance Metrics
Metrics determine how "close" two vectors are.

1.  **Cosine Similarity:**
    *   *Formula:* Measures the cosine of the angle between two vectors. Ignores magnitude.
    *   *Preference:* **Preferred in RAG.** Embedding models are typically normalized during training, making magnitude irrelevant. Cosine similarity focuses purely on direction (semantic meaning).
2.  **Euclidean Distance (L2):**
    *   *Formula:* Straight-line distance between points.
    *   *Usage:* Good if magnitudes matter (rare in text embeddings). Equivalent to Cosine if vectors are normalized.
3.  **Dot Product:**
    *   *Usage:* Fastest computation. Equivalent to Cosine if vectors are normalized. Often used in production for speed.

### Keyword vs. Semantic Retrieval
*   **Keyword (Lexical):** Matches exact words. Good for "iPhone 15 Pro Max price" where exact terms matter. Fails on "How much does Apple's latest pro phone cost?"
*   **Semantic:** Matches meaning. Good for conceptual queries. Fails on exact IDs, codes, or rare names not well-represented in the embedding space.
*   **Conclusion:** Neither is perfect alone. This leads to **Hybrid Search**.

---

## Search Strategies

### Hybrid Search
Combines Keyword (Sparse) and Semantic (Dense) retrieval.

#### How It Works
1.  Perform Keyword Search (BM25) $\rightarrow$ Get List A with scores.
2.  Perform Semantic Search (Vector) $\rightarrow$ Get List B with scores.
3.  **Normalize Scores:** Convert both lists to a common scale (e.g., 0-1).
4.  **Combine:** Merge lists using weighted sum or Reciprocal Rank Fusion (RRF).
    *   *RRF Formula:* $Score = \sum \frac{1}{k + rank_i}$ (Robust against score scale differences).

#### When to Opt for Hybrid Search
*   **Mixed Queries:** Users mix specific terms (names, dates) with natural language.
*   **Domain Specific Jargon:** Where semantic models might drift but keywords are precise.
*   **High Precision Requirements:** When missing an exact match is critical (e.g., legal clauses, product codes).

### Balancing Relevance and Diversity
Retrieving 10 chunks that are nearly identical provides no new information.
*   **Problem:** Redundancy in top-k results.
*   **Solution: Maximal Marginal Relevance (MMR).**
    *   *Algorithm:* Iteratively selects chunks that are highly relevant to the query *and* dissimilar to already selected chunks.
    *   *Hyperparameter:* $\lambda$ (lambda).
        *   $\lambda=1$: Pure relevance (no diversity).
        *   $\lambda=0$: Pure diversity (ignores query).
        *   Typical: 0.5–0.7.
*   **Alternative:** Clustering retrieved results and picking the best from each cluster.

### Scaling Embeddings
As datasets grow to millions/billions of vectors:

1.  **Sharding:** Distribute vectors across multiple nodes/servers.
2.  **Quantization:** Use INT8 or Binary quantization to reduce memory footprint by 4x–32x.
3.  **Index Optimization:** Use DiskANN for billion-scale indexes to keep RAM usage low.
4.  **Caching:** Cache frequent query embeddings and their results.
5.  **Partitioning:** Partition data by metadata (e.g., tenant ID, date) to limit search scope before vector search.
6.  **Model Efficiency:** Use smaller, distilled embedding models (e.g., `bge-small`) if accuracy loss is acceptable for speed gains.

**Summary Strategy for Scale:**
*   *Small (<1M):* HNSW in RAM, Float32.
*   *Medium (1M–100M):* HNSW with INT8 Quantization.
*   *Large (>100M):* DiskANN or IVF-PQ (Product Quantization) with Hybrid Search and Re-ranking.