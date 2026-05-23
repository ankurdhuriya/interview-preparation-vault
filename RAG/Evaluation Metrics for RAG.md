

Evaluating RAG systems requires a two-pronged approach: assessing the **Retriever** (did we find the right info?) and the **Generator** (did we use it correctly?). Traditional Information Retrieval (IR) metrics are foundational, but RAG-specific metrics address the nuances of context utilization and hallucination.

## Traditional IR Metrics

These metrics evaluate the quality of the retrieved list of documents/chunks against a set of "ground truth" relevant documents.

### 1. Precision@k and Recall@k
*   **Precision@k:** The proportion of retrieved documents in the top-$k$ results that are relevant.
    *   $Precision@k = \frac{\text{Relevant Docs in Top } k}{k}$
    *   *Focus:* Accuracy. "Of what I showed you, how much was useful?"
*   **Recall@k:** The proportion of all relevant documents in the corpus that were retrieved in the top-$k$ results.
    *   $Recall@k = \frac{\text{Relevant Docs in Top } k}{\text{Total Relevant Docs in Corpus}}$
    *   *Focus:* Completeness. "Did I miss any important info?"
*   **Trade-off:** High precision often lowers recall (being picky), and vice versa.

### 2. MRR (Mean Reciprocal Rank)
*   **Definition:** The average of the reciprocal ranks of the first relevant document for each query.
*   **Formula:** $MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$
    *   If the first relevant doc is at rank 1, score is 1. If at rank 2, score is 0.5. If not in top-$k$, score is 0.
*   **Limitation:** **Only considers the FIRST relevant document.** It ignores if there are multiple relevant chunks or if other relevant chunks are ranked poorly. Unsuitable for RAG where synthesizing multiple sources is common.

### 3. MAP (Mean Average Precision)
*   **Definition:** The mean of the Average Precision (AP) scores for a set of queries. AP calculates precision at every rank where a relevant document is found and averages them.
*   **Calculation (MAP@5 Example):**
    *   Query Result: [Rel, Irrel, Rel, Irrel, Rel] (R=Relevant, I=Irrelevant)
    *   Precision@1 (R): 1/1 = 1.0
    *   Precision@3 (R): 2/3 = 0.67
    *   Precision@5 (R): 3/5 = 0.60
    *   AP = $(1.0 + 0.67 + 0.60) / 3 \text{ (total relevant)} = 0.75$
*   **What it Reveals:** Unlike raw Precision, MAP rewards **ranking relevant documents higher**. A system that puts relevant docs at ranks 1, 2, 3 gets a higher MAP than one putting them at 3, 4, 5.
*   **Limitation:** Assumes binary relevance (relevant/not relevant) and doesn't account for graded relevance (how *much* more relevant one doc is than another).

### 4. NDCG (Normalized Discounted Cumulative Gain)
*   **Order-Awareness:** NDCG is **order-aware**. It assigns higher weights to relevant documents appearing at the top of the list and discounts relevance as rank increases.
*   **Graded Relevance:** Unlike Precision/Recall, NDCG can handle graded relevance (e.g., 0=irrelevant, 1=somewhat relevant, 2=highly relevant).
*   **Impact of Reverse Order:** If all relevant chunks are retrieved but in reverse order (least relevant first), NDCG drops significantly because the "Gain" is heavily discounted by the low position. MRR would also be low if the *most* relevant isn't first. Precision@k would remain unchanged (if all are still in top-k).

---

## RAG-Specific Context Metrics

These metrics evaluate the **retrieved context** specifically in relation to the **query** and the **ground truth answer**. They are often evaluated using LLM-as-a-Judge.

### 1. Context Precision
*   **Definition:** Measures how much of the retrieved context is relevant to the query. It penalizes irrelevant noise in the context window.
*   **Weighted Sum Approach:**
    *   It evaluates each chunk in the retrieved context for relevance to the query.
    *   Formula: $Context Precision@K = \frac{\sum_{k=1}^{K} (\text{Precision@k} \times v_k)}{\text{Total Relevant Chunks}}$
    *   Where $v_k$ is 1 if the chunk at rank $k$ is relevant, else 0.
*   **Low Score Reasons:**
    *   Retriever returns too many irrelevant chunks (low signal-to-noise).
    *   Poor chunking (chunks contain mixed topics).
    *   Query ambiguity leading to broad, unfocused retrieval.
*   **Difference from Standard Precision:** Standard Precision counts documents. Context Precision evaluates the *utility* of the text within the context window for answering the specific query.

### 2. Context Recall
*   **Definition:** Measures the extent to which the retrieved context aligns with the **ground truth answer**. It answers: "Is all the information needed to answer the question present in the retrieved context?"
*   **Ground Truth Claims:**
    *   The ground truth answer is broken down into atomic "claims" or facts.
    *   The evaluator checks if each claim can be inferred from the retrieved context.
    *   $Context Recall = \frac{\text{Claims Supported by Context}}{\text{Total Claims in Ground Truth}}$
*   **Impact on Completeness:** Low Context Recall means the LLM *cannot* answer the question fully because the info is missing from the context, forcing it to hallucinate or say "I don't know."
*   **Root Causes for Low Scores:**
    *   Retriever failed to find the specific document containing the answer.
    *   Chunking split the answer across multiple chunks, and not all were retrieved.
    *   Embedding model failed to capture semantic similarity between query and answer source.

### 3. Context Relevancy
*   **Reference-Free vs. Reference-Dependent:**
    *   **Context Precision/Recall:** **Reference-Dependent.** They require a "Ground Truth Answer" to compare against.
    *   **Context Relevancy:** **Reference-Free.** It evaluates the relationship between the **Query** and the **Retrieved Context** directly, without needing a ground truth answer.
*   **When to Use:**
    *   Use Context Relevancy when you don't have labeled ground truth answers (unsupervised evaluation).
    *   Use Precision/Recall when you have a golden dataset for rigorous benchmarking.
*   **High Relevancy / Low Precision Scenario:**
    *   *Scenario:* The retrieved chunks are topically related to the query (high relevancy) but contain too much extraneous information or fail to pinpoint the specific answer (low precision).
    *   *Implication:* The retriever is "on the right track" semantically but lacks focus. Needs better chunking or re-ranking.

---

## Generation Quality Metrics

These metrics evaluate the **LLM's final output**.

### 1. Faithfulness
*   **Definition:** Measures whether the generated answer is factually consistent with the **retrieved context**. It detects hallucinations.
*   **Assessing Hallucinations:**
    *   Breaks the generated answer into claims.
    *   Checks if each claim is supported by the retrieved context.
    *   $Faithfulness = \frac{\text{Claims Supported by Context}}{\text{Total Claims in Generated Answer}}$
*   **Difference from Context Precision:**
    *   **Context Precision:** Evaluates the *input* (retrieved chunks). Are the chunks relevant?
    *   **Faithfulness:** Evaluates the *output* (generated answer). Did the LLM stick to the chunks?
    *   *High Context Precision + Low Faithfulness:* The retriever did its job (good chunks), but the LLM ignored them and made things up (bad generator/prompting).
*   **Improving Faithfulness:**
    *   Stronger prompt instructions ("Answer ONLY using context").
    *   Better re-ranking (remove noisy chunks).
    *   Using models with lower temperature.
    *   Citation-forcing (ask LLM to cite chunk IDs).

### 2. Response Relevancy
*   **Definition:** Measures whether the generated answer addresses the **user’s original query**.
*   **Difference from Context Relevancy:**
    *   **Context Relevancy:** Is the *retrieved data* relevant to the query?
    *   **Response Relevancy:** Is the *final answer* relevant to the query?
*   **Why Both are Needed:**
    *   You can have relevant context but an irrelevant answer (LLM went off-topic).
    *   You can have an irrelevant context but a relevant answer (LLM used its internal knowledge, which might be hallucinated or outdated).
*   **Risks of Sole Reliance on Response Relevancy:**
    *   An answer can be highly relevant to the query but completely fabricated (high relevancy, zero faithfulness).
    *   *Example:* Query: "Who won the 2026 World Cup?" Answer: "France won." (Relevant to query, but potentially hallucinated if not in context).
*   **Diagnosing Generator Issues:**
    *   **Low Response Relevancy:** The LLM misunderstood the query or the prompt was unclear.
    *   **High Response Relevancy + Low Faithfulness:** The LLM is ignoring the context and relying on parametric memory (hallucinating).
    *   **Low Response Relevancy + High Faithfulness:** The LLM strictly followed irrelevant context (Garbage In, Garbage Out).

### Summary Matrix for Diagnosis

| Scenario          | Context Precision | Context Recall | Faithfulness | Response Relevancy | Diagnosis                                                    |
| :---------------- | :---------------: | :------------: | :----------: | :----------------: | :----------------------------------------------------------- |
| **Ideal**         |       High        |      High      |     High     |        High        | System working perfectly.                                    |
| **Missing Info**  |        Low        |    **Low**     |     N/A      |        Low         | Retriever failed to find answer. Improve retrieval/chunking. |
| **Noisy Context** |      **Low**      |      High      |     Low      |       Medium       | Retriever found answer but added junk. Improve re-ranking.   |
| **Hallucination** |       High        |      High      |   **Low**    |        High        | Retriever good, LLM ignored context. Fix prompt/model.       |
| **Off-Topic**     |       High        |      High      |     High     |      **Low**       | LLM misunderstood query. Fix prompt/query transformation.    |