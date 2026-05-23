# Low rank adoption for large language model

[Paper](https://arxiv.org/pdf/2106.09685)

## 1. Core Intuition & Math
*   **Hypothesis:** Task-specific adaptation has low **intrinsic rank**. We don't need to update all weights, only a low-rank subspace.
*   **Formula:** $h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A x)$
    *   $W_0 \in \mathbb{R}^{d \times k}$: Frozen pre-trained weights.
    *   $A \in \mathbb{R}^{r \times k}$: Trainable down-projection (Gaussian init).
    *   $B \in \mathbb{R}^{d \times r}$: Trainable up-projection (**Zero init**).
    *   $r \ll d, k$: Rank (e.g., 8 vs 4096).
    *   $\alpha$: Scaling hyperparameter.
*   **Scaling Factor ($\frac{\alpha}{r}$):** Decouples rank from update magnitude. Keeps variance stable when changing $r$. Common heuristic: $\alpha = r$ or $\alpha = 2r$.
*   **Orthogonality & Signal:** Pre-trained weights W0​ capture dominant principal components. LoRA updates are largely **orthogonal** to these, injecting task-specific nuances into unused dimensions without overwriting general knowledge (catastrophic forgetting).

## 2. Initialization & Stability
*   **$B=0$ Init:** Ensures $\Delta W = 0$ at $t=0$. Model starts exactly as base model. Prevents initial noise/distortion.
*   **Gradient Flow at Start:**
    *   $\nabla_B \neq 0$ (depends on $Ax$, which is random noise).
    *   $\nabla_A = 0$ (depends on $B^T$, which is 0).
    *   **Result:** $B$ learns first, then $A$. Smooth, stable convergence.

## 3. Backward Pass (Gradients)
*   **$\nabla_B = \frac{\alpha}{r} \delta_h (Ax)^T$**
*   **$\nabla_A = \frac{\alpha}{r} B^T \delta_h x^T$**
*   **$\nabla_{W_0}$:** Computed but **discarded** (`requires_grad=False`). Saves ~99% of training memory.

## 4. Why LoRA Wins Production (vs. Adapters/Prompt Tuning)
| Feature               | Adapter Layers                   | Prompt Tuning               | **LoRA**                          |
| :-------------------- | :------------------------------- | :-------------------------- | :-------------------------------- |
| **Inference Latency** | **High** (Serial non-linear ops) | **Medium** (KV Cache bloat) | **Zero** (Mergeable)              |
| **Context Window**    | Unaffected                       | **Consumed** by prefixes    | Unaffected                        |
| **Deployment**        | Hard (Custom kernels)            | Easy                        | **Easy** (Standard arch)          |
| **Multi-Tenancy**     | Poor (Memory heavy)              | Poor (KV limits)            | **Excellent** (Swap tiny weights) |

*   **Fusion Math:** Because LoRA is linear, $W_{merged} = W_0 + \frac{\alpha}{r}BA$. Inference becomes a single matmul: $y = W_{merged}x$.
*   **Adapters Fail Fusion:** Non-linearities (ReLU/GELU) inside adapters prevent matrix merging ($f(Wx) \neq W'x$).

## 5. Target Modules: Llama 3 & Qwen 2.5
*   **Attention Layers (`q_proj`, `v_proj`):** Good for style/tone. Minimal params.
*   **MLP Layers (`gate_proj`, `up_proj`, `down_proj`):** Critical for **knowledge injection** and **reasoning** (Coding/Math).
*   **Recommendation:**
    *   *Simple Chat:* `q_proj`, `v_proj`.
    *   *Complex Domain (Code/Med):* `all-linear` (Attention + MLP).
    *   *Qwen Specific:* Responds well to full `all-linear` due to dense pre-training.

## 6. Experimental Strategy: Finding Optimal Rank ($r$)
*   **Don't sweep full dataset.** Use a **Cascade Approach**:
    1.  **Proxy Sweep:** 1-5% data, few steps. Filter unstable ranks.
    2.  **Validation Sweep:** Top 3 candidates on 10-20% data. Find the **Elbow Point** (diminishing returns).
    3.  **Final Train:** Only the winner goes to 100% data.
*   **Keep Ratio Constant:** Fix $\frac{\alpha}{r}$ (e.g., 2) while sweeping $r \in \{4, 8, 16, 32, 64\}$.
*   **Heuristics:**
    *   *Style:* $r=4-8$.
    *   *General QA:* $r=8-16$.
    *   *Coding/Math:* $r=16-32+$ (MLPs need higher rank).
*   **Monitor Forgetting:** Check general benchmarks (MMLU) to ensure high $r$ isn't causing overfitting/catastrophic forgetting.

## 7. Precision & Numerical Stability

- **Base Model (W0):** Typically kept in **FP16** or **BF16** to save VRAM.
- **LoRA Weights (A,B):** Should be maintained in **FP32** during the optimizer step.
    - _Why?_ LoRA updates are small perturbations. In FP16, these small gradients can suffer from **underflow**(becoming zero) or **overflow**, leading to training instability or "NaN" losses.
- **Merging Precision:** When merging Wmerged=W0+ΔW, ensure both are cast to the same precision.
    - _Risk:_ Adding a high-precision ΔW to a low-precision W0​ can result in precision loss if the magnitude of ΔW is smaller than the epsilon of W0​'s format.
- **QLoRA Note:** If using 4-bit quantization, LoRA weights are kept in FP16/BF16, but the base model remains frozen in 4-bit. This requires specific kernel support (e.g., `bitsandbytes`) to handle the mixed-precision matrix multiplication.

## 8. PyTorch Implementation Nuances
```python
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=8, alpha=16):
        super().__init__()
        self.W0 = nn.Linear(in_f, out_f, bias=False)
        self.W0.requires_grad_(False) # Freeze
        self.A = nn.Linear(in_f, r, bias=False)
        self.B = nn.Linear(r, out_f, bias=False)
        self.scaling = alpha / r
        
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight) # Crucial

    def forward(self, x):
        return self.W0(x) + self.B(self.A(x)) * self.scaling
```
*   **Memory Savings:** For $d=4096, r=8$, LoRA has ~65k params vs 16M for full FT. **99.6% reduction.**

## 9. Conceptual Deep Dive: Rank & Orthogonality

- **Rank (Dimensionality of Information):**
    - **Definition:** The number of linearly independent rows or columns in a matrix. It represents the "degrees of freedom" or the dimensionality of the subspace spanned by the matrix.
    - **In LoRA:** A full weight matrix W0 has high rank (capturing general language). The update ΔW=BA low rank r. This forces the model to learn only the **most essential, compressed directions** of change for the new task, acting as a bottleneck that filters out noise and prevents overfitting.
- **Orthogonality (Independence of Features):**
    - **Definition:** Two vectors are orthogonal if their dot product is zero (A⋅B=0). In high-dimensional space, this means they are perpendicular and carry **uncorrelated information**.
    - **In LoRA:** Pre-trained weights W0​ already occupy the primary principal components of the data distribution. LoRA updates are learned in residual directions that are largely **orthogonal** to W0​'s dominant features. This ensures the adapter injects _new_ task-specific knowledge without interfering with or overwriting the base model's existing general capabilities (minimizing catastrophic forgetting).
