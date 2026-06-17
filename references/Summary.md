# RouterBench Router Papers Summary

Search date: 2026-06-05

Scope used here:

- Included: papers that propose, train, analyze, or benchmark a router method/model and use RouterBench or RouterBench-style model-routing evaluation.
- Excluded: papers whose main contribution is expanding, replacing, or curating a dataset/benchmark rather than proposing a router method.
- Reward check: I searched for the exact homework/Kaggle metric `Reward_{0.85} = 0.85 * P_bar - 0.15 * C_bar / C_max`. I did not find a public paper that uses this exact fixed formula. Several papers use related performance-cost utilities, cost constraints, AIQ, budgeted reward, regret, or Pareto/frontier metrics, but they are not the same metric unless explicitly reimplemented with `alpha=0.85` and global `C_max` normalization.

## Included Papers

| PDF | Paper | Why included | Router idea | Notes for this project |
|---|---|---|---|---|
| `routerbench_2403.12031.pdf` | RouterBench: A Benchmark for Multi-LLM Routing System | Foundational RouterBench paper; includes routing baselines and evaluation setup. | Frames routing as choosing among LLMs by cost-quality trade-off. | Useful for terminology and baselines, but it is more benchmark-focused than method-focused. It does not use the exact `Reward_0.85` homework formula. |
| `graphrouter_2410.03834.pdf` | GraphRouter: A Graph-based Router for LLM Selections | Router/model paper. | Represents relationships among queries/models as graph structure and routes by learned graph signals. | Relevant if we want model/query relational priors beyond TF-IDF/KNN. Higher complexity than our current CPU-friendly pipeline. |
| `carrot_2502.03261.pdf` | CARROT: A Cost Aware Rate Optimal Router | Router-method paper with explicit cost awareness. | Optimizes routing decisions under cost-quality trade-offs, aiming for rate-optimal allocation. | Conceptually close to our reward objective, but not the exact fixed `0.85/0.15` reward. |
| `uniroute_2502.08773.pdf` | UniRoute / Universal Model Routing for Efficient LLM Inference | Router-method paper focused on generalizing routing across model pools. | Learns transferable model-routing behavior rather than overfitting to one fixed model set. | Useful idea if we want robust routing across hidden/public-private split drift. |
| `causal_llm_routing_2505.16037.pdf` | Causal LLM Routing via End-to-End Regret Minimization | Router-method paper. | Uses causal/regret-based objective to reduce bad routing decisions. | Relevant because our local CV can overfit; regret-style training may be safer than pure top-1 classifier training. |
| `irt_router_2025.acl-long.761.pdf` | IRT-Router: Learning to Route LLMs with Item Response Theory | Router-model paper. | Models prompt difficulty and model ability using Item Response Theory. | Strong fit for this repo because train data contains per-model binary/continuous performance and cost. We already tried related IRT/NIRT ideas; local/public evidence should decide whether to revisit. |
| `cost_aware_contrastive_routing_2508.12491.pdf` | Cost-Aware Contrastive Learning for LLM Routing | Router-method paper. | Learns query/model representations with contrastive objectives while accounting for cost. | Good candidate idea for embeddings: train reward-aware query/model similarity rather than only query TF-IDF similarity. |
| `adaptive_llm_routing_budget_2508.21141.pdf` | Adaptive LLM Routing under Budget Constraints | Router-method paper. | Routes adaptively under explicit budget constraints. | Relevant to the cost side of `Reward_0.85`, but homework metric is unconstrained scalar reward, not a hard budget. |
| `port_training_free_online_routing_2509.02718.pdf` | PORT / Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving | Router-method paper. | Online/training-free routing that updates routing behavior from serving feedback. | Less directly useful for offline Kaggle because test feedback is unavailable, but useful for practical deployment discussion. |
| `cara_cross_attention_routing_2509.09782.pdf` | CARA / Cross-Attention Routing: One Head, Many Models | Router-model paper. | Uses cross-attention between query and candidate model information to select a model. | Interesting if building a neural router. More expensive than current TF-IDF/GBM/KNN approaches. |
| `llm_routing_dueling_feedback_2510.00841.pdf` | LLM Routing with Dueling Feedback | Router-method paper. | Learns routing from pairwise/dueling feedback rather than full labels. | Relevant to pairwise ranking variants in this repo. May help if model-level labels are noisy. |
| `equirouter_routing_collapse_2602.03478.pdf` | When LLM Routing Collapses: The Shortcut Model Selection Problem and EquiRouter | Router-analysis and router-method paper. | Studies shortcut/collapse failure modes and proposes EquiRouter to avoid degenerate routing. | Important warning: a high local score may come from shortcuts or over-selecting one model. This supports keeping diverse candidate submissions. |
| `contextual_queueing_bandits_2602.02061.pdf` | Contextual Queueing Bandits for LLM Routing | Router-method paper. | Combines contextual bandits with queueing/service constraints. | More deployment-oriented than Kaggle-oriented. Useful for report discussion, not a direct offline training recipe. |
| `icl_router_aaai_40628.pdf` | ICL-Router: Learning In-Context Examples for Router Prediction | Router-method paper. | Uses in-context examples to improve router prediction. | Related to prompt/LLM-as-router methods. Potentially costly and quota-heavy, so only worth a small smoke test. |

## Reward_0.85 Finding

The homework PDF says the Kaggle score is `Reward_{0.85}` and points to the Kaggle challenge page for the definition. In this repo, the implemented Kaggle-aligned version is:

```text
Reward_0.85 = 0.85 * mean(performance) - 0.15 * mean(cost) / global_max_cost
```

I did not find public RouterBench papers that use this exact fixed formula with `alpha=0.85` and global `C_max` normalization. The closest matching families are:

- cost-aware reward or utility optimization;
- budget-constrained routing;
- Pareto or AIQ-style quality-cost evaluation;
- regret minimization;
- pairwise or contrastive routing objectives that can be adapted to optimize our reward labels.

For this Kaggle task, the safest interpretation is: use these papers for router architecture/objective ideas, but compute labels and validation with the local `Reward_0.85` formula, not with the paper's original metric.

## Papers Intentionally Excluded

These are relevant to the broader routing literature but were not downloaded because their main contribution is benchmark/dataset expansion or platform evaluation, not a router method/model:

- LLMRouterBench-style benchmark expansion papers.
- RouterArena / arena-style evaluation papers.
- VL-RouterBench or multimodal RouterBench variants.
- TwinRouterBench / adversarial or reliability benchmark variants.
- General survey papers without a concrete router method.

## Practical Takeaways for This Repo

1. Keep `Reward_0.85` as the training/validation target when possible. Do not blindly optimize AIQ, raw accuracy, cost-only budget, or paper-specific utility.
2. IRT-Router, pairwise/dueling feedback, and cost-aware contrastive learning are the most directly reusable families for this dataset because train rows expose per-model performance and cost.
3. Online/bandit/queueing papers are better for report context than Kaggle implementation because the test set provides no online feedback.
4. Collapse/shortcut papers support the current practice of generating multiple candidate submissions and not trusting one local CV winner too strongly.
5. If implementing one new idea, the lowest-risk next step is a reward-matrix ranker or contrastive/ranking model trained on `0.85 * performance - 0.15 * normalized_cost`, then blended with the current TF-IDF/KNN/Ridge candidates.
