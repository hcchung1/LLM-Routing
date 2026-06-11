# 2026 Spring Introduction to NLP Final Project

## LLM Routing

### Overview
Welcome to the INLP 2026 Final Project: LLM Routing Optimization Challenge!

In the rapidly evolving landscape of Large Language Models (LLMs), a single model is rarely the best choice for every possible query. Different models excel in different domains, and they come with vastly different inference costs.

The goal of this competition is to build an intelligent LLM Router. Given a user query, your router must dynamically select the most appropriate model from a pool of 11 candidate models (anonymized as `Model_A` through `Model_K`). A successful router will learn to balance the trade-off between maximizing response quality (performance) and minimizing API expenses (cost).

### Description
In real-world LLM applications, routing queries to the right model can save massive amounts of compute while maintaining high user satisfaction. For complex reasoning tasks, you might want to route to a highly capable but expensive model. For simple informational queries, a smaller, cheaper model might suffice.

### Dataset
You are provided with a dataset of queries and the corresponding performance and cost metrics for 11 different LLMs.

- **train.csv**: Contains the ID, the text query, and the ground-truth performance (0.0 to 1.0) and cost for each of the 11 candidate models.
- **test.csv**: Contains only the ID and the text query. You must predict the best model to route to for these queries.
- **sample_submission.csv**: A sample submission file showing the correct format.
Your task is to build a routing mechanism, whether it's a lightweight machine learning classifier, a similarity-based KNN router, or an LLM-as-a-Judge, to predict the optimal pred_model for each query in the test set.

### Evaluation
Submissions are evaluated on a custom **Reward** metric that explicitly balances performance and cost.

### Metric
The evaluation metric is $Reward_{0.85}$, which heavily favors performance but applies a penalty for excessive cost. It is calculated globally across your entire submission as follows:

$Reward_{0.85} = 0.85 \times \bar{P} - 0.15 \times \frac{\bar{C}}{\overline{C_{max}}}$

Where:

- $\bar{P}$ is the average performance of the models you selected across all test queries.
- $\bar{C}$ is the average cost of the models you selected across all test queries.
- $\overline{C_{max}}$ is the average maximum cost available per query (used as a global normalization factor to scale costs between 0 and 1).
Your goal is to maximize this Reward score.

### Submission File
For each ID in the test set, you must predict the chosen model for the `pred_model` column. The prediction must be an exact string match to one of the 11 candidate models (e.g., `Model_A`, `Model_B`, …, `Model_K`).

The file should contain a header and have the following format:

```
ID,pred_model
1,Model_A
2,Model_C
3,Model_K
etc.
```
