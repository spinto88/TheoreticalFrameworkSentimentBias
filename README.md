# Theoretical Framework for Sentiment Bias

A probabilistic model that infers ideological scores for media outlets and sentiment coefficients for political subjects from sentiment-classified social media posts.

## Model

Each media outlet `i` is assigned a latent ideological score `z_i`, and each political subject `j` has a sensitivity coefficient `a_j` and a baseline bias `b_j`. The effective sentiment field for outlet `i` covering subject `j` is `z_ij = z_i * a_j + b_j`. Given `z_ij`, the probability of a post being negative, neutral, or positive follows a Boltzmann distribution. Parameters are inferred by maximizing the regularized log-likelihood via differential evolution.

## Notebooks

### `SyntheticDataEstimation.ipynb`
Validates the inference procedure on controlled synthetic data. Generates artificial sentiment counts for 20 media outlets and 3 subjects with known `z`, `a`, and `b` values, then recovers those parameters via maximum likelihood and measures recovery quality using Spearman correlation.

### `DataFitting.ipynb`
Applies the model to real data from the 2024 US presidential election (`Data/data2024.csv`), covering ~38k Facebook posts from 40 media outlets and four candidates (Trump, Kamala, Vance, Walz). Produces inferred ideological scores for each outlet and sentiment coefficients for each candidate, and validates the outlet scores against MediaBiasFactCheck ground-truth labels.

### `Model_consistency.ipynb`
Tests the robustness of the inferred outlet scores by re-fitting the model on progressively subsampled versions of the real data (from 100% down to 10% of total mentions) and computing Spearman correlation of each result against the full-data solution.
