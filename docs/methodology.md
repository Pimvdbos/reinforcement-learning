# Methodology

## Research question

Can a PPO-based reinforcement-learning agent learn a dynamic allocation policy that adds value relative to simple diversification and classical mean-variance optimization?

The project is designed as an empirical policy-learning study rather than a one-step return-forecasting exercise. The agent repeatedly observes a market state, chooses a portfolio allocation, receives the next day's return, and updates its policy to maximize discounted cumulative reward.

## Markov Decision Process

### State

For 14 assets, the observation is

\[
s_t =
[f_t,\; w_{t-1},\; c_t]
\in \mathbb{R}^{116},
\]

where:

- \(f_t \in \mathbb{R}^{101}\) is the engineered feature vector;
- \(w_{t-1}\in\mathbb{R}^{14}\) is the previous target allocation;
- \(c_t\) is the residual cash fraction.

The 101 engineered features consist of 98 asset-specific inputs, two market-level inputs, and one dynamic portfolio drawdown feature.

The state is only an approximation of a Markov state. Financial markets can contain long-memory, latent regimes, structural breaks, and exogenous shocks that cannot be compressed perfectly into the selected rolling indicators.

### Action

The policy produces one continuous component per asset. The environment maps the action to the long-only simplex. The public refactor performs the normalization inside the environment as a defensive invariant, even though Stable-Baselines3 also respects Box action bounds.

### Transition

The key timing invariant is:

1. observe features available at date \(t\);
2. choose allocation \(w_t\);
3. realize the exogenous asset-return vector \(r_{t+1}\);
4. update portfolio value and construct \(s_{t+1}\).

This separation prevents the next-day return from entering the decision state.

### Reward

The submitted reward is

\[
R_{t+1}
=
\log(1+r_{p,t+1})
-
\lambda_r r_{p,t+1}^2
-
c_{\mathrm{tr}}\lVert w_t-w_{t-1}\rVert_1.
\]

The first term rewards compound growth. The second is a symmetric quadratic penalty on the realized one-period portfolio return. It can discourage large absolute daily moves, but it is not a direct estimate of conditional variance. The final term discourages large target-weight changes.

The public implementation supports two cost-accounting conventions:

- `reward_only`: preserves the submitted experiment, where the turnover term affects reward but not reported wealth;
- `net`: deducts the same proportional turnover-cost proxy from realized wealth and applies it consistently across PPO and benchmarks.

## PPO

Proximal Policy Optimization is an on-policy actor-critic algorithm. Stable-Baselines3 `MlpPolicy` parameterizes a continuous stochastic policy and a state-value function. PPO constrains policy updates using the clipped surrogate objective:

\[
L^{CLIP}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)\hat{A}_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right].
\]

The clip mechanism stabilizes optimization; it should not be interpreted as a financial variance penalty. Risk preferences in this project enter through the reward function.

The final experiment used:

- learning rate: `8.27494264609064e-05`
- rollout length: `512`
- batch size: `32`
- gamma: `0.9896501306459291`
- GAE lambda: `0.9501835963152316`
- entropy coefficient: `0.11274745595800702`
- risk-aversion coefficient: `0.1344558062397565`
- seed: `1986`
- 200,000 training timesteps for the principal experiments

## Feature engineering

For every asset:

1. one-day log return;
2. five-day log return;
3. 20-day realized volatility;
4. 20-day momentum;
5. 14-day RSI;
6. Bollinger Band width;
7. Bollinger Band position.

Market context is represented by the S&P 500 one-day log return and 20-day rolling market volatility. Current portfolio drawdown is injected dynamically by the environment.

`RobustScaler` is fit separately inside each chronological training split and then applied to the corresponding validation/test period.

## Evaluation

### Chronological holdout

The first 80% of the aligned feature sample is used for training and the final 20% for evaluation. The split is never shuffled.

### Walk-forward validation

Five expanding-window folds are produced with `TimeSeriesSplit`. The scaler is refit using each fold's training sample, PPO is retrained for that fold, and all reference strategies use the same train/test boundary.

### Reference policies

**Equal Weight** assigns \(1/N\) to every asset.

**Mean-Variance Optimization** estimates expected returns and the covariance matrix from training prices only and selects a long-only maximum-Sharpe portfolio. The submitted PyPortfolioOpt implementation used a 2% annual risk-free rate in the MVO objective; reported evaluation Sharpe ratios use a zero risk-free rate, matching the original metric function.

**Random Allocation** samples a fresh long-only weight vector from a symmetric Dirichlet distribution each day. It is a diagnostic control: its purpose is to check whether a learned policy behaves differently from uninformed feasible allocations, not to serve as an investable benchmark.

### Metrics

The submitted experiment reports:

- total return;
- annualized return;
- annualized Sharpe ratio with zero risk-free rate;
- maximum drawdown.

## Explainability

The original project applied Kernel SHAP to the deterministic PPO policy. In the public refactor the target action component is resolved from the exact asset ordering stored by the environment. This is important because multi-ticker data providers may reorder downloaded columns.

SHAP values answer a local model-sensitivity question: how changing the observation around a given state changes a selected action component. They do not establish causal relationships between technical indicators and future returns.

## References

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268).
