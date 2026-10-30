# Breakpoints (seed 0, game breakout)

## Tiny step budget

| policy_steps | agent mean | random mean | ratio |
|---:|---:|---:|---:|
| 1000 | 0.450 | 0.500 | 0.90× |
| 4000 | 0.575 | 0.625 | 0.92× |
| 16000 | 0.675 | 0.375 | 1.80× |

## Imagination horizon

| horizon | policy_steps | agent mean | random mean | ratio |
|---:|---:|---:|---:|---:|
| 1 | 8000 | 0.675 | 0.475 | 1.42× |
| 5 | 8000 | 0.725 | 0.400 | 1.81× |

## Tiny model

Overrides: `{'latent_dim': 16, 'transformer_width': 32, 'transformer_layers': 1, 'actor_hidden': 32, 'critic_hidden': 32, 'transformer_heads': 2, 'cnn_channels': [4, 8, 16]}`

| policy_steps | n_params | agent mean | random mean | ratio |
|---:|---:|---:|---:|---:|
| 8000 | 73717 | 0.225 | 0.575 | 0.39× |

Honest note: breakpoints use 40 eval episodes on Breakout (seed 0) and a reduced step budget — they show where *this toy* collapses, not where the upstream 500k-step NanoDreamer fails.
