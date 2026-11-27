# Breakpoints (seed 0 unless noted). Not part of the status.

## Sequence length

Fresh traces, not the status set.

| rounds | length | vanilla | FBT | ALF |
|---:|---:|---:|---:|---:|
| 4 | 13 | 0.1750 | 3.3983 | 0.1671 |
| 8 | 25 | 0.3407 | 3.4191 | 0.3430 |
| 12 | 37 | 0.4276 | 3.3250 | 0.4204 |
| 24 | 73 | 2.0546 | 3.3311 | 1.7159 |
| 40 | 121 | 2.8285 | 3.0388 | 2.4102 |

## Jacobi passes

Vanilla stays at 1 pass. FBT and ALF use the listed pass count; the metric is the last pass.

| passes | vanilla | FBT | ALF |
|---:|---:|---:|---:|
| 1 | 0.4301 | 0.4541 | 0.4238 |
| 2 | 0.4301 | 0.5108 | 0.4239 |
| 3 | 0.4301 | 3.3274 | 0.4239 |
| 4 | 0.4301 | 3.6787 | 0.4239 |
| 8 | 0.4301 | 3.9197 | 0.4239 |

## Inference read (trained ALF, not retrained)

`nearest` is a uniform window of four eligible sources. `last` reads only the deepest sublayer output. Input injection stays on.

| read | vanilla | FBT | ALF |
|---|---:|---:|---:|
| full | 0.4301 | 3.3274 | 0.4239 |
| nearest | 0.4301 | 3.3274 | 0.4236 |
| last | 0.4301 | 3.3274 | 0.4240 |

## Depth (retrained, seed 0, same step budget)

Layer count 4 is the main run, copied here. Other depths are separate trainings.

| layers | vanilla | FBT | ALF |
|---:|---:|---:|---:|
| 1 | 0.4296 | 3.6069 | 0.4278 |
| 2 | 0.4157 | 2.5477 | 0.4174 |
| 4 | 0.4085 | 3.3519 | 0.4055 |
| 8 | 0.4154 | 2.2068 | 0.4145 |

## Low-rank memory (ALF retrained, seed 0)

Rank 4 is the main run. Ranks 1 and 16 are separate ALF trainings at the same width and step budget. FBT and vanilla are the main seed-0 numbers.

| rank | ALF nats | ALF params |
|---:|---:|---:|
| 4 | 0.4055 | 54376 |
| 1 | 0.4182 | 52840 |
| 16 | 0.4198 | 60520 |
