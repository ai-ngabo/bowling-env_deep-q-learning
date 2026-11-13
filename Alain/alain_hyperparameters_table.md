# Hyperparameter experiments: Alain

**Member Name:** Alain

Every experiment trains for 300,000 steps and changes one hyperparameter from the baseline (Exp 1). **Training last 20** = average reward of the last 20 training games (used to rank). **Greedy test** = the saved model playing 10 games with no random moves, like `play.py`. Random play scores about 23.

![Learning curves](Alain/reward_curves.png)

![Effect of each hyperparameter](Alain/hyperparameter_effects.png)

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Training last 20: 27.0. Learning curve stayed flat around 27. Games about 2350 steps, longer than normal: hesitates before throwing. Greedy test: one-button trick (presses only UPFIRE), 30 points. Why: the standard settings improved early and then stopped. In 300,000 steps the agent only found the UPFIRE trick, which already beats random play; aimed throws need much longer training. |
| 2 | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Training last 20: 23.4 (baseline 27.0). Learning curve rose to 33, then fell back to 23. Games about 2379 steps, longer than normal: hesitates before throwing. Greedy test: never throws (presses only DOWN). Why: updates 10 times bigger than the baseline overshoot, so the Q-values keep jumping and what it learned around game 50 was overwritten (unstable learning). By the end it preferred DOWN, which never throws. |
| 3 | 0.0005 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Training last 20: 20.3 (baseline 27.0). Learning curve rose to 31, then fell back to 20. Games about 2296 steps, longer than normal: hesitates before throwing. Greedy test: one-button trick (presses only UPFIRE), 30 points. Why: updates 5 times bigger than the baseline gave the same instability as Exp 2: the reward peaked mid-run, then collapsed below random play. Both large learning rates were my worst runs. |
| 4 | 0.00001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Training last 20: 30.2 (baseline 27.0). Learning curve improved, from 24 to 30. Games about 2155 steps. Greedy test: mixes buttons (DOWNFIRE 52%, FIRE 29%, UPFIRE 19%), 30 points. Why: updates 10 times smaller are slow but stable. It stayed near random for about 100 games, then improved and kept what it learned instead of overwriting it. From about game 105 it settled on a steady 30 points per game. |
| 5 | 0.0001 | 0.99 | 16 | 1.0 | 0.05 | 100000 | Training last 20: 27.4 (baseline 27.0). Learning curve improved, from 24 to 27. Games about 2297 steps, longer than normal: hesitates before throwing. Greedy test: rolls every ball into the gutter (presses only DOWNFIRE). Why: half as many samples per update makes each update noisier. The training reward ended at the baseline's level, but the final network's Q-values favour DOWNFIRE, so the noisy updates left an unreliable final model. |
| 6 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 100000 | Training last 20: 24.7 (baseline 27.0). Learning curve rose to 34, then fell back to 25. Games about 2340 steps, longer than normal: hesitates before throwing. Greedy test: never throws (presses only NOOP). Why: bigger batches should make updates smoother, but it peaked at 34 and then lost it. With one run per setting, part of this is probably bad luck. Its final network prefers NOOP and never throws. |
| 7 | 0.0001 | 0.98 | 32 | 1.0 | 0.05 | 100000 | Training last 20: 23.6 (baseline 27.0). Learning curve rose to 30, then fell back to 24. Games about 2326 steps, longer than normal: hesitates before throwing. Greedy test: one-button trick (presses only UPFIRE), 30 points. Why: looking about 50 steps ahead instead of 100 makes it harder to connect a throw with the pins that fall later, so learning was less stable than the baseline and ended lower. |
| 8 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 100000 | Training last 20: 27.3 (baseline 27.0). Learning curve rose to 34, then fell back to 27. Games about 2215 steps. Greedy test: one-button trick (presses only UPFIRE), 30 points. Why: keeping 10% random moves gave more varied experience (it reached the highest peak, 34), but the extra random moves also spoil throws, so its scores stayed mixed. Its final network still settled on the UPFIRE trick. |
| 9 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 150000 | Training last 20: 30.1 (baseline 27.0). Learning curve improved, from 22 to 30. Games about 2342 steps, longer than normal: hesitates before throwing. Greedy test: never throws (presses only NOOP). Why: exploring for longer gave it more varied experience and its training reward climbed steadily to 30, but its final network prefers NOOP, so the 5% random moves were doing the throwing. Good training score, unusable for the demo. |
| 10 | 0.0001 | 0.99 | 32 | 0.7 | 0.05 | 100000 | Training last 20: 25.4 (baseline 27.0). Learning curve rose to 31, then fell back to 25. Games about 2332 steps, longer than normal: hesitates before throwing. Greedy test: never throws (presses only NOOP). Why: starting with less random play gave less varied early experience. It improved early (to 31) but fell back, and its final network never throws. |

## Summary

- **Best:** Exp 4 (lr 1e-05), training last 20: 30.2.
- **Worst:** Exp 3 (lr 0.0005), training last 20: 20.3.
- **What helped / what hurt:** the learning rate mattered most. The smallest (0.00001) was the best and most stable; the two larger ones (0.0005 and 0.001) were the worst, because each learned something and then lost it. Batch size, gamma and the epsilon changes only moved the training score a few points around the baseline, which is within run-to-run noise.
- **Why greedy play collapses:** after 300,000 steps the six buttons have almost equal Q-values (I measured a typical gap of 0.001 to 0.008 between the best and second-best button, against Q-values around 1.3), so the highest one is the same button in every frame. During training the 5% random moves hide this. Exp 4 is the only model whose choice changes with the screen, yet it still settles on exactly 30 points, the same as the UPFIRE trick.
- **Greedy check:** 9 of 10 models press one button for the whole game.
- Single runs are noisy: differences of a few points may be luck rather than the hyperparameter.

## Final configuration

**Exp 4:** learning rate 0.00001, gamma 0.99, batch size 32, epsilon 1.0 to 0.05 over 100,000 steps. Model: `Alain/exp04/dqn_model.pth`.

![Best experiment training](Alain/best_experiment_training.png)

Why I chose it:
- Highest training score (30.2), and its learning improved and stayed, with no collapse.
- Steady game length (about 2,155 steps), so it throws without hesitating.
- The only model that reacts to the screen when playing greedily, like `play.py`.

Limit: it still scores exactly 30 per game, the same as the one-button trick, so 300,000 steps is not enough to learn aimed throws. For the group's final model, this lower learning rate should be trained for much longer.
