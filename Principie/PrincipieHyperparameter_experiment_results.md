# Hyperparameter Experiment Results

Member Name: Principie Cyubahiro

Epsilon start was 1.0 in every experiment. The greedy average is the mean reward once epsilon has reached its floor, which reflects the trained agent. The random baseline to beat is 23.1 as per arXiv:1511.06581v3 [cs.LG] 5 Apr 2016 link: [https://arxiv.org/pdf/1511.06581](https://arxiv.org/pdf/1511.06581).

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 00. Baseline exp01 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Got the greedy average reward of 25.7, and best game got 48 rewards. Only just above the random baseline of 23.1, so the agent barely learned. The exploration and exploitation rewards were almost equal, which tells us the learned policy was no better than random play. This is the baseline tried to beat by tuning in the next experiments. |
| 01. Principie exp01 | 2.5e-4 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Increased the learning rate to 2.5e-4 to try to make the network learn faster, while expecting some instability risk. It came out slightly worse than the baseline: average reward 25 and best game 46, both just below the baseline’s 25.7 and 48. So the higher learning rate did not help here. |
| 02. Principie exp02 | 5e-5 | 0.99 | 32 | 1.0 | 0.05 | 100000 | After the higher rate failed, I lowered the learning rate to 5e-5, below the default. This helped: greedy average reward 26, the best so far, with a best game of 51. So a higher learning rate makes the Q-network unstable here, and keeping it low works better. I carried 5e-5 forward into the next experiments. |
| 03. Principie exp03 | 5e-5 | 0.997 | 32 | 1.0 | 0.05 | 100000 | After realising that lower learning rate helped, I decided to carry it for the rest of the experiments. Basically, on this experiment, I raised a discount factor or gamma from 0.99 (default value set) to 0.997 which also improved the performance where greedy average reward was 26.3 and the highest gotten reward was 52, and left it he best so far. |
| 04. Principie exp04 | 5e-5 | 0.999 | 32 | 1.0 | 0.05 | 100000 |Seeing the the higher gamma improving the experiments results, I increased it a bit to 0.999 and unfortunately got lower results for greedy average reward and the best in single game, those were 25.2 and 51 respectively. |
| 05. Principie exp05 | 5e-5 | 0.997 | 64 | 1.0 | 0.05 | 100000 |Since the discount factor of 0.997 worked better, I carried it again for the rest of experiments going forward and with that lr of 5e-5 but only changed the batch-size from 32 to 64 (which determines the number of experiences feeded into the training network) and got the best one with 28.4 for greedy average reward and 54 for the best in single game. |
| 06. Principie exp06 | 5e-5 | 0.997 | 128 | 1.0 | 0.05 | 100000 |Left the previous hyperparameters untouched except batch-size which I changed from 64 to 128 after seeing that the Q-network learns well with bigger batch size and as expected got the best ever performing agent which collected 30.5 as greedy average reward and 55 reward in the single episode |
| 07. Principie exp07 | 5e-5 | 0.997 | 128 | 1.0 | 0.01 | 100000 |At this point some parameters and their good values were going to be untouched, those are gamma, batch-size, and learning rate. Here, I only changed the epsilon end to 0.01 making an agent exploit mostly later in fewer episodes and did not get the best results at all. |
| 08. Principie exp08 | 5e-5 | 0.997 | 128 | 1.0 | 0.1 | 100000 | The approach used in experiment 8 did not help, therefore I tried to make an epsilon end a little bit bigger to 0.1 and got an improved ones with 29.4 greedy average reward and 43 as the best in single episode. |
| 09. Principie exp09 | 5e-5 | 0.997 | 128 | 1.0 | 0.05 | 50000 |In the nineth experiment, I ,ade an epsilon decay 50_000 which was to make an epsilon shrinks faster to let an agent use the learnt policies earlier and here are the numbers 26.4 and 50, gredy average reward and the best in single game respectively and that was not the best, the sixth experiment was still the champion. |
| 10. Principie exp10 | 5e-5 | 0.997 | 128 | 1.0 | 0.05 | 200000 | The last experiment, I increased an epsilon decay higher and these are numbers; 27.5 and 46, gredy average reward and the best in single game respectively. |

## Summary

Best experiment: exp10 won, despite not being the best by looking the numbers recorded during the experiments. We used each model to play a real ALE bowling game, the same environment launched during the experimentation, and won by having more rewards of 43.3, the highest ever achieved.

Mostly, the hypaparameters that helped arethe  batch size of 128, and lower learning late.
