# Evaluating Demonstrations (the Good, the Bad and the Worse)

[![License](http://img.shields.io/:license-mit-blue.svg)](LICENSE)

## Description

Poster and Code for the project in [Reinforcement Learning](http://studiegids.uva.nl/xmlpages/page/2018-2019/zoek-vak/vak/63460) of the MSc in Artificial Intelligence at the University of Amsterdam. Joint project of [Gabriele Bani](https://github.com/Hiryugan), [Andrii Skliar](github.com/askliar), [Gabriele Cesa](https://github.com/Gabri95) and [Davide Belli](https://github.com/davide-belli)
	
## Main Idea

Using **single** human demonstration has been shown to outperform humans and beat state of the art models in hard exploration problems [[Learning Montezuma's Revenge from a Single Demonstration](https://arxiv.org/abs/1812.03381)].

However, it takes an experienced professional to provide good demonstration to the model, which might be impossible in real problems. It might also be difficult to obtain optimal demonstrations. Can we still learn **optimal policies** from **sub-optimal demonstrations**?

## Approach

Basic **idea**: divide the trajectory in *n* splits. Train on the last one until convergence, then select the previous split. Repeat until the first split, so to learn from increasingly difficult exploration problems.

## Results

![Results](https://cdn.pbrd.co/images/HT2ggjz.png)
##### Figure: Returns over episodes in Maze (left), MounainCar (middle) and LunarLander (right).

- Non optimal demonstrations can lead to optimal results, but better demonstrations lead to better learning and give more reliable 
- In Maze, using bad demonstrations rather than suboptimal ones results in a better final policy because of a higher degree of exploration.
- With more complex environments, we expect demonstrations to allow for a much faster training than training from scratch. 
- The current implementation is very sensitive to hyperparameter choices; there is a need for a more automatic and reliable version of the backward algorithm to overcome this issue.