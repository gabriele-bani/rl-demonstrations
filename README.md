# Evaluating Demonstrations (the Good, the Bad and the Worse)

[![License](http://img.shields.io/:license-mit-blue.svg)](LICENSE)

## Description

Poster and Code for the project in [Reinforcement Learning](http://studiegids.uva.nl/xmlpages/page/2018-2019/zoek-vak/vak/63460) course of the MSc in Artificial Intelligence at the University of Amsterdam. Joint project of [Gabriele Bani](https://github.com/Hiryugan), [Andrii Skliar](github.com/askliar), [Gabriele Cesa](https://github.com/Gabri95) and [Davide Belli](https://github.com/davide-belli)
	
## Main Idea

Using **single** human demonstration has been shown to outperform humans and beat state of the art models in hard exploration problems [[Learning Montezuma's Revenge from a Single Demonstration](https://arxiv.org/abs/1812.03381)].

However, it takes an experienced professional to provide good demonstration to the model, which might be impossible in real problems. It might also be difficult to obtain optimal demonstrations. Can we still learn **optimal policies** from **sub-optimal demonstrations**?

## Approach

Basic **idea**: divide the trajectory in *n* splits. Train on the last one until convergence, then select the previous split. Repeat until the first split, so to learn from increasingly difficult exploration problems.

## Results

<p align="center">
  <img src="https://lh3.googleusercontent.com/hSxpIu3jzT1zvFF1zl7utrCFOvhLpbLfE6vOqIzZmtR5alThixhB5Cftw3B-e4YZgeSvo52G6K1IYOi7cVmsSSmTJdp-r_nx3adCexEGYKeeItyLbbOIkCwRsCo7VcM6acm0-Tp9wK0=w2400"/><br />
  <b>Figure:</b><i> Returns over episodes in Maze (left), MounainCar (middle) and LunarLander (right). </i>
</p>

##### 

- Non optimal demonstrations can lead to optimal results, but better demonstrations lead to better learning and give more reliable 
- In Maze, using bad demonstrations rather than suboptimal ones results in a better final policy because of a higher degree of exploration.
- With more complex environments, we expect demonstrations to allow for a much faster training than training from scratch. 
- The current implementation is very sensitive to hyperparameter choices; there is a need for a more automatic and reliable version of the backward algorithm to overcome this issue.

## Copyright

Copyright © 2018 Gabriele Bani.

<p align=“justify”>
This project is distributed under the <a href="LICENSE">MIT license</a>. This was developed as part of the Reinforcement Learning course taught by Herke van Hoof at the University of Amsterdam. Please follow the <a href="http://student.uva.nl/en/content/az/plagiarism-and-fraud/plagiarism-and-fraud.html">UvA regulations governing Fraud and Plagiarism</a> in case you are a student.
</p>
