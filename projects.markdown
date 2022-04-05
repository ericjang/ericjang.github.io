---
layout: page
title: Projects
permalink: /projects/
---

## Selected Publications

# Core Machine Learning

- 11/2016 [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144): A method for backpropagation through discrete categorical samples in neural networks.
- 10/2018 [WAIC, but Why? Generative Ensembles for Robust Anomaly Detection](https://arxiv.org/abs/1810.01392): We use generative models to perform out-of-distribution detection. Concurrently with Nalisnick '18, we show that likelihood models can suffer OoD problems when performing OoD tasks and improve their robustness with uncertainty estimation via ensembling.
- 7/2020 [Meta-Learning Requires Meta-Augmentation](https://arxiv.org/abs/2007.05549/)
Data augmentation techniques are ubiquitous in ML and often result in a substantial improvement in performance. We formalize "meta-augmentation" and show that you can apply it to pretty much any meta-learning problem and any meta-learner.

# Control and Robotics

- 9/2017 [Time Contrastive Networks](https://sermanet.github.io/tcn/): We came up with a really data-efficient, unsupervised feature representation for robotic imitation learning. Using this as a state representation, we can get a robot to imitate a human pouring liquids with a single video demonstration.
- 9/2017 [Deep Reinforcement Learning for Vision-Based Robotic Grasping](https://goo.gl/pyMd6p): We came up with a simulated benchmark for robotic bin picking from RGB images and evaluated 6 off-policy learning algorithms on it.
- 10/2018 [Grasp2Vec: Learning Object Representations from Self-Supervised Grasping](https://sites.google.com/site/grasp2vec/): Robots that teach themselves instance grasping without any object labeling.
- 6/2018 [QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation](https://sites.google.com/corp/view/qtopt): We use reinforcement learning to learn grasping from RGB images on real robots. Robot learns emergent manipulation behaviors like retrial and object singulation!.
- 5/2019 [Watch, Try, Learn: Meta-Learning from Demonstrations and Rewards](https://sites.google.com/corp/view/watch-try-learn-project/): We train an agent to imitate visual manipulation skills from raw pixels. The agent meta-learns how to re-try from failures, allowing it to fix its own mistakes in a single trial.
- 11/2021 [BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning](https://sites.google.com/view/bc-z/home): We train a mobile manipulation policy to empty bins, open doors, and follow unseen language commands using the simplest, task-agnostic method possible: behavior cloning teleoperated demonstrations from raw pixel observations.
- 3/2022 [Practical Imitation Learning in the Real World via Task Consistency Loss](https://arxiv.org/abs/2202.01862): We use the BC-Z network architecture to control a mobile manipulator to open doors. As end-to-end robotic learning scales up to more general settings, real-world evaluation becomes hard. The sim2real gap must be kept small if we are to trust simulated evaluation. This work proposes a novel CycleGAN-based data augmentation regularizer to keep the sim2real gap small.
- 3/2022 [Bayesian Imitation Learning for End-to-End Mobile Manipulation](https://arxiv.org/abs/2202.07600): The TCL work above used a self-supervised learning + data augmentation approach for closing the sim2real gap. We investigate a *model-centric* regularization approach via VIB, which brings up our latched door opening capability to 96% success across 10 different doors. 
- 4/2022 [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://say-can.github.io/): Language Models can not only tell jokes, they can also do "long term thinking" for robots! The benefits go both ways; the robot's value functions ground language sampling to what is realistic in its current context. We scale this to hundreds of mobile manipulation tasks in a real kitchen.


A full list of publications can be found on [Google Scholar](https://scholar.google.com/citations?user=JOYf6ygAAAAJ&hl=en&oi=ao).
