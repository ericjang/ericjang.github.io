---
layout: post
title:  "How to Understand ML Papers Quickly"
date:   2021-01-25
summary: Five simple questions to ask when reading any ML paper.
---

My [ML mentees](https://blog.evjang.com/2020/06/free-office-hours-for-non-traditional.html) often ask me some variant of the question "how do you choose which papers to read from the deluge of publications flooding Arxiv every day?” 

The nice thing about reading most ML papers is that you can cut through the jargon by asking just five simple questions. I try to answer these questions as quickly as I can when skimming papers.

# 1) What are the inputs to the function approximator?

E.g. a 224x224x3 RGB image with a single object roughly centered in the view. 

# 2) What are the outputs to the function approximator?

E.g. a 1000-long vector corresponding to the class of the input image.

Thinking about inputs and outputs to the system in a method-agnostic way lets you take a step back from the algorithmic jargon and consider whether other fields have developed methods that might work here using different terminology. I find this approach especially useful when reading [Meta-Learning papers](https://arxiv.org/abs/2007.05549). 

By thinking about a ML problem first as a set of inputs and desired outputs, you can reason whether the input is even sufficient to predict the output. Without this exercise you might accidentally set up a ML problem where the [output can't possibly be determined by the inputs](https://news.ycombinator.com/item?id=24173440). The result might be a ML system that [performs predictions in a way that are problematic for society](https://arxiv.org/abs/2002.06673). 

# 3) What loss supervises the output predictions? What assumptions about the world does this particular objective make?

ML models are formed from combining biases and data. Sometimes the [biases are strong](https://en.wikipedia.org/wiki/Linear_regression), other times [they are weak](https://lilianweng.github.io/lil-log/2020/08/06/neural-architecture-search.html). To make a model generalize better, you need to add more biases or add more unbiased data. There is [no free lunch](https://en.wikipedia.org/wiki/No_free_lunch_theorem). 

An example: many optimal control algorithms make the assumption of a stationary episodic data generation procedure which is a Markov-Decision Process (MDP). In an MDP, “state” and “action” deterministically map via the environment’s transition dynamics to “a next-state, reward, and whether the episode is over or not”. This structure, though very general, can be used to formulate a loss that allows learning Q values to follow the [Bellman Equation](https://en.wikipedia.org/wiki/Bellman_equation).

# 4) Once trained, what is the model able to generalize to, in regards to input/output pairs it hasn’t seen before?

Due to the information captured in the data or the architecture of the model, the ML system may generalize fairly well to inputs it has never seen before. In recent years we are [seeing more](https://en.wikipedia.org/wiki/AlphaGo) and [more](https://en.wikipedia.org/wiki/GPT-3) [ambitious levels of generalization](https://openai.com/blog/dall-e/), so when reading papers I watch out to see any surprising generalization capabilities and where it comes from (data, bias, or both). 

There is a lot of noise in the field about better inductive biases, like causal reasoning or symbolic methods or object-centric representations. These are important tools for building robust and reliable ML systems and I get that the line separating structured data vs. model biases can be blurry. That being said, it baffles me how many researchers think that the way to move ML forward is to reduce the amount of learning and increase the amount of hard-coded behavior. 

We do ML precisely because there are things we don't know how to hard-code. As Machine Learning researchers, we should focus our work on [making learning methods better](http://www.incompleteideas.net/IncIdeas/BitterLesson.html), and leave the hard-coding and symbolic methods to the Machine Hard-Coding Researchers. 

# 5) Are the claims in the paper falsifiable? 

Papers that make claims that cannot be [falsified](https://en.wikipedia.org/wiki/Falsifiability) are not within the realm of science. 


P.S. for additional hot takes and mentorship for aspiring ML researchers, sign up for [my free office hours](https://blog.evjang.com/2020/06/free-office-hours-for-non-traditional.html). I've been mentoring students over Google Video Chat most weekends for 7 months now and it's going great. 
