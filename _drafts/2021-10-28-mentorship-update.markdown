---
layout: post
title:  "ML Mentorship Update"
date:   2021-10-28
categories:
summary: A compilation of research and career advice I've given mentees over the last year.
---

I've been doing free office hours for non-traditional ML researchers for 2 years, usually during Sunday weekends.

This has culminated in one paper.




# Things that didn't work

- regular meeting cadence. most mentees could not keep up regular progress updates on a weekly basis, and after awhile I would never hear back from them.

- a slack channel. there was very little activity. I suppose organizing popular discords and slack requires some amount of community manager activity, something that I don't really enjoy doing.




# Combatting Adverse Selection

Unfortunately, students without PhDs tend to be not as good at doing research.  

Test

Focus on implementing the baselines.

- make sure mentee knows how to do well on a kaggle competition and can write code independently. You'd be surprised how little practical skills people can learn at university despite taking grad level ML classes.
- re-implement a paper


## Hypothesis Testing

Before you write code, it's good to always ask yourself "does the current set of observations have a consistent set of physical laws / cause-effect relationships that fully explain those observations"?

Why am I seeing the results I am seeing?

## General

## RL

-start simple with supervised learning: any time you solve a new problem, you will need to implement a BC baseline anyway, or some kind of value function (e.g. predict returns given a trajectory of actions and a current observation), so you might as well get good at tuning these things.

## Generative Models

All of ML can be cast into the framework of generative modeling on joint densities. Thinking in terms of density modeling cuts out a lot of the noise (i.e. cognitively motivated approaches to reinforcemet learning), and cutting out the noise is very important to being able to build large scale systems.



## On thinking simple

my gut instinct (and this is 30% experience and 70% wishful thinking) is that 1) log likelihood is enough and though other optimization objectives are reasonable, it is largely a distraction from scaling generalization compared to increasing data
and 2) trying to define generalization / OoD / robust ML / compositionality / etc. etc. is kind of a distraction as well due to the limits of language-based reasoning and the fact that we struggle to come up with mathematical isormorphisms to human perception and intuition
i.e. my personal motto these days is "don't overthink it, just train classifiers on harder and harder problems"  and that will be more directionally correct than trying to make things more principled

<!-- [colin-talk]: https://www.youtube.com/watch?v=iHWkLvoSpTg -->

## Things Undervalued by Cutthroat Academic System

not enough time is spent teaching how to collaborate effectively

Some advisers are "chill". Others push their student to the brink of tears. Both have their strengths and weaknesses. chill advisers will, on average, produce less successful students. More intense advisers will produce stronger students on average, with some students that are grateful (in hindsight) and others that burn out. 