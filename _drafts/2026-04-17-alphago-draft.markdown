---
layout: scrollytelling
title: "AutoGo : Automated Go Research"
date:  2026-04-12
---

To teach myself the "agentic LLM way of doing ML research" and learn something out of my comfort zone, I've been spending the last few months implementing AlphaGo from scratch. I've long admired the AlphaGo breakthrough, and wanted to understand it at a deeper level beyond reading the paper.

<TODO: visualization of go games being played by the bot against Katago>

Modern AI coding assistants make it possible to do a large amount of research and engineering with a very small team. In this case, just one person. I wanted to figure out how to create a highly performant AI system from scratch on a modest computational budget.

[Github repo](https://github.com/ericjang/autogo)

[Play Online](https://autogo.evjang.com)

The current version hosted online was trained with a budget of less than $500 of rented spot compute. A big thank you to Vincent Weisser of [Prime Intellect](https://app.primeintellect.ai/) for donating the GPU credits! 

This blog post is a tutorial that explains not only how AlphaGo works, but also what I've learned designing autoresearch infrastructure for Go, but the research engineering lessons I picked up that broadly transfer to other domains. 

## Why Go?

Why choose Go as a testbed for auto-research? 

Fast iteration and rapid feedback are very important in deep learning research. Go strikes a balance of being a challenging problem with a high skill ceiling and great strategic depth, yet being very quick to iterate on and evaluate. You can get interesting behaviors out of models with only a few million parameters.

Despite the game's simplicity, AlphaGo covers a "sampling platter" of modern deep learning topics: distributed RL systems, neural network architectures, training-time and test-time scaling laws. Additionally, self-play, Nash equilibria, mixed strategies, synthetic data and recursive self-improvement are top-of-mind for frontier labs, so I think Go is a lightweight yet rich environment for studying those dynamics.

All of these lead me to think that this sort of code will be a useful harness / environment for frontier autonomous research agents. Again, iteration speed is everything, even in frontier labs, so the goal of this codebase was to build "the simplest and fastest way to study interesting deep learning problems of all kinds using Go as the testbed". Just as the game itself has a high skill ceiling, there are many aspects of the "system" that can be tuned for performance, so "automated Go research engineer" is also a job with a high skill ceiling.

Finally, I have always found it deeply profound that simply querying a function approximator for value can be an arbitrarily accurate replacement for simulation. It is a miracle that macroscale effects can be predicted accurately without microscale simulation. Extrapolating this principle, I wonder if long-standing questions of computational hardness (P = NP?) are even the right ones to be asking.

## Questions

The wall clock time of training a GPT-2 level LLM has fallen by [nearly two orders of magnitude](https://github.com/karpathy/nanochat?tab=readme-ov-file#time-to-gpt-2-leaderboard) in 7 years. Is the same true for Go?

AlphaZero was trained on thousands of TPUs for several days. [KataGo](https://arxiv.org/abs/1902.10565), the strongest open-source Go bot to date, was trained on dozens of V100s in about 19 days. To accomplish this, Katago used a number of tricks to improve computational efficiency (playout cap randomization, policy target pruning, global-pooling, auxiliary policy targets, ownership and score targets). However, this comes at the expense of complexity and readability. 

Can we simplify AlphaGo to few hundred lines of code that can still achieve frontier performance on a modest compute budget? Does the Bitter Lesson just take care of things 10 years later? What is the tradeoff between "simple scaling" and "compute multipliers" here? We will implement basic AlphaGo first, and then overlay the optimizations later.


# AlphaGo Tutorial

The first thing you can do when starting out is to take a strong expert (human or go bot) and collect a bunch of gameplay playing against itself. Then predict what an expert player does. As you scale up this data, you should be able to match human expert. 

However, you should take care to know that "imitating the expert" is NOT how alphago scales to superhuman play. At best, you can match its performance, but additional games will rarely give you the signal you need to actually surpass it. 




katago v katago with a komi of 7.5 yields 

"imitate the winner" 





## Pitfalls

LearnAlphaGo codebase has a full set of experiments I ran in setting up the project.

You can see that I spent a lot of time tuning things and trying to automate long-running experiments before I fully understood the key learning signal that one needed to optimize for.


# The Recipe

I tried a lot of things, but the ultimate recipe ended up being quite simple, and actually came down to properly understanding how the training dynamics of Go differ.



The "cheat" recipe

1a. collect MCTS(your network) vs. MCTS(your network). For the first iteration, you can initially replace your own network with katago to collect some high quality training data.
1b. Evaluate MCTS(your network) against katago.
2. Make sure MCTS generates enough "disagreement" from your policy network (i.e. top choice of MCTS does not match top choice of your policy network).
2. train policy + value network from the data. This improves your policy network to guess better. If you want to cheat and skip the "tabular rasa" part, you can use katago as part of your "self-play league" and train on the katago games as well
3. Go back to step 1, repeat.


It's more important to have high quality MCTS moves that supervise your policy network than have a sheer volume of games, unlike in RL for robotic locomotion.




## How to go about a new problem

At a meta-level, here are the steps I took to discover the above.

This section can be thought of as my "personal research taste"

There is a tension between a number of factors, often very related to the "explore" vs. "exploit" tradeoff in RL.
- it's wise to make sure infra is good, but you don't want to spend all your time polishing infrastructure that is for the wrong goal
- there's a tension between "closing the loop as fast as possible to learn what you don't know" and "doing every step systematically and making sure the foundations are correct"
- there's a tension between "scaling it up" to take advantage of the bitter lesson and the raw benefit of compute, and missing the key bottleneck and wasting a bunch of resources when the smaller-scale projects didn't work

- it can be very tempting to change 3-4 things between experiments, instead of a controlled baseline with only one idea. Maybe you don't have enough compute & time to do everything systematically, and you're *fairly confident* that the ideas are good. 



The first thing to do is to gain some intuition for how to play the game. I vibe coded a basic Go game.

### Be paranoid about correctness

This is especially important in the LLM coding era - it's so easy to skip checking the correctness of code and just to ask Claude to write the new feature.

If you ask claude to autonomously speed up a function, you have to check that the function still passes all the parity tests.

Sometimes "yolo" just works, but it often relies on good initialization where most things are already correct and working (see next section). 

### Good initialization

Never start from something "not working" and try to make it work -> every step should have something that "works".

When trying out new ideas, starting with a good initialization saves a lot of time. I almost always started my experiments with the best initial data collection policy I had, and then later relaxed that requirement.

Before I had my own models, I would use katago to generate that data. 

Don't try to solve Go tabular rasa on day one - even AlphaGo had to bootstrap off of human expert games to start. 

It is kind of like in bouldering gyms, where you might start midway through the route to see if you can finish it, and then solve it from the start once you are confident you can do it from halfway. 

Generally, if you want to beat some high score or attain some difficult capability, you want to first start by cheating a bit, and see if you can attain the high score when everything is handed to you on a silver platter. 

Bug-free is important 

Several months into the project I realized that my katago baseline was weak because I was not informing it of the correct "komi", which is an extra bonus score given to white to correct for black's advantage of going first. This led to katago playing not as aggressively as it should have, because it was under the impression it only had to win by 0.5 points, not 7.5 points. Oops! 

There were also some other bugs on katago using Tromp-Taylor instead of Chinese scoring rules, which are simpler to implement.

## Spend time optimizing for fast iteration

If you find yourself waiting for something, and you see yourself having to run that process over and over again, it's worth optimizing that loop to be faster.

## Interpretability

By interpretability, I mean "you have to know what's going on in as much detail as you can"

Generally favoring methods that allow more debugging and analysis (e.g. synchronous alternating collect vs. training, the ability to backtrack to a stationary distribution)

One of the ah-ha moments for me was to start measuring the "% of time MCTS disagreed with the policy network it was using".

## Weight Decay

When things are not working, it's very tempting to keep adding configs and new features, and before you know it, you have a monstrous codebase with a lot of non-useful knobs and dials. The complexity increase will then slow both the LLM assistant as well as you down when it comes to extending the codebase or building understanding. 

If you have a working setup and configuration, it's worth trying to prune away other model architectures that didn't work, reducing branching.

## Performance Optimization

I spent a long time trying to keep the MCTS code simple, avoid multithreading and virtual losses, but ultimately the number of samples was the key bottleneck, and there was no getting around it. Multithreaded MCTS in C++ with virtual losses to prevent threads from over-exploring the same path is [considerably harder to read]() but also offered a 11x speedup, which made it worth it in the end.

It's interesting to contemplate the possibility that we may just bypass python all together and go from Markdown -> Rust / C++ / Cuda kernels directly in the near future, perhaps with python for analysis, plotting, etc.


It's very helpful to start with alternating synchronously between train and collect jobs before attempting to max throughput with async RL and simultaneous training + data collect. Helps a lot with catching stability issues in training, which are much harder to diagnose / backtrack in async mode. Once you get synchronous baseline working, then you can look into speeding things up with async.


Python is only kept for human alignment purposes, so that we have code that is legible to our understanding.


Inspect the learning signal

Important to be very clear about what we are learning. we are not self-imitating successesful rollouts, we are self-imitating the "MCTS improvement over the current policy network" 

MCTS essentially discovers what the policy does not know


It's off-policy in that you can take old games, and apply the *current* policy network + MCTS search to those to get new labels. You can even take winning game perspectives from other agents, but you need to use the *current* policy network to search on those board states to get your new MCTS labels

Animation about parallel scheduling across games (massively parallel) and within MCTS search (sequential in depth, parallel across leaves)
show each board move yielding groups of 8 leaves at a time sent to a big shared inference engine


## Mistakes

### Research Mistakes

- the biggest one by far was a conceptual misunderstanding of the difference between iterative self-play training (imitate the winners) vs. MCTS policy improvement (predict the outcome of MCTS search so you can search better)

- I came up with a halfway idea which was to save out every MCTS branch, since that essentially comprised a "short game all the way to the leaf node". However, this is unnecessary, because the MCTS policy distribution already captures all the information.


### Good Engineering Decisions

- dev container and docker workers for running cpp / katago compiled dependencies, instead of hacking the compilation into ray scripts.
- Frequently checking the performance bottlenecks and running `/optimize` loops to tune the throughput of games

- Building distributed systems with reproducibility. 

If you have asynchronous pushing / pulling from a replay buffer, this can affect the variance of your updates. Ensuring that the system has "soft synchronization" within some tolerable push/pull ratio (e.g. it never falls below 0.8 or exceeds 1.2) is a good practice that makes experiments more reproducible. This allows you to de-couple your experiment from the hardware used to run it.


### Engineering Mistakes

Over-engineering:

- building a soft-sync distributed RL setup with replay buffers, protobuf serialization. At the end of the day, a simple 
- using Ray to orchestrate jobs and distribute work. having claude figure out how to read ray logs of launched jobs to get output instead of just running a `ssh "docker exec ..."` command directly to the remote host. I just implemented my simple "job pool"
- Building a "hill climbing" dashboard when there still weren't great signs of life, and serializing everything to a persistent eval DB.



for me, I started with using katago as a baseline that I could generate high quality expert data, and then later see if I could relax that. Starting with an "easy" problem where everything is handed to you on a silver platter allows you to quickly notice when things are not working as expected due to bugs.

Even though I used katago, I wanted to figure out as many things for myself and see how simple I 

1. This is probably the most important lesson I have, which is that you always want to be starting from something that works, and trying to make it better, rather than trying to go from zero-to-one: something that doesn't work and you have no baseline.

2. you often want to start with as good of an initialization as possible, basically "one step away from success". So I began my experiemnts mostly 


## Open Problems



Optimizing throughput

- leaf node parallelism can only be so large - if you make it too big, then you start to explore down actions that are actually suboptimal, and it biases the MCTS distribution towards being "diffuse". So the more leaf parallelism you add, you actually need to adjust down the temperature of choosing actions that are not the best one so far.

Therefore, it makes sense to evaluate other games concurrently


## ExpliGo Demo

- makes a move
    - explain *why* that move was picked
        - MCTS N visit count distribution heatmap
            - raw policy "intuition" starting point
            - simulations: of each move, estimate how many of the resulting games result in win, how many result in loss

- predict future occupancy of all boards

LLM: explain why is a board position 


