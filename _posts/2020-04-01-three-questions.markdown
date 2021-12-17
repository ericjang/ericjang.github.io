---
layout: post
title:  "Three Questions that Keep Me Up at Night"
date:   2020-04-01
categories:
summary: A Google interview candidate recently asked me&#58; "What are three big science questions that keep you up at night?". This was a great question because one's answer reveals so much about one's intellectual interests - here are mine.
---



# Q1: Can we imitate "thinking" from only observing behavior? 

Suppose you have a large fleet of autonomous vehicles with human operators driving them around diverse road conditions. We can observe the decisions made by the human, and attempt to use imitation learning algorithms to map robot observations to the steering decisions that the human would take.

However, we can't observe what the homunculus is thinking directly. Humans read road text and other signage to interpret what they should and should not do. Humans plan more carefully when doing tricky maneuvers (parallel parking). Humans feel rage and drowsiness and translate those feelings into behavior.

Let's suppose we have a large car fleet and our dataset is so massive and perpetually growing that we cannot train it faster than we are collecting new data. If we train a powerful black-box function approximator to learn the mapping from robot observation to human behavior [1], and we use active-learning techniques like [DAgger](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf) to combat false negatives, will that be enough to acquire these latent information processing capabilities? Can the car learn to think like a human, and how much?

Inferring low-dimensional unobserved states from behavior is a [well-studied](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) technique in statistical modeling. In recent years, meta-reinforcement learning algorithms have increased the capability of agents to change their behavior in the presence of new information. However, no one has applied this principle to the scale and complexity of "human-level thinking and reasoning variables". If we use basic black-box function approximators (ConvNets, ResNets, Transformers, etc.), will it be enough? Or will it still fail even with a million lifetimes worth of driving data?

In other words, can simply predicting human behavior lead to a model that can learn to think like a human?

![mib_homunculus](/assets/threeq/MIB_homunculus3.jpeg)

One cannot draw a hard line between "thinking" and "pattern matching", but loosely speaking I'd want to see such learned latent variables reflect basic deductive and inductive reasoning capabilities. For example, a logical proposition formulated as a steering problem:

![complicated parking sign logic](/assets/threeq/parking.jpeg)

This could also be addressed via other high-data environments:

- Observing trader orders on markets and seeing if we can recover the trader's deductive reasoning and beliefs about the future. See if we can observe rational thought (if not rational behavior).
- Recovering intent and emotions and desire from social network activity.

# Q2: What is the computationally cheapest "organic building block" of an Artificial Life simulation that could lead to human-level AGI?

Many AI researchers, myself included, believe that competitive survival of "living organisms" is the only true way to implement general intelligence.

If you lack some mental power like deductive reasoning, another agent might exploit the reality to its advantage to out-compete you for resources.

If you don't know how to grasp an object, you can't bring food to your mouth. Intelligence is not merely a byproduct of survival; I would even argue that it is Life and Death itself from which all semantic meaning we perceive in the world arises (the difference between a "stable grasp" and an "unstable grasp").

How does one realize an A-Life research agenda? It would be prohibitively expensive to implement large-scale evolution with real robots, because we don't know how to get robots to self-replicate as living organisms do. We could use synthetic biology technology, but we don't know how to write complex software for cells yet and even if we could, it would probably take billions of years for cells to evolve into big brains. A less messy compromise is to implement A-Life in silico and evolve thinking critters in there.

We'd want the simulation to be fast enough to simulate armies of critters. Warfare was a great driver of innovation. We also want the simulation to be rich and open-ended enough to allow for ecological niches and tradeoffs between mental and physical adaptations (a hand learning to grasp objects).

- Chemistry? Cells? Ribosomes? I certainly hope not.
- How do nutrient cycles work? Resources need to be recycled from land to critters and back for there to be ecological change.
- Is the discovery of fire important for evolutionary progression of intelligence? If so, do we need to simulate heat?
- What about sound and acoustic waves?
- Is a rigid-body simulation of MuJoCo humanoids enough? Probably not, if articulated hands end up being crucial.
- Is Minecraft enough? 
- Does the mental substrate need to be embodied in the environment and subject to the physical laws of the reality? Our brains certainly are, but it would be bad if we had to simulate neural networks in MuJoCo.
- Is conservation of energy important? If we are not careful, it can be possible through evolution for agents to harvest free energy from their environment.

In the short story [Crystal Nights by Greg Egan](https://www.quora.com/What-are-the-strong-AI-ideas-in-Crystal-Nights-by-Greg-Egan), simulated "Crabs" are built up of organic blocks that they steal from other Crabs. Crabs "reproduce" by assembling a new crab out of parts, like LEGO. But the short story left me wanting for more implementation details...

# Q3: What Gives Rise to Time?

I recently read [The Order of Time](https://www.goodreads.com/no/book/show/38714658-the-order-of-time) by Carlo Rovelli and being a complete Physics newbie, finished the book feeling more confused and mystified than when I had started.

The second law of thermodynamics, $$\Delta S>0$$, states that entropy increases with time. That is the only physical law that is requires time "flow" forwards; all other physical laws have [Time-Symmetry](https://en.wikipedia.org/wiki/T-symmetry): they hold even if time was flowing backwards. In other words, T-Symmetry in a physical system implies conservation of entropy.

Microscopic phenomena (laws of mechanics on position, acceleration, force, electric field, Maxwell's equations) exhibit T-Symmetry. Macroscopic phenomena (gases dispersing in a room, people going about their lives), on the other hand, are T-Asymmetric. It is perhaps an adaptation to macroscopic reality being T-Asymmetric that our conscious experience itself has evolved to become aware of time passing. Perhaps bacteria do not need to know about time...

But if macroscopic phenomena are comprised of nothing more than countless microscopic phenomena, where the heck does entropy really come from?

Upon further Googling, I learned that this question is known as [Loschmidt's Paradox](https://en.wikipedia.org/wiki/Loschmidt%27s_paradox). One resolution that [I'm partially satisfied with](https://physics.stackexchange.com/questions/19970/does-the-scientific-community-consider-the-loschmidt-paradox-resolved-if-so-wha) is to consider that if we take all microscopic collisions to be driven by QM, then there really is no such thing as "T-symmetric" interactions, and thus microscopic interactions are actually T-asymmetric. A lot of the math becomes simpler to analyze if we consider a single pair of particles obeying randomized dynamics (whereas in Statistical Mechanics we are only allowed to assume that about a population of particles).

Even if we accept that macroscopic time originates from a microscopic equivalent of entropy, this still begs the question of what the origin of microscopic entropy (time) is.

Unfortunately, many words in English do not help to divorce my subjective, casual understanding of time from a more precise, formal understanding. Whenever I think of microscopic phenomena somehow "causing" macroscopic phenomena or the cause of time (entropy) "increasing", my head gets thrown for a loop. So much T-asymmetry is baked into our language!

I'd love to know of resources to gain a complete understanding of what we know and don't know, and perhaps a new [language to think about Causality](https://en.wikipedia.org/wiki/Linguistic_relativity) from a physics perspective.

Therein lies the big question: if the goal is to replicate the billions of years of evolutionary progress leading up to where we are today, what are the basic pieces of the environment that would be just good enough?

If you have thoughts on these questions, or want to share your own big science questions that keep you up at night, let me know in the comments or on Twitter! #3sciencequestions


