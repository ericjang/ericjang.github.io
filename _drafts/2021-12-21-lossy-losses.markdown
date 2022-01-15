---
layout: post
title:  "Embodied AI is Lossy Compression"
date:   2021-11-24
categories:
summary: The question of whether embodiment is necessary for realizing AGI can be understood as a debate between lossy and lossless compression.
---

The surprising degree to which GPT-X can reason competently about a world despite not having any explicit symbol grounding (e.g. "seeing")



Colin Raffel sent me an interesting essay I hadn't come across before, 

http://mattmahoney.net/dc/rationale.html


I ran into a DeepMind colleague at CoRL conference who lamented to me that the scale of models today is "excessive", and that embodied AI might be able to do the same tasks with a fraction of the compute, because the grounding is provided in a more explicit, constrained way (can only fit so much energy expenditure inside a body). I think "agents trained to solve embodied tasks" is to "large-scale generative models" as lossy compression is to lossless compression objectives.


likelihood models are 1:1 compression scheme via bits-back / entropy coding, so I guess there is still the question of what parameterization AI researchers actually should train on, despite all parameterizations being more or less AI complete. Should we train likelihood models? or train something that compresses? or train an RL agent that performs lossy compression, where the lossy parts are that which are irrelevant to life/death/survival within an environment?  
