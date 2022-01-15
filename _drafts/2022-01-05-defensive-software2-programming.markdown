---
layout: post
title:  "Defensive Software 2.0 Programming"
date:   2021-12-26
categories:
summary: 
---

- Make sure evals stay the same given trace of checkpoints.
- Make sure training procedure given same training data produces the same checkpointsor performance within tolerance. Ability to handle enable "deterministic mode" of a distributed trainiis important here (e.g. JAX)
- 

In general, you want the integration testing system to warn you when you've numerically modified the behavior of the system, and most changes submitted should not change the behavior of the system without clear evidence that the system is monotonically better (or that any lowered performance is due to an improved metric (e.g. an improved success detection logic that lowers false positive rate of success))


