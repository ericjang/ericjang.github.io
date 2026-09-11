---
layout: post
title: "Smooth Exponentials for Robotics"
date:  2026-09-10
summary: Predictions on where robotics is going in the next few years
---

I recently reread Dario Amodei’s essays "Machines of Loving Grace" and "Adolescence of Technology." In these essays and [his interviews](https://dwarkesh.com/p/dario-amodei-2), Dario speaks often of a "smooth exponential" of AI capabilities, as if machine intelligence "emerge\[s\] spontaneously from the right combination of data and raw computation."

It is sobering to contemplate what it means to experience a "smooth, unyielding increase in AI’s cognitive capabilities." Taken very literally, no single agent can really alter the smoothness of the curve, any more than a single atom can alter the course of an exergonic reaction. We could pull an all-nighter trying to rush that paper through, or we could shut down our lab and protest outside OpenAI and Anthropic offices, and progress would continue all the same[^gdp]. There are just too many ways for it to happen, so it is overwhelmingly likely to happen.

In scientific research, one typically expects progress to come in fits and starts. Research has a tendency to feel like a "hero’s journey," where a lone scientist or a small team gets stuck on a hard problem that no one has tackled before, suffers for a while, and then perseveres to create breakthroughs. From the point of view of a single project, progress can sputter out at any moment and must be encouraged with curiosity and persistence. Locally, research is a punctuated equilibrium.

And yet, capability increases in agentic coding models have not felt punctuated at all for the last year. Users of coding assistants have experienced a never-ending onslaught of [smarter frontier models](https://epoch.ai/benchmarks/frontiermath-tier-4-v2?view=graph&tab=release-date), as well as weekly open-source model releases that never lag far behind. The smooth exponential does not take vacations or suffer office drama or even encounter gifts of luck; it just climbs up steadily.

Why does progress feel so smooth and fast, despite individual research efforts being difficult? To use the language of statistical mechanics, when there are enough participating particles, the idiosyncratic behavior of any single particle is drowned out by the aggregate behavior of the ensemble.

## The Beginning of the Robotics Exponential

What does a "smooth exponential" look like for robotics? We will soon start to see a scale-up of robotic intelligence very similar to that of LLMs, perhaps compressed into the span of 12-24 months. Several key unlocks have recently happened that now make me believe that robotics is in the early days of a "smooth capability exponential." 

**More ecosystem participants:** There are enough people working on the right problems across university labs, well-funded startups, and industry research labs to create a sustained trend of progress. This is especially salient if you visit Chinese robotics companies. Every permutation of hardware form factor (biped, wheeled base, dexterous hand, gripper), data collection strategy, AI model architecture (WAM, VLA, etc.), and go-to-market strategy (consumer, enterprise, entertainment, research) has many strong companies pushing in that direction.

The Chinese electronics and hardware ecosystem has dramatically lowered the barrier to entry for entrepreneurs to create robots, and strong open-source VLAs are making it easier and easier to get robust capabilities with a bit of post-training data. New entrants can start further ahead, so there will be many more people advancing robotics.

**Modern LLMs can be evaluated for robotics capabilities:** Frontier models like GPT 6 Astra and Fable 5.1 are starting to do robotic tasks quite competently. It is notable that they are the same models used by millions of users around the world, rather than specialized models trained to control robots, so their capabilities are publicly verifiable across a wide variety of domains. Companies like Robocurve are starting to run [robotic manipulation capabilities directly on LLMs](https://openai.robocurve.org/gpt-6-astra/). It is only a matter of time before open-source models [catch up](https://qwen.ai/blog?id=qwenvla) in capabilities as well.

Because LLMs are now capable of performing robotics tasks zero-shot, I believe robotics evals will be dragged into the world of modern LLM leaderboards and inherit good practices like test-time scaling curves and extrapolative predictions for perplexity scaling. 

**A reusable large-scale training corpus:** One of the most profound breakthroughs in robot learning in the last few years is the discovery that the data that you scale up for broad generalization in robotics can look quite different from the data needed to control a specific robot.

The classical view in machine learning is that to generalize to domain *Y*, you need to scale up data on *Y*, or at least a broad dataset that "covers" *Y*. I used to believe that to solve home robotics, we needed millions of hours of teleoperated robot data from customer homes, paralleling the Tesla Autopilot strategy for urban driving.

It turns out that in the modern era, you can scale up generalization on a broad domain *X* to "learn intelligence," and only use a small amount of real robot data from *Y* in post-training to acquire control abilities. *X* transfers sufficient knowledge to *Y* despite a fairly large gap between the domains. Initially, *X* was just "other robot teleoperation" ([Open-X Embodiment 2023](https://arxiv.org/abs/2310.08864)), but soon became UMI-style grippers ([Sunday Robotics, 2025](https://www.sunday.ai/blog/no-robot-data)), followed by egocentric video ([Nvidia EgoScale, 2026](https://research.nvidia.com/labs/gear/egoscale/), [Dyna 2026](https://www.dyna.co/dyna-2)). If this trend continues, we may even start to see general pretraining for robotics happen on non-egocentric videos ([Rhoda 2026](https://www.rhoda.ai/research/scaling-web-video-pretraining)).

Because broad pretraining data for robotics is somewhat hardware-agnostic, we are starting to see a standardized pretraining corpus emerge for robotics. The exact composition varies between firms, but the rough orders of magnitude in 2026 are (X implies a single digit number):

- **~X million hours** of egocentric data
- **~X hundred thousand hours** of UMI gripper data
- **~X tens of thousands of hours** of teleoperated robot data, potentially across multiple form factors
- **~X hours** of high-quality, hardware-specific demonstrations for high-performance demos

Because pretraining is relatively decoupled from hardware specifics, the same pretraining corpus is likely to benefit all robotics labs, and therefore scaling laws on robotic imitation can be studied in a fairly reproducible way. A common "data pyramid" of hardware-agnostic data, in turn, unlocks action perplexity and action mean-squared-error scaling laws on large held-out datasets without requiring real-world robots.

In a talk in April to a small audience of about 100 people, I predicted that we would see the first 2-3 general-purpose home robots go on the market in June 2027, swiftly followed by a dozen or more companies in China and the US accomplishing the same thing. 

The first generation of such products will not be useful enough to replace entire humans in the home or otherwise, but will be general enough to be interesting to a few thousand enthusiast users. They might do basic pick-and-place tasks around the house, or fold your laundry into a neat pile, along with a small suite of party tricks. Like ChatGPT and coding agents, they will follow a progression from "not very useful" to "very useful" to "economically disruptive." The consumer home will be the first place that tolerates an "interesting-but-not-very-useful robot," but as capabilities transition to the "very useful" category, we should expect to see explosive adoption in enterprise and manufacturing at much larger volumes.


## What to Bet On

There are far more people building and investing in general-purpose learning robots today than there were 4 years ago. This is a wonderful time to be getting started and looking at robotics with a fresh set of eyes in the age of powerful coding agents.

One implication for investors looking for opportunities in robotics is that, in a smooth exponential, we may no longer be part of a conventional VC disruption paradigm, in which one player disrupts the entire market with a brilliant invention or innovative angle.

Locally, we will still encounter power-law returns among firms, but globally, the bulk of value creation is the smooth exponential of robotics progress arising from the massive influx of researchers and developers joining the field. I would prefer to bet on their collective success than against it! Returns will accrue to those who keep their eyes on the long-term growth of the robotics ecosystem rather than storytelling about how one atom in the mixture is going to win the entire market. I believe that there are massive returns to growing the entire robotics pie the way that Nvidia has done for compute, or Unitree has done for humanoid research. 

# Footnotes

[^gdp]: Loss of free will to alter the smooth curve sounds exaggerated at first, but these trends really do appear: 

    1. Smartphones and AI, as transformative and world-changing as they are, do not even register as "spikes" in global GDP. Starting around the Industrial Revolution, the aggregate growth rate for frontier economies (like the UK, and later the US) accelerated to roughly 1.5%–2% per year on a real per-capita basis—a compound rate that has held [strikingly steady for nearly 250 years](https://web.stanford.edu/~chadj/facts.pdf#:~:text=For%20much%20of%20the%20last%20century%2C%20the,%2450%2C000%20by%202014%2C%20a%20nearly%2017%2Dfold%20increase).

    2. Many AI papers have been published in the last 6 decades, but per Rich Sutton’s [Bitter Lesson](https://incompleteideas.net/IncIdeas/BitterLesson.html), increases in intelligence are mostly dominated by increases in compute spent (via algorithmic search and learning). Neural scaling laws formalize this a bit better: in aggregate, capability tracks compute.

    3. There is a "Bitter Lesson" for productivity itself, which is that increases in productivity (GDP) are [mostly correlated with energy input per capita](https://ourworldindata.org/grapher/energy-use-per-person-vs-gdp-per-capita). The GDP vs. energy consumption plot ensembles away the impact of scientific innovation and "improved energy efficiency multipliers" into a smooth trend across time. Invention, new methods, and key people were critical along the way, but zoom out far enough, and productivity is more about energy consumption than invention. 
