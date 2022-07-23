---
layout: post
title: "Making Robotics More like Generative Modeling"
date:  2022-07-23
summary: How we can learn from large-scale datasets for training robotic systems. However, this is not so much about how we take gradient steps or train our models, but rather how we as researchers can iterate quickly on such systems. I'll cover both algorithmic and engineering processes in service of this goal.
---

I recently [gave a talk (YouTube)](https://www.youtube.com/watch?v=lHXp6j6YrY4&t=4921s) at the [RSS'22 L-DOD workshop](https://sites.google.com/view/l-dod-rss2022). Here's the transcript and slides of the talk, in blog form, lightly edited. 

Since this is a workshop about large offline datasets for robotic learning, I don't need to convince the audience that when it comes to making robots that are capable, what often matters the most is the quality of the data. Data is especially helpful because it can help your software systems handle situations not seen prior to test time. In the broader field of machine learning, they call this "out of distribution generalization (OoD)". In robotics we call it "working in unstructured environments". They really mean the same thing.

We (the learning community) believe in the simplicity and elegance of deep learning methods, and we know that the recipe works. Here are some snapshots of works I've contributed to while I was at Google:

![rss1](/assets/rss22/slide1.png)

1. Qt-Opt can grasp objects not seen during training.
2. Grasp2Vec is goal-conditioned and can grasp objects not seen during training.
3. BC-Z is language-conditioned manipulation of objects that generalizes to unseen language commands
4. Door opening work at EveryDay robots where we can open doors from visuomotor policies and generalize to unseen doors.
5. SayCan can do even more language commands and use language models for planning.

I'm not even going to cover how these algorithms work, because that's not important. What really matters is that once you have a large diverse dataset, almost any mix of learning techniques (supervised, unsupervised, offline RL, model-based) should all work. I suspect that for any of these datasets, if you applied a different learning method to the same data, you can probably get the robot to do something that generalizes.

# Yes, but ...

All this progress is really exciting. In the future we'll have robots that are more capable and generalize to more things, but there's something that's been bothering me lately... if we look to our colleagues sitting a few desks away, progress in generative modeling is extremely astounding, especially since the development of GPT-3 and Scaling Laws. 

![rss2](/assets/rss22/slide3.png)

There's also developments like Transformer architectures and ever-larger datasets in NLP. The salience of the inputs and outputs of these generative models are really astounding. On the left you have some of the outputs out of ImaGen, a generative model made by Google. You can ask it to render "a hamster wearing an orange beanie holding a sign that says 'I love JAX'", and it will render it correctly. There's a large language model called PaLM now that can explain why jokes are funny. They train these models on really advanced hardware like TPUv4, and over in computer vision researchers are starting to develop some really advanced architectures like Vision Transformers. 

Looking at the pace of progress in generative modeling, I can't help but feel a little envious as a roboticist. I'm still training ResNet-18s, and that's an architecture that's almost 7 years old.

I know that Moravec's Paradox says that robotics is hard compared to the more cognitive-type tasks. Manipulation is indeed difficult, but intuitively it feels like being able to pick up objects and transport them is ... just not as impressive as being able to conjure the fantastical or explain jokes. So this talk is about comparing and contrasting some of the differences between generative modeling and robotics, and what we might learn from those fields to speed up the pace of our development.

Let me give a definition of what I think generative modeling is. It's not just about rendering pretty pictures or generating large amounts of text. It's a framework with which we can understand *all* of probabilistic machine learning. The core of generative modeling consists of two questions:

1. How many bits are you modeling. 
2. How well can you model them?

Starting in 2012 there was the AlexNet breakthrough - that was an image-conditioned neural network that predicts one of a thousand classes. If you take log2(1000), that's about 10 class bits. So you can think of AlexNet as an image-conditioned generative model over 10 bits. If you upgrade the difficulty of the task to MS-CoCo captioning, that's image-conditioned again, but this time you're generating about a tweet's worth of text. Let's say that's on the order of 100 bits. Dialogue modeling is similar (O(100) bits), except it's text-conditioned instead of image-conditioned. If you're doing image generation, e.g. text-to-image, that's on the order of 1000 bits. 

Generally, modeling more bits requires more compute, and that's why we see models being scaled up, but it also confers more bits of label supervision and more expressive outputs. As we train larger and larger models, you start to be able to exploit structure in the data so that you can learn much richer structure.

![rising-tide](/assets/rss22/rising-tide.png)

There is a great essay by Rich Sutton called (The Bitter Lesson)[] in which he talks about how most of the progress in AI seems to be riding on this rising tide of compute. I asked DALLE-2 to draw a depiction of this, where you have this ocean wave of compute that is lifting all the methodss up. You have Vision Algorithms, NLP, and Yann LeCun's "LeCake" all being buoyed up by this trend.

What gives us the most generalization in this regime? You have large over-parameterized models that can handle bigger datasets, and are able to attend to all the features in the prior layers (attentio, convolutions, MLPs). Finally, if you have a lot of compute and a stable training objective, it will almost always work. 

I asked DALL-E 2 to draw "a pack mule standing on top of a giant wave", and this is how I think of generative modeling taking advantage of the Bitter Lesson. You have a huge wave of compute, you have a "workhorse" that is a large transformer, or a modern resnet, and at the very top you can choose whatever algorithm you like for modeling: VQVAE, Diffusion, GAN, Autoregressive, etc. Scale and good architectures is what enables all that progress.

![rising-tide2](/assets/rss22/rising-tide2.png)

By comparison, this is what the state of robotic generalization looks like. Speaking for myself, I'm still training small architectures, I have yet to use a Vision Transformers yet, and here is the roboticist and their safety harness. 

![rising-tide3](/assets/rss22/rising-tide3.png)

I don't meant to sound excessively negative here. I work on robotics full time, and I want more than anyone for the robotics community to leverage a lot more generalization in our work.

In some ways this is not very surprising - if you look at the field of generative modeling, they don't have to work on all the annoying problems that roboticists have to deal with, like setting up the data problem and handling deployment. 

In any case I want to compare generative modeling to robotics in three different dimensions and examine how we can do things better: optimization, evaluation, and expressivity. 

# Optimization

Let me first start by explaining a simple generative model, and then cast robotics into the language of generative modeling.

![pixelrnn1](/assets/rss22/pixelrnn1.png)

consider a PixelRNN, which is an architecture for generating images. You start with a prior for your first pixel's first red channel. Your model tells the canvas (top row) what pixel (3-bit uint8) it wants to paint. Your canvas will be drawn exactly as commanded, so it copies the uint8 value onto the canvas, and then you read the canvas back into your model to predict the next channel - the green channel. You then feed in the R,G canvas values back into the RNN, and so on and so forth, generating RGBRGBRGB...

In practice for image generation you can use diffusion or transformers, but let's assume for simplicity it's a RNN that runs forward. 

Now let's cast the problem of general control as a PixelRNN. You want to draw an MDP - a sequence of states, actions, and rewards. You want to draw a beautiful MDP which corresponds to an agent accomplishing some task. Again, you start with a prior that samples some initial state, which in this case is the RL environment giving you some starting state. This is the first input to your model. Your RNN samples the first "pixel" (A), and again, the canvas draws the A exactly as you asked. But unlike the previous example where the canvas is always handing back to you your previous RNN outputs, now the next two pixels (R, S) are decided by this black box called "the environment", which takes in your action and all the previous states and computes R, S in some arbitrary way.

![pixelrnn2](/assets/rss22/pixelrnn2.png)

There's a "painter object" that takes your RNN actions but it draws most of the pixels for you, and it can be arbitrarily complex function. 

If we contrast this to the previous example, there is this more challenging setting where you're trying to sample the image that you want, but there is a black box that's in getting in the way , deciding what it's going to draw.

Furthermore, there's a classic problem in control where if your environment draws a state that you didn't really expect, then there's a question of how you issue a corrective action so you can return to the image you're trying to draw. Also, unlike image generation, you actually have to generate the image sequentially, without being able to go back and edit pixels. This also presents optimization challenges since you can't do backprop through the black box. 

Here's an idea - if we want to understand how RL methods generalize, we ought to benchmark them not with control environments, but apply them to image generation techniques and compare to modern image generation techniques. There's some work by Hinton and Nair in 2006 where they model MNIST digit synthesis with a system of springs. DeepMind has revived some of this work on using RL to synthesize smaller images. 

You can think of this benchmark as a classic autoregressive generation problem. you can inject your environment into the painter process by having the sampling of green and blue pixels (reward, next state) be some fixed black-box transition with respect to the previous pixels (state). You can make these dynamics as stateful as you want, giving us a benchmark for studying RL in a "high generalization" setting where we can directly compare them to supervised learning techniques tasked with the same degree of generalization.

Lately there's been some cool work like Decision Transformer, Trajectory Transformer showing that upside-down RL techniques do quite well at generalization. One question I'm curious about these days is how they compare to online (PPO) or offline RL algorithms (CQL). Evaluation is also conveninent under this domain because you can evaluate density (under an expert fully-observed likelihood model) and see if your given choice of RL algorithm generalizes to a large number of images when measuring the test likelihood. This would be a cool benchmark. 

# Evaluation

As an introduction to probability, if you want to measure the success rate of a robot on some task, you might model it as a binomial distribution over the likelihood of success given a random trial, i.e. "how many samples do you need to run to get a reasonable estimate of how good it is"?

The variance of a binomial distribution is $$ p(1-p)/N $$, where $$p$$ is your sample mean (estimated success rate) $$N$$ is the number of trials. In the worst case, if you have p=50% (maximal variance), then you need 3000 samples before your standard deviation is less than 1%!

If we look at benchmarks from computer vision, incremental advances have been super critical to drive progress in those fields. In ImageNet object recognition, a 10-bit generative modeling problem, progress has been pretty aggressive since 2012 - a 3% error rate reduction for the first three years followed by a 1% reduction every year or so. There's a huge number of people studying how to make this work. Maybe we're saturating on the benchmark a bit, but in the 2012-2018 regime, there was a lot of progress.

![imagenet-top1](/assets/rss22/imagenet-top1.png)

Similarly in other areas of generative modeling, researchers have been pushing down the perplexity of language models and likewise the bits-per-dimension of generative models on images.

![penn](/assets/rss22/penn_treebank.png)
![cifarbpd](/assets/rss22/cifar_bpd.png)

Let's compare some evaluation speeds for general benchmarks. The 2012 ImageNet object recognition test set has 150,000 images in the test set. It will take about 25 minutes to evaluate every single test example, assuming a per-image inference speed of 10ms. In practice, the evaluation is much faster because you can mini-batch the evaluation and get SIMD vectorization, but let's assume we're operating in a robotics-like setting where you have to process images serially because you have 1 robot.  

Because there are so many images, you can get your standard error estimate within 0.1% (assuming a top-1 accuracy of 80% or so). Maybe you don't really need 0.1% resolution to make progress in the field - 1% is probably sufficient. In practice it's really nice in ImageNet to be able to evaluate on so many diverse test situations and get within 0.1% standard error estimates of how good your model really is.

Moving up the complexity ladder of evaluation, let's consider evaluating neural networks for their end-to-end performance in some simulated task. Habitat-Sim is one of the faster simlators out there, and it's nice because it's very "bare metal" - there is not much indirection or overhead between the neural net inference and the stepping of the environment. It can step at 10,000 steps per second, and let's assume a typical navigation episode is about 200 steps. With neural net inference in the loop, that results in a 2 second evaluation per episode. 

If you want to evaluate an end-to-end robotic system with a similar level of diversity as what we do with ImageNet, then it'll take up to 4 days to crunch through 150k eval scenarios. It's not exactly apples-to-apples because each episode is really 200 or so inference passes, but we can't treat the images within an episode as independent validation episodes, because we only know whether the task succeeded or not. So we have to estimate success rate from 150k episodes, not images.

When I worked on BC-Z, each episode took about 30 seconds to evaluate in the real world, and we had a team of 8 operators who could run evaluations and measure success rates. each operator could do about 300 episodes a day before they got tired and needed a break. this means that if you have 10 operators, that gets you about 3000 evaluations per day, or roughly 1% standard error on your success rate estimates.

If it takes a whole day to evaluate your model, this creates a ridiculous constraint on your productivity, because you can only try one idea a day. You can't work on small ideas anymore that incrementally improve performance by 0.1%, because you simply can't measure those treatment effects anymore. You have to shoot for the moon and go for big jumps in performance. Which sounds nice but is hard to do in practice.

When you factor in the iteration process for doing robotic machine learning, it's very easy to have the number of evaluation trials dwarf those of your training data in the first place. For instance, if you're evaluating for 2 months, that's about 60k episodes! At that point you have so much new data that a completely different set of algorithms become appropriate. This is why I think that data collection is overrated - what actually is much harder is solving the problem of evaluation. 

![robot-scale](/assets/rss22/robot-scale.png)

A few years ago people were still tackling problems like getting arms to open singular doors. You didn't need to generalize too much, and these papers would evaluate on the order of 10 episodes or so. 10-50 trials is not actually enough for statistical robustness, but that's what roboticists do anyway. In BC-Z we did on the order of 1000 trials for the final evaluation. But what happens as we scale further? If we end up using datasets like Ego-4D to train extremely general robotic systems, like 100,000 tasks, how many trials would we need to evaluate such general systems? Once you have something kind of working, how do you continue to try ideas on those 100,000 tasks? The cost of evaluation here becomes absurd. 

We have enough data; the real world evaluation is the bottleneck!

![fish-eval](/assets/rss22/fish-eval.png)

## How to Speed Up Evaluation?

Here are some ideas on how we can speed up evaluation.

One way we can do this is to work on generalization and robotics separately. To a large extent, this is what the Deep Learning community does already, with sizeable communities of computer vision and generative modeling researchers that don't work directly with deploying their ideas to actual robots, but hope that their ideas transfer quickly to robots once their models acquire powerful generalization capabilities. One success story here is how ResNets, which were developed in Computer Vision, have dramatically simplified a lot of robotic visuomotor modeling.

Here's an "evaluation pyramid" with which you can separate different "tiers" of evaluation difficulty. 

![eval-pyramid](/assets/rss22/eval-pyramid.png)

On top of the "evaluation pyramid", you have a more domain-specific setup that is closer to the actual robotic task you are trying to solve. Iterating on this directly is very slow so you want to spend as little time here as possible. You'd hope that the foundation models you train and evaluate on the lower layers help inform you what ideas work without having to do every single evaluation at the top layer. On the bottom you have the most general perception benchmarks that are like Kaggle competitions. You can use all the benefits of iterating purely in the digital realm, study large-scale datasets and generalization more directly here. Moving up the stack, you can have some simulated benchmarks which study the problem in a "bare metal" way, with only the simulator and the neural net running, and all the code related to real world robotics like battery management and real-world resets are non-existent.

Again, the field already works this way. Most people who are interested in contributing to robotics don't necessarily *move* robots; they might train vision representations instead of work on semantic segmentation. The challenge here is that it's not clear to me how improvements in perceptual benchmarks map to improvemennts in robotic. policies. For example, if you're improving mAP metric on semantic segmentation or video classification accuracy, or even lossless compression benchmark - which in theory should contribute something eventually - it's not clear to me how improvements in representation objectives actually map to improvements in the downstream task. You have to eventually test on the end-to-end system.

There's a cool paper I like from Google called "Challenging Common Assumptions in Unsupervised Learning of Disentangled Representations", where a lot of progress in representation learning don't guarantee improvements in downstream tasks unless you are performing evaluation and model selection with the final downstream criteria you care about.

Another way to reduce the cost of evaluation is to make sure your data collection and evaluation processes are the same. In BC-Z we had people collecting both autonomous policy evaluation data and expert teleoperation data at the same time. If you're doing shared autonomy, you can use interventions to collect HG-dagger data to gather interventions for the policy, which gives you useful training data. At the same time, the average number of interventions you do per episode tells you roughly how good the policy is. Here's a plot showing how success rate scales as a function of interventions.

Another thing you can do is look at scalar metrics instead of binomial ones, as you require fewer episodes to get good confidence intervals. 

Autonomous RL is another natural way to merge evaluation and data collection, but it does require you to either use human raters for episodes or to engineer well-designed reward functions. All of these approaches will require a large fleet of robots deployed in real world settings, so this still doesn't get around the pain of iterating in the real world.

An algorithmic approach to evaluating faster is to improve sim-to-real transfer. If you can simulate a lot of robots in parallel, then you're no longer constrained. In work led by Mohi Khansari, Daniel Ho, and Yuqing Du, we developed this technique called "Task Consistency Loss" where we regularize the representations from sim and real to be invariant, so that policies should behave similarly under sim and real. The less the sim2real gap is, the more you can virtualize eval.

# Expressivity

If we compare the output configurations of modern generative models, a 64x64x3 RGB image at 8 bits per channel is 36864 bits. A language model can generate any number of tokens, but if we fix the output window to 2048 tokens, at 17 bits per token that's 36793 bits - comparable to the image models. 

As your models become super expressive, if they model the data distribution well, there is a large qualitative jump in how people think about these models. Some people are starting to think that Language Models are partially conscious because of how expressive they are.

Now how expressive are robotic policies and tasks today? In BC-Z most of our tasks consisted of about 6 objects on the table and the robot had to move one object on top of another or push some object around, for a total of 100 tasks. log2(100) is about 7 configuration bits. Given the state of the world, the robot is able to move the atoms into one of N states, where N can be described in 7 bits of information. Similarly, SayCan can do about 550 manipulation tasks, which is pretty impressive by current multitask robotics standards, but it adds up to about 10 configuration bits.

![robot-expressivity](/assets/rss22/robot-expressivity.png)

It's not a perfect apples-to-apples comparison because the definition of information is different between the two, but it's rather just to provide a rough intuition of what matters when humans size up the relative complexity of one set of tasks vs. another.

One of the challenges here is that our robotic affordances are not good enough. If you look at the Ego4D dataset, a lot of tasks here require bimanual manipulation, but most of our robots today we're still using mobile manipulators with wheeled base, one arm. It's a limited affordance where you can't go everywhere and obviously you only have one arm so that excludes a lot of the interesting tasks.

![limited-affordance](/assets/rss22/limited-affordance.png)

I think expressivity of our robotic learning algorithms are limited by our hardware. That's one of the reasons I went to Halodi - I want to work on more expressive robotic affordances.

![halodi-eve](/assets/rss22/halodi-eve.png)

The last point I'd like to make is that as our robots become more expressive, we are not only going to need Internet-scale training data, but also internet-scale evaluation. If you look at progress in LLMs, there are lots of papers that study prompt-tuning and what existing models can do. There's a paper called BigBench that compiles a bunch of tasks and asks what we can interrogate from these models. OpenAI does their DALLE-2 Beta and GPT-3 API. Their engineers can learn from users experimenting with their AI systems in the wild.

My question for the audience is, what is the robotics equivalent of a GPT-3 or DALLE-2 API, where the broader Internet community can interrogate a robotic policy and understand what it can do?

Here's a table that summarizes the comparison between optimization, evaluation, and expressivity:

||**Generative Modeling**|**Robotics**|
|Optimization and Generalization: can you compress the test set efficiently?|Model has complete control over which pixels it paints|Model samples an action and a stateful black box paints the next two tokens
|Evaluation: how quickly can you iterate?|O(25min) to get binomial success rate std <  0.1%|O(months) to get success rate std < 1%
|Expressivity: How rich are your outputs, in bits?|O(1000) bits make good use of scale and higher-capacity networks|Task configuration space about 10 bits, dramatically limited by robot morphology

# Q/A

