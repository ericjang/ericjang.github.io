---
layout: book
title: "New Book: AI is Good for You"
fulltitle: "AI is Good for You"
summary: "I wrote a book about the last 10 Years and the next 10 Years of Artificial Intelligence"
permalink: /book/
# redirect_to: https://www.amazon.com/dp/B0C1WRN9VR
---

![paperback on display](/assets/book/promo.jpg)
<!-- <img src="/assets/book/promo.jpg" alt="paperback" width="300"/> -->

[LINK TO PAPERBACK](https://www.amazon.com/gp/product/B0C9SH1GPV/) 

[LINK TO KINDLE](https://www.amazon.com/dp/B0C1WRN9VR)

[FEEDBACK AND ERRATA FORM](https://forms.gle/hJqZvxxQFRkHtm797)


When we think of “AI”, our imagination often drifts to scary places: the killer robots in Terminator and The Matrix and A Space Odessey. Even in sci-fi stories where the humans remain in charge, AI is often portrayed as an enabler of dystopian futures where technology has created a huge rift between the rich and the poor.

Given how bleak most science fiction covering AI is, it’s no surprise that most news media approach the topic with mysticism and trepidation. How do we control AI if it becomes smarter than humans? What kinds of jobs will our children have if all jobs can be done by robots? Journalists are often all-too-willing to amplify these doomsday scenarios to drive engagement.

Instead of adding fuel to the narratives about how AI technology can go horribly wrong, I want to share my optimistic take on how AI technology can make things go wonderfully right. It has the potential to create unfathomable abundance in everyone’s lives. Not only am I optimistic that human-level AI be good for society, but that it is technologically feasible within the decade. I will share a blueprint on how to do just that.

This book is divided into three parts: (1) where AI technology is today, (2) six ingredients for accelerating us towards increasingly general AI in the next decade, and (3) the societal consequences of achieving this. This book draws from the author's experience working as a robotics researcher at Google, as well as his current role as Vice President of AI at 1X Technologies.


# Contents

1. Introduction

Where is AI Today?

2. What is Intelligence?
3. Artificial Intelligence
4. Software 2.0: The Hard Parts
5. Four ways to create an AGI
6. The Neurobiological Software Stack

Ingredients for General Intelligence
7. Artificial Life
8. Jungle Basketball
9. Human-in-the-Loop Learning
10. Just ask for Generalization
11. Learning Robots in the Real World
12. Building an AGI Team

AGI and Humanity
13. Why AGI is Good for You
14. The Power Struggle for AI
15. Reality, Just the Way You Like It
16. AI Beauty
17. Project Ideas


# Who is this book for?

I wrote this book for the 14 year old version of myself, when I was starting to do science fair projects in high school and think about what it meant to be intelligent. I also wrote this book for the 22 year old version of myself, having just started my first research engineering job at Google and getting my feet wet in deep learning and robotics. This book is the culmination of all the practical wisdom I have on making AI systems work, accumulated over a decade of research projects. It will be useful for students who want to think about biological and machine intelligence in a unified way, or to technologists charting their own path to creating Artificial General Intelligence. 

# Introduction (First Chapter)

<div class="heading2">
<h1>Introduction</h1>
</div>

<div class="oneandhalf">
<p>Artificial Intelligence (AI) is the engineering discipline of creating a machine that is as smart as a person. No one quite knows how to do this yet, so this is also an active field of research. Because AI concerns the lofty goals of understanding and replicating intelligent behavior that is on par with animals and humans, it is quite the multidisciplinary field, spanning neuroscience, robotics, biology, physics, computer chip design, and philosophy.</p>

<p>When thinking of AI, the imagination often drifts to scary places. It might bring to mind the killer machines in The Terminator, The Matrix, and 2001: A Space Odyssey. Even in movies like RoboCop and Blade Runner where humans remain in charge, AI is still used to show a vast technological gulf between rich mega corporations and working-class people.</p>

<p>Given the bleakness of most science fiction covering AI, it’s no surprise that media also regard AI with mysticism and trepidation. How do we control AI if it becomes smarter than humans? What kind of employment will our children have if robots can do all the jobs? What if AI were to ingest prejudices from the sordid parts of the internet and parrot those prejudices back into our policing systems? These are common questions and concerns I get from friends and family. Members of the media eagerly amplify these doomsday scenarios to drive audience engagement.</p>

<p>Instead of adding fuel to the man-versus-machine narratives about how AI technology can go horribly wrong, I want to balance the conversation with an optimistic take on how AI technology can make things go wonderfully right. AI is not just about increasing the economic output of humanity; it will create abundance in everyone’s lives in big ways and small. This abundance will manifest in simple, everyday conveniences, like robots that free up your time by doing some of your daily chores to profound economic unlocks, such as expert-level personal tutors for eight billion people.</p>

<p>The general public can separate what they regard as “AI” into two categories: Narrow AI and General AI (or AGI). Narrow AI refers to software systems that can perform individual capabilities, but do not understand the world outside of that task. Narrow AI is already here, and powers many data-driven software like TikTok recommendations and self-driving cars. As amazing as they are, you cannot ask the TikTok algorithm to drive your car, and you cannot ask your car to recommend what to watch next. AGI refers to a truly human-level AI that can do everything a human can—if it existed, you could ask it to do all these tasks and more.</p>

<p>On the opposite side of the Doomers who say, “AI will end humanity,” is a plethora of serious AI researchers who doubt we will get to human-level artificial general intelligence (AGI) anytime soon. They have endured many AI booms and busts throughout the decades, with initial overexuberance rapidly tempered by the realization that AI is still woefully incompetent at tasks any human child could do. These researchers believe that despite great strides of the last decade, machines still need a handful of crucial missing ingredients to attain human-level intelligence. I think this time it is not the case—all the ingredients are within reach, and we can build an AGI within the decade.</p>

<p>I started writing this book in 2019 because I felt that despite the many research breakthroughs in narrow AI applications, such as translation and image understanding and robotics, not enough research existed on how to assemble all the pieces together. However, the four ensuing years (2020–2023) have seen such enormous strides in general-purpose AI models that the community is now ready to accept the possibility that we can have near-human level intelligence applied to many tasks.</p>

<p>ChatGPT, a helpful and intelligent chatbot developed by OpenAI and released to the public in late 2022, reached hundreds of millions of users worldwide in just a few months. Today, people from all walks of life (my mom included) use AI chatbots daily to draft emails and learn more about any topic. The relatively narrow AI systems built in the last five decades have been expanding in scope, to the point they don’t feel quite so narrow anymore. This book is the culmination of everything I believe to be true about the art and science of building general-purpose AIs, distilled over a decade of my working on it and thinking about it. The time is ripe to make a serious attempt at developing an AGI, and I hope this book convinces you of that.</p>

<p>I propose several technical ideas that will allow us to scale up technological capabilities quickly. These are presented in three parts:</p>

<ol>
<li>Part I looks at where AI technology is today.</li>
<li>Serving as my book’s centerpiece, Part II introduces the six ingredients I deem necessary for building AGI.</li>
<li>Part III explores the societal consequences of achieving AGI, addressing concerns about job displacement and killer robots that everyone inevitably wonders about.</li>
</ol>

<p>Building an AGI would be an earth-shattering technological achievement, to say the least, that would shape the course of human history in ways few can imagine. Certain risks and power struggles will arise from the creation of powerful machines. But I am fairly confident that if done right, such machines would help bring modern comforts to every person on the planet, provide companionship in old age, and assist all in leading happy and productive lives. There is so much more to AGI than simply sweeping bathrooms and balancing corporate budgets—such systems will be windows into new truths about the mind and the self. These entities will force humanity to confront its place and values in a vast universe.</p>

</div>

<div class="heading2">
<h1>Why AGI?</h1>
</div>


<div class="oneandhalf">
<p>The idea of creating artificial beings from nonliving matter has long captured the imagination of artists, philosophers, and engineers. An early example of such a creation is the golem from Jewish folklore—a clay being human-like in form and function, and said to be animated to life with an incantation. In the Greek myth tragedy of Pygmalion, a sculptor’s love for his statue is so profound that it enchants the statue to life. In the 1968 film 2001: A Space Odyssey, a mentally disturbed spaceship computer goes rogue and rebels against its mission, showing a disturbingly human-like fear of death when it is about to be shut down. The 2013 film Her portrays a complex and alluring operating system that emotionally outgrows its human companion. So, why does the story of creating artificial people, and all the possibilities and risks that entail, recur so frequently across the ages?</p>

<p>Perhaps it is about answering humanity’s great questions: Where did people come from? What makes us human? Why are we here? Until recent history, people turned to creation mythology to answer this question—gods, maize people, eggs laid by a water dragon and mountain fairy, and so on. Even though modern science has advanced our understanding of the chemical origins of life, these stories retain a sort of narrative beauty the laws of physics are incapable of telling. A character from the science fiction film Ex Machina remarks, “If you’ve created a conscious machine, it’s not the history of man. That’s the history of gods.”</p>

<p>“History of gods”—what a conceited claim! Still, what could be more human than narcissism and self-reflection? Science fiction stories of robot uprisings align with ancient mythologies of humans rebelling against their godly creators. These fictional portrayals of robots mirror human qualities, such as ego and vengeance, but also love and compassion. Myths of people creating golems and sentient robots contain allegories of hubris and of the dangers of toying with powers beyond our understanding.</p>

<p>Assuming the nervous system is a specialized computer for assisting in survival and reproduction, and we know how to build general-purpose computers in silicon, it should be possible to replicate a human mind in a silicon computer. We refer to this as “artificial general intelligence” (AGI): a machine that possesses the same level of aliveness, intelligence, and emotional capability as a human. Of course, the hard part is figuring out how to build one!</p>

<p>Despite the daunting challenges involved in creating AGI, my life ambition is to create a few AGIs of my own. You’re reading a speculative blueprint on how we might replicate key aspects of human-level artificial general intelligence. Still, the path ahead to achieving AGI will be long, challenging, and uncertain.</p>

<p>This “napkin sketch” of AGI should be accessible to any general audience interested in AI. I explain things in the simplest way possible without math equations or code. To some extent, it’s a ten-year plan for my career, and I believe it will encourage many aspiring AI researchers and engineers to pursue a similar path for themselves. Never has technology made it so easy for a handful of individuals to have a massive impact on the trajectory of human civilization.</p>

<p>I look forward to you joining me on this journey.</p>


<p>I look forward to you joining me on this journey.</p>

[Continue Reading](https://www.amazon.com/gp/product/B0C9SH1GPV/) 
