A deep Dive into Transformers

A tutorial on transformers, synthesizing a bunch of papers on the subject and explaining how they work.

Transformer Circuits

https://www.youtube.com/watch?v=KV5gbOmHbjU

A lot of robotics + control problems are converging to the same architecture.


Transformers as Efficient Non-Parametric Models

Non-parametric models, loosely defined, express distribution not with parameters, but in terms of trainingdata.

Transformers are thought of as parametric models.

show that Query-Key-Value attention is essentially equivalent to locally weighted regression (Atkeson, Moore, Stefan Schaal 1997)
with values corresponding to labels and keys corresponding to look-up table embeddings.

Stacking QKV attention (or LWR) repeatedly amounts to the ability to query large parts of the training set iteratively as you go deeper.

Related works:
- Retro

BYOL - ammortize inference on negatives
simclr is the "non-parametric" equivalent of BYOL

prototypical networks for few shot learning - prototypes are like neighbors, you do retrieval

