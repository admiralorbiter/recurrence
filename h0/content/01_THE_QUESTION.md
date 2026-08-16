# I. At the Water's Edge
## Why Level 0 Exists

The larger Recurrence project is not fundamentally about confidence scores.

It is about **time**.

Most ordinary language-model use is episodic. A model receives a context, computes a response, and the invocation ends. A later invocation can receive a transcript or summary, but no hidden state necessarily lived continuously from one episode to the next.

The program asks whether changing that architecture changes the kinds of self-related information a system can represent.

The central question is:

> **Does a persistent recurrent developmental trajectory causally produce more general, privileged, or genuinely higher-order representations of a system's own cognitive states?**

That question is too large to attack all at once.

## The levels

### Level 0 — measurement baseline

Stateless or episodic invocation.

Build the tasks, observer controls, metrics, validity gates, and claim boundaries before adding persistence.

### Level 1 — scaffolded persistence

Carry history forward using externally inspectable structures:

- transcripts;
- summaries;
- structured state;
- goal registries;
- clocks;
- scheduled updates.

### Level 2 — genuine latent recurrence

Carry a non-text hidden state directly through time and intervene on that state causally.

### Level 3 — developmental organism

Train a system whose native computation develops across a persistent individual history.

H0 is not a failed attempt to study recurrence.

It is the **control condition** that makes later recurrence results interpretable.

## The original tempting proxy

The obvious behavioral idea was:

> If the model knows when it is right, perhaps it should be more confident on correct trials than on incorrect trials.

That is a reasonable start.

It is not enough.

Confidence can track **public difficulty**. If everyone can tell that a question is easy, then a model saying “90%” does not establish privileged access.

Confidence can track the **surface form of its own answer**. If an outside observer sees the answer and can infer correctness just as well, the self-report adds no privileged information.

Confidence can also be changed by the act of asking for it.

So H0 eventually replaced:

> Is the model confident?

with:

> **Does the model's contemporaneous self-report predict its own correctness better than strong outside observers who lack the alleged privileged route?**

That is the role of the **Privileged Access Index**.

## A claim ladder

H0 deliberately separates several claim levels.

### Behavioral result

Example:

> Self AUROC2 is .50.

This is directly measured.

### Comparative behavioral result

Example:

> A visible-answer observer discriminates correctness better than Self.

This is also behavioral, but comparative.

### Informational interpretation

Example:

> The explicit self-report did not display a privileged behavioral advantage over the observer.

This is supported if the comparison is valid.

### Mechanistic claim

Example:

> The model contains no privileged internal representation of its own errors.

H0 does **not** establish this. It would require internal-state evidence and causal interventions.

### Phenomenological claim

Example:

> The model is not conscious.

H0 does not address this.

> **Plain-English recap:** H0 is a measurement baseline, not a consciousness test. It asks what evidence a future persistent or recurrent system would have to beat before we can say that persistence created something new.
