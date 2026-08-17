# The Map
## How each sprint forced the next

H1 is not six independent benchmark reports. It is a chain of increasingly difficult alternatives.

<div class="expedition-table">
<table>
<thead><tr><th>Sprint</th><th>Question</th><th>What survived</th><th>Why the next sprint became necessary</th></tr></thead>
<tbody>
<tr><td><strong>S04</strong></td><td>What can explicit memory already do?</td><td>Full history is extremely powerful; format determines what is easy to retrieve.</td><td>Reading state is not maintaining state.</td></tr>
<tr><td><strong>S05</strong></td><td>Can explicit state persist across ticks?</td><td>The deterministic scaffold is stable; autonomous model writing is not.</td><td>A stable state machine still might add nothing beyond replay.</td></tr>
<tr><td><strong>S06</strong></td><td>Does incremental processing matter beyond final replay?</td><td>Deterministic state is replayable; model reconstruction is lossy; compact state bounds cost.</td><td>If scheduling alone adds no special continuity, perhaps quiet processing does.</td></tr>
<tr><td><strong>S07</strong></td><td>Can null-interval computation selectively consolidate knowledge?</td><td>The tested explicit write mechanisms self-pollute rather than consolidate.</td><td>If state is useful, does it at least causally govern behavior?</td></tr>
<tr><td><strong>S08</strong></td><td>Which representation controls behavior when history and state conflict?</td><td>Episodic history has far more leverage than structured state under balanced conflict.</td><td>If “current self-state” is not authoritative, what does the model treat as its own history?</td></tr>
<tr><td><strong>S09</strong></td><td>Can it track source ownership and know when its own attribution is wrong?</td><td>Ownership is weak and cue-dependent; no resolved positive Self-framing advantage over a matched observer.</td><td>Only a private recurrent channel can now test stronger continuity and privileged-access hypotheses.</td></tr>
</tbody>
</table>
</div>

## The river metaphor

Mnemosyne is the figure of memory. The river is temporal passage.

At the beginning, the question is:

> Does carrying memory through the river change the system?

By S06, the deterministic part of the answer is:

> Not if the archive and the transition rule let us reconstruct the same terminal state later.

By S08, the authority question becomes:

> Even if we can edit the dashboard, does the model treat the dashboard or the journey log as more authoritative?

By S09, ownership becomes:

> Does the system know which actor produced each line in the log—and is “you” any more than a role label?

## Scale of the first horizon

Across E03–E09, H1 includes:

- more than 4,000 live inference trials;
- deterministic, model-written, and replay-based memory conditions;
- 100-tick stability streams;
- 720 scheduled-versus-replay trials;
- 1,248 quiet-interval trials and 576 reflection traces;
- 800 State × Memory intervention trials;
- 320 E08 ownership trials;
- 320 E09 metacognitive probes;
- multiple protocol refreezes and post-confirmatory statistical repairs.

The volume matters less than the pattern: every major claim was repeatedly attacked by easier explanations.

<div class="interactive-lab" data-widget="journey-detail">
<div class="kicker">Choose a sprint</div>
<h2>What did each turn in the river change?</h2>
<div id="journey-detail"></div>
</div>
