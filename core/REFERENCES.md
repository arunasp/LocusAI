# References

Sources for the claims in [ARCHITECTURE.md](ARCHITECTURE.md), grouped by
the design constraint each one supports. Cited so the biology can be
checked rather than taken on trust — the project's premise is that the
mapping is correct against real biology, which is only meaningful if the
mapping is traceable.

Entries give author, year, title and venue. Where a volume or page range
is not certain it is omitted rather than guessed.

## Levels of analysis

- Marr, D. (1982). *Vision: A Computational Investigation into the Human
  Representation and Processing of Visual Information*. W. H. Freeman.
  — the computational / algorithmic / implementational split this project
  targets at all three levels.

## Structural priority and fast pathways

- LeDoux, J. E. (1996). *The Emotional Brain: The Mysterious Underpinnings
  of Emotional Life*. Simon & Schuster. — the thalamo-amygdala "low road"
  acting before the cortical route completes.
- LeDoux, J. E. (2000). Emotion circuits in the brain. *Annual Review of
  Neuroscience*, 23, 155–184.
- Aston-Jones, G. & Cohen, J. D. (2005). An integrative theory of locus
  coeruleus-norepinephrine function: adaptive gain and optimal
  performance. *Annual Review of Neuroscience*, 28, 403–450. — tonic and
  phasic modes as an exploration/exploitation switch, and the basis for
  neuromodulatory control of competition sharpness.

## Discreteness and fixed action patterns

- Lorenz, K. & Tinbergen, N. (1938). Taxis und Instinkthandlung in der
  Eirollbewegung der Graugans. *Zeitschrift für Tierpsychologie*, 2. —
  the egg-retrieval movement that completes after the stimulus is removed.
- Tinbergen, N. (1951). *The Study of Instinct*. Oxford University Press.
  — releasing stimuli and the structure of fixed action patterns.

## Decision thresholds and commitment

- Green, D. M. & Swets, J. A. (1966). *Signal Detection Theory and
  Psychophysics*. Wiley. — the optimal criterion as a function of the
  relative cost of the two error types, which is what makes the
  commitment threshold scale with reversibility.

## Learning: encoding, consolidation, habit

- Schultz, W., Dayan, P. & Montague, P. R. (1997). A neural substrate of
  prediction and reward. *Science*, 275, 1593–1599. — reward prediction
  error as the encoding trigger, and its decay as prediction improves.
- Schultz, W. (1998). Predictive reward signal of dopamine neurons.
  *Journal of Neurophysiology*, 80, 1–27.
- Balleine, B. W. & Dickinson, A. (1998). Goal-directed instrumental
  action: contingency and incentive learning and their cortical
  substrates. *Neuropharmacology*, 37, 407–419. — the goal-directed
  versus habitual distinction underlying the separate procedural pathway.
- Graybiel, A. M. (1998). The basal ganglia and chunking of action
  repertoires. *Neurobiology of Learning and Memory*, 70, 119–136. —
  firing dropping out mid-sequence once a sequence is chunked.
- McGaugh, J. L. (2000). Memory — a century of consolidation. *Science*,
  287, 248–251.
- McGaugh, J. L. (2004). The amygdala modulates the consolidation of
  memories of emotionally arousing experiences. *Annual Review of
  Neuroscience*, 27, 1–28. — significance at encoding time modulating
  trace strength, which is why salience is an input to encoding rather
  than a later ranking pass.
- Hebb, D. O. (1949). *The Organization of Behavior*. Wiley. — the local
  plasticity rule underlying eligibility-trace learning.

## Memory systems, consolidation and reconsolidation

- McClelland, J. L., McNaughton, B. L. & O'Reilly, R. C. (1995). Why
  there are complementary learning systems in the hippocampus and
  neocortex: insights from the successes and failures of connectionist
  models of learning and memory. *Psychological Review*, 102, 419–457. —
  fast sparse one-shot storage alongside slow statistical storage.
- Nadel, L. & Moscovitch, M. (1997). Memory consolidation, retrograde
  amnesia and the hippocampal complex. *Current Opinion in Neurobiology*,
  7, 217–227. — multiple trace theory: both traces coexisting rather than
  one replacing the other.
- Nader, K., Schafe, G. E. & LeDoux, J. E. (2000). Fear memories require
  protein synthesis in the amygdala for reconsolidation after retrieval.
  *Nature*, 406, 722–726. — retrieval returning a trace to a labile state,
  which is why the read path is also a write path.
- Frey, U. & Morris, R. G. M. (1997). Synaptic tagging and long-term
  potentiation. *Nature*, 385, 533–536. — a weak trace captured by a
  nearby unrelated strong event; cross-item, non-local promotion.

## Working memory and capacity

- Fuster, J. M. & Alexander, G. E. (1971). Neuron activity related to
  short-term memory. *Science*, 173, 652–654. — persistent delay-period
  activity.
- Mongillo, G., Barak, O. & Tsodyks, M. (2008). Synaptic theory of working
  memory. *Science*, 319, 1543–1546. — activity-silent maintenance via
  residual presynaptic calcium.
- Cowan, N. (2001). The magical number 4 in short-term memory: a
  reconsideration of mental storage capacity. *Behavioral and Brain
  Sciences*, 24, 87–114. — capacity as interference and competition
  rather than a size budget.

## Sparsity and competition

- Olshausen, B. A. & Field, D. J. (1996). Emergence of simple-cell
  receptive field properties by learning a sparse code for natural
  images. *Nature*, 381, 607–609. — sparse coding through competition, so
  that recruitment is the computation rather than a masking step applied
  after it.
- Mocanu, D. C. et al. (2018). Scalable training of artificial neural
  networks with adaptive sparse connectivity inspired by network science.
  *Nature Communications*, 9. — sparse evolutionary training; unused
  connections ceasing to exist rather than being zeroed.
- Evci, U. et al. (2020). Rigging the lottery: making all tickets winners.
  *International Conference on Machine Learning (ICML)*. — gradient-guided
  regrowth outperforming random regrowth.

## Pattern separation and completion

- Hopfield, J. J. (1982). Neural networks and physical systems with
  emergent collective computational abilities. *Proceedings of the
  National Academy of Sciences*, 79, 2554–2558. — autoassociative
  attractor dynamics.
- Treves, A. & Rolls, E. T. (1994). Computational analysis of the role of
  the hippocampus in memory. *Hippocampus*, 4, 374–391. — dentate gyrus
  separation as a precondition for CA3 completion, and the capacity
  consequences of degraded separation.

## Sleep

- Tononi, G. & Cirelli, C. (2003). Sleep and synaptic homeostasis: a
  hypothesis. *Brain Research Bulletin*, 62, 143–150.
- Tononi, G. & Cirelli, C. (2014). Sleep and the price of plasticity: from
  synaptic and cellular homeostasis to memory consolidation and
  integration. *Neuron*, 81, 12–34. — global downscaling with gist
  extraction as its byproduct, distinct from replay.

## Method

- Nisbett, R. E. & Wilson, T. D. (1977). Telling more than we can know:
  verbal reports on mental processes. *Psychological Review*, 84,
  231–259. — the limit on introspective evidence: reliable for what
  happened, much weaker for why. First-person observation is used here to
  surface candidate mechanisms, never to confirm them.
