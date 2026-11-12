# Evaluation planning decisions

Working plan agreed on 2026-10-10. This records proposed work, not launched
experiments or completed findings. Experiment configuration remains in
`/home/paul/Documents/ConceptBasedLearning`.

## Narrative and environment roles

- Organize evaluation by competence, RQ1 robustness, RQ2 frozen-skill
  transfer, and RQ3 robustness/transfer under tandem training. Integrate
  ablations into the claims they assess.
- Use Seaquest for the detailed architectural comparison. Acknowledge its
  role as the principal development environment.
- Name This Game is the preferred additional training environment for
  assessing whether the main patterns extend beyond Seaquest.
- Keep Time Pilot's final analytical role open. Its relative simplicity may
  limit its value for mechanical robustness. Judge its role from task demands
  and modification coverage; retain completed results regardless of direction.
- Keep Freeway and Phoenix visible in applicable competence, robustness,
  and transfer results. Do not assume why learning is unstable.
- NEXUS supports Seaquest and Kangaroo. Kangaroo is out of scope after
  unsuccessful training attempts, so NEXUS remains Seaquest-only.

## Structured full-context OC skill comparison

Currently omitted from the final suite. Independently embedding object rows
and then gathering only the target does not expose surrounding objects jointly
to the skill policy and is not a substitute for this proposed comparison.

If implemented, give skills the entire structured relative object list,
using the OC meta-policy information as the reference, plus explicit target
conditioning. Document global features, masks, roles, and ordering. Ensure
non-target rows can influence skill actions. For transfer, define a compatible
schema and treatment of different object counts.

The intended contrast is restricted target-centric information versus access
to additional environmental context, under the same heuristic, skill rewards,
POV action convention, training budget, and evaluation protocol where possible.
Document unavoidable network differences. This differs from enriching only
the selected target's features.

The specific question is whether information restriction adds robustness to
layout changes beyond hierarchical control and object-centric skills defined
through their rewards. Both variants retain that decomposition and reward
design. A structured full relative object list tests whether the restriction
is needed in the tested settings; it does not test reliance on absolute screen
coordinates if both variants exclude them.

Interpret outcomes symmetrically:

- Comparable competent base policies, restricted skills stronger under layout
  changes: supports a benefit from restriction in those settings.
- Comparable competence and layout robustness: restriction is not required
  for the observed robustness in those settings; it may still offer simplicity.
- Full-context skills more robust: additional context helps in those settings.
- Full-context skills fail to learn competently: this does not isolate a
  robustness effect; report the acquisition limitation.

Pixel skills already probe broad contextual input within the hierarchy, but
they also introduce visual encoding and network differences. Their results are
relevant evidence, not an equivalent isolation of the object-context question.
Neither comparison establishes universal necessity of restriction or isolates
hierarchy from skill reward design as separate causes.

This comparison is particularly valuable if the thesis attributes an advantage
specifically to removing object context: pixel skills also change modality and
network architecture, while monolithic OC changes the control structure.
Without it, retain the narrower claim supported by those comparisons.
Do not interpret failure of an inadequately structured input as evidence that
extra context causes overfitting. If the structured variant fails to learn,
report an acquisition result rather than an isolated robustness effect.

## Candidate follow-up runs (not launched)

- [ ] Audit completed final-suite coverage before adding experiments.
- [ ] Consider structured full-context OC skills on Seaquest first if the
      direct information-restriction contrast remains central to the claim.
- [ ] Extend matched skill-input and no-POV comparisons to Name This Game:
      restricted ego reference, target-marked pixels, no-POV, and structured
      full-context OC if implemented. Reuse existing monolithic/tandem runs
      where protocols match.
- [ ] Consider the same non-NEXUS comparisons on Time Pilot if its task and
      modifications provide an informative test of the relevant claim.
- [ ] Evaluate source-trained pixel/no-POV/full-context OC variants on
      selected transfer targets where interfaces are compatible. State source
      competence and use the same target scaffolding.
- [ ] Decide whether additional tandem skill-input ablations address an
      unresolved question; the current tandem suite does not test every variant.

Use short training pilots to check implementation and learning viability.
Distinguish training budget from evaluation episode horizon. For final
comparisons, match relevant budgets and protocols or label a separate budget
study. A short failed pilot does not establish inferior asymptotic competence
or robustness. Do not tune comparison horizons to favor observed results.

## Writing follow-ups

- [ ] Align the introduction's old RQs with robustness, transfer, and tandem
      robustness/transfer. Competence is now a prerequisite analysis.
- [ ] Replace the methodology's undefined OC variant if the structured
      full-context comparison is implemented.
- [ ] Replace evaluation/discussion TODOs with completed-run evidence and
      final coverage. Keep native training a reference rather than a ceiling.
- [ ] Fix the introduction's existing `chap:experiments` reference to the
      evaluation chapter when revising the introduction.
