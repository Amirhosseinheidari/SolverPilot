# M26 Selector Postmortem

## What failed

The ridge model predicts relative log runtime (`log(IPM/DS)`) reasonably as a smooth structural target, but the production objective is mean absolute runtime including feature cost. These objectives are not equivalent.

On validation, DS wins many small/easy instances by small amounts. IPM wins fewer but can win expensive instances by hundreds of milliseconds. A model that treats relative errors symmetrically can therefore look plausible while generating unacceptable absolute regret.

`seymour1` is the clearest failure: the model predicts DS, but IPM is roughly 2.3x faster; this single decision dominates validation regret.

## What did NOT fail

- M25's selection opportunity is not invalidated.
- corpus integrity and objective agreement are unchanged.
- the split is not leaking groups.
- feature cost accounting is present.
- the final test was not consumed.

## What a future selector must change

Without touching the sealed test, the next model family should optimize an objective aligned to deployment cost. Candidate directions include:

1. predict absolute regret of switching away from the default/SBS rather than winner labels;
2. use a selective switcher that defaults to the safe single backend and switches only when predicted savings exceed a margin;
3. weight training errors by the absolute runtime consequence of a wrong decision;
4. use cross-validation only inside the existing train+validation pool and keep the 16-instance M26 test sealed for one final gate.

Any future milestone must pre-register that model family before opening the test.
