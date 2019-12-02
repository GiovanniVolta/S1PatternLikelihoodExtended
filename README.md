# S1PatternLikelihoodExtended
S1 pattern likelihood extended for high energy analysis

### Motivation
The main motivation for developing the extended version of S1 Pattern Likelihood cut is that at higher energies it can contribute to the multiple scatters rejections, where it expected that the Compton scattering dominates. The idea behind the multiple scatter rejection is that for these events the light pattern on the PMTs should be more distorted, due to the almost simultaneous multiple interactions, compare to single scatter events. The cut is also sensitive to other anomalous events: the dominant type of this event is known as the gamma-X event.

### Cut defination
The method used is based on Log Likelihood calculation. A model of PMT pattern is obtained from simulation, hence, the Poisson Log Likelihood Ratio (λ) is computed for each event to be tested with respect to the model.
The cut has been developed through λ obtain only with the bottom PMT array and it has been optimised up to the RoI of neutrinoless double beta decay analysis. The data quality criteria has been defined in two parameter spaces using background data: (z; S1 pattern fit bottom hax) and (S1; S1 pattern fit bottom hax).
Finally, the acceptance and rejection power has been investigated thought the "photoelectric peak" events of the same background sample.

