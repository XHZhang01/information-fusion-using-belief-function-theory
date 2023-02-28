# BFT_IV2023


## Run the simulation

```
python3 main.py
```
## Simulation data

/data/all_veh_data.yaml

## Code function

bel_ass.py: assigns belief mass with Gaussian distribution based on distance between model-given value and measured value to each possibility

conf_eval.py: evaluates conflict among opinions at the same time instant before fusion

conf_hand.py: moves part of belief mass from each possibility to uncertainty based on degree of conflict

dcr.py: fuses a number of opinions with Dempster Combination Rule

main.py: initialzes the framework and runs it with the simulation data

models.py: offers value at each time instant based on usr-defined models

opi_gen.py: generates opinion based measured data given by different sensors and model-given data

set_unc.py: sets uncertainty of opinion at current time instant based on perturbation of belief distribution within a given window size

test_wbf.py: tests samples with Weighted Belief Fusion operator

wbf.py: updates belief at current time instant with previous belief and current fused opinion
