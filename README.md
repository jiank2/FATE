# FATE
Implementations of ICLR 2024 submission "Deceptive Fairness Attacks on Graphs via Meta Learning"

## Requirements
Main dependency: python 3.8, pytorch 1.12.1, deeprobust 0.2.4

## Run
To generate the poisoned graph with FATE, go to `src/` folder and run the following code:
```
python fate_attack.py --dateset pokec_n --fairness statistical_parity --ptb_mode flip --ptb_rate 0.05 --attack_steps 3 --attack_seed 25 
```

To train the victim model, go to `src/` folder and run the following code:
```
python train.py --dateset pokec_n --fairness statistical_parity --ptb_mode flip --ptb_rate 0.05 --attack_steps 3 --attack_seed 25 --attack_method fate --victim_model gcn --hidden_dimension 128 --num_epochs 400 
```

To test under different settings, please feel free to refer to the detailed parameter settings listed in Appendix D.


# License
Deceptive Fairness Attacks on Graphs via Meta Learning is licensed under CC BY-NC-ND 4.0. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/