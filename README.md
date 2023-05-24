# FATE
Implementations of NeurIPS 2023 submission "FATE: Fairness Attacks on Graph Learning"

## Requirements
Main dependency: python 3.8.13, pytorch 1.12.1, deeprobust 0.2.4

## Run
To generate the poisoned graph, go to `src/` folder and run the following code:
```
python fate_attack.py --dateset pokec_n --fairness statistical_parity --ptb_mode flip --ptb_rate 0.05 --attack_steps 3 --attack_seed 25 
```

To train the victim model, go to `src/` folder and run the following code:
```
python train.py --dateset pokec_n --fairness statistical_parity --ptb_mode flip --ptb_rate 0.05 --attack_steps 3 --attack_seed 25 --attack_method fate --victim_model gcn --hidden_dimension 128 --num_epochs 400 
```

To test under different settings, please feel free to refer to the detailed parameter settings listed in Appendix C.