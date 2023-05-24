import os

for dataset in ["pokec_z", "pokec_n", "bail"]:
    cmd = f"python baseline_attack.py --dataset {dataset} --model gcn --attack_type random --ptb_rate 0.05 0.1 0.15 0.2 0.25"
    os.system(cmd)
