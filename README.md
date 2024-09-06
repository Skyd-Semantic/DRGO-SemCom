# Distortion Resilience for Goal-Oriented Semantic Communication, Transactions on Mobile Computing, Sep. 2024
Original paper: https://arxiv.org/abs/2309.14587

Part of this work is presented as invited speaker at IEEE ICMLCN 2024

# Citation
Anyone who want to use this code, please cite the following references
```
@article{DRGO-SemCom,
  Title = {Distortion Resilience for Goal-Oriented Semantic Communication},
  Author = {Minh-Duong Nguyen, Quang-Vinh Do, Zhaohui Yang, Won-Joo Hwang, Quoc-Viet Pham},
  journal={IEEE Transactions on Mobile Computing},
  month={Sep.},
  year={2023}
}
```

# Clone
```
git clone https://github.com/Skyd-Semantic/DRGO-SemCom.git
```

# Training
```commandline
python main.py --memory-size 100000 --initial-steps 50000 --batch-size 32 --max-episode 4000 --max-step 200 --max-episode-eval 5 --max-step-eval 50 --semantic-mode learn --pen-coeff 0  --noise 0.01 --lamda 0.001 --poweru-max 10 --plot-interval 800000 --user-num 20 --drl-algo ddpg-ei --ai-network resnet9 --lr-actor 1e-3 --lr-critic 3e-3

