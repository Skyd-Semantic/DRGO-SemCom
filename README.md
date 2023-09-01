# Distortion Rate Resilience: A joint communication and computation framework for Goal-Oriented Semantic Communication

# Citation
Anyone who want to use this code, please cite the following references
```
@article{DRGO-SemCom,
  Title = {Distortion Rate Resilience: A joint communication and computation framework for Goal-Oriented Semantic Communication},
  Author = {Minh-Duong Nguyen, Quang-Vinh Do, Zhaohui Yang, Viet Pham, Won-Joo Hwang},
  journal={IEEE Transactions on SomeThing Special},
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

