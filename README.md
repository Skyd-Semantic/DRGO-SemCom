# Distortion Rate Resilience: A joint communication and computation framework for Goal-Oriented Semantic Communication

# Citation
Anyone who want to use this code, please cite the following references
```
@article{DRGO-SemCom,
  Title = {Distortion Rate Resilience: A joint communication and computation framework for Goal-Oriented Semantic Communication},
  Author = {Minh-Duong Nguyen, Quang-Vinh Do, Zhaohui Yang, Won-Joo Hwang},
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
 python main.py --initial-steps 50000 --max-episode 400 --max-step 200 --max-episode-eval 5 --max-step-eval 50 --semantic-mode learn --pen-coeff 0.0 --noise 0.01 --lamda 0.001 --poweru-max 10 --plot-interval 80000 --user-num 10```

