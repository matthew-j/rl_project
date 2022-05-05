## Asynchronous RL Algorithms for Super Mario Bros
![Alt Text](media/mario.gif)
![Alt Text](media/1_4.gif)
### Setup
Ensure all dependencies in requirements.txt are installed. 
### To run a demo
```
python trainer.py a3c -p 0 -r --model_file experiments/exp1/a3c_best.pt
```

### To train
```
python trainer.py <algorithm of choice>
```

Algorithm choices: "a3c", "qlearn", and "nqlearn"

Example: 

```
python trainer.py a3c
```

```
python trainer.py -r -p 0 a3c --actions easy_movement --model_file experiments/exp2/a3c_best_right_only_no_noop.pt
```

Gym Super Mario Bros. environment: [](https://github.com/Kautenja/gym-super-mario-bros)
