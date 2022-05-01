## Asynchronous RL Algorithms for Super Mario Bros
![Alt Text](images/mario.gif)

### Setup
Ensure all dependencies in requirements.txt are installed. 
### To run a demo
```
python trainer.py a3c --processes 0 --render True --model_file experiments/exp1/a3c_best.pt
```

### To train
```
python trainer.py <algorithm of choice>
```

Example: 

```
python trainer.py a3c
```