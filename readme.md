## Asynchronous RL Algorithms for Super Mario Bros
![Alt Text](media/mario.gif)
![Alt Text](media/1_4.gif)
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