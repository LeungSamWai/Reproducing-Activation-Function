### 4.2.2 Possion Equation with a smooth solution

The corresponding code can be seen in Folder "regular",

```
python3 poly_sin_gaussian.py --dim 2 --width 50 --id 0
```

The dim represents the dimension of the problem, width is the width of the Neural network and id is the ID of the GPU.

### 4.2.3  PDE with low regularity

The corresponding code can be seen in Folder "singular",

```
python train_singular.py --dim 10 --width 100 --id 0 --method poly_sin_gaussian
```

The dim represents the dimension of the problem, width is the width of the Neural network and id is the ID of the GPU and method input the activation we want to try.



### 4.2.4 PDE with an oscillatory Solution

The code is in Folder "oscillation",

```
python train_oscillation.py --dim 2 --width 50 --freq 3 --id 0 --method poly_sin_gaussian
```

The dim represents the dimension of the problem, width is the width of the Neural network, freq is the frequency of the solution and id is the ID of the GPU and method input the activation we want to try.

