# Reproducing Activation Function for Deep Learning
We apply our reproducing activation functions to the image and audio regression task in [SIREN](https://github.com/vsitzmann/siren).

## Prepare the environment
Please follow [SIREN](https://github.com/vsitzmann/siren) to set up the environment for the code.

## Coordinate-based Data Representation

The implementation of Sine-Poly-Gaussian activation function is in the following python file,
```
./modules.py
```

To run the code for image regression, use the following command. We have set up the hyper-parameter in the file,
```
python run_img.py
```

To run the code for audio regression, use the following command. We have set up the hyper-parameter in the file,
```
python run_audio.py
```
