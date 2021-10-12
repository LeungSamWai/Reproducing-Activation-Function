import os

a_mean = 0 # mean of initialization of coefficient of x
a_std = 0.1 # std of initialization of coefficient of x
b_mean = 1 # mean of initialization of coefficient of x^2
b_std = 0.1 # std of initialization of coefficient of x^2
c_mean = 2 # mean of initialization of coefficient of sin
c_std = 0.1 # std of initialization of coefficient of sin
d_mean = 30 # mean of initialization of coefficient of scaling within sin
d_std = 0.001 # std of initialization of coefficient of scaling within sin

command = 'python experiment_scripts/train_img.py '\
          '--model_type=sin_poly2_gaussian_params --fig astronaut ' \
          '--a_mean {} --a_std {} --b_mean {} --b_std {} --c_mean {} --c_std {} --d_mean {} --d_std {} '\
          ' --experiment_name img_astronaut/sin_poly2_gaussian_params_a-{}_{}_b-{}_{}_c-{}_{}_d-{}_{}'.format(a_mean, a_std, b_mean, b_std, c_mean, c_std, d_mean, d_std,
                                                                                                              a_mean, a_std, b_mean, b_std, c_mean, c_std, d_mean, d_std)
print(command)
os.system(command)