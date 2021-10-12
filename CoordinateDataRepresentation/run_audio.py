import os

a_mean = 2 # mean of initialization of coefficient of sin
a_std = 0.1 # std of initialization of coefficient of sin
b_mean = 30 # mean of initialization of coefficient of scaling within sin
b_std = 10 # std of initialization of coefficient of scaling within sin
c_mean = 1 # mean of initialization of coefficient of guassian
c_std = 0.1 # std of initialization of coefficient of guassian
d_mean = 0.01 # uniform initialization of coefficient of scaling of guassian
d_std = 0.05 # uniform initialization of coefficient of scaling of guassian

command = 'python experiment_scripts/train_audio.py '\
          '--model_type=sin_gaussian_params_uniform --wav_path=data/gt_counting.wav ' \
          '--a_mean {} --a_std {} --b_mean {} --b_std {} --c_mean {} --c_std {} --d_mean {} --d_std {} '\
          ' --experiment_name radio_counting/sin_gau_uniform_a-{}_{}_b-{}_{}_c-{}_{}_d-{}_{}'.format(a_mean, a_std, b_mean, b_std, c_mean, c_std, d_mean, d_std,
                                                                                                     a_mean, a_std, b_mean, b_std, c_mean, c_std, d_mean, d_std)

print(command)
os.system(command)