# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial
import torch
import numpy

torch.manual_seed(1)
torch.cuda.manual_seed(1)
numpy.random.seed(1)

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default='audio',
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

p.add_argument('--wav_path', type=str, default='../data/gt_bach.wav', help='root for logging')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=2001,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=10,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--a_mean', type=float)
p.add_argument('--b_mean', type=float)
p.add_argument('--c_mean', type=float)
p.add_argument('--d_mean', type=float)
p.add_argument('--a_std', type=float)
p.add_argument('--b_std', type=float)
p.add_argument('--c_std', type=float)
p.add_argument('--d_std', type=float)

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

audio_dataset = dataio.AudioFile(filename=opt.wav_path)
coord_dataset = dataio.ImplicitAudioWrapper(audio_dataset)

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type in ['sine', 'sin_gaussian_params', 'sin_params', 'sin_gaussian_params_uniform', 'sin_poly2_gaussian_params']:
    model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', in_features=1, a=opt.a_mean, a1=opt.a_std, b=opt.b_mean, b1=opt.b_std, c=opt.c_mean, c1=opt.c_std, d=opt.d_mean, d1=opt.d_std)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode=opt.model_type, fn_samples=len(audio_dataset.data), in_features=1)
else:
    raise NotImplementedError
model.cuda()
model.cuda()
model.load_state_dict(torch.load(opt.checkpoint_path))

def write_audio_summary(logging_root_path, model, model_input, gt, model_output, writer, total_steps, prefix='train'):
    gt_func = torch.squeeze(gt['func'])
    gt_rate = torch.squeeze(gt['rate']).detach().cpu().numpy()
    gt_scale = torch.squeeze(gt['scale']).detach().cpu().numpy()
    pred_func = torch.squeeze(model_output['model_out'])
    coords = torch.squeeze(model_output['model_in'].clone()).detach().cpu().numpy()

    fig, axes = plt.subplots(3,1)

    strt_plot, fin_plot = int(0.05*len(coords)), int(0.95*len(coords))
    coords = coords[strt_plot:fin_plot]
    gt_func_plot = gt_func.detach().cpu().numpy()[strt_plot:fin_plot]
    pred_func_plot = pred_func.detach().cpu().numpy()[strt_plot:fin_plot]

    axes[1].plot(coords, pred_func_plot)
    axes[0].plot(coords, gt_func_plot)
    axes[2].plot(coords, gt_func_plot - pred_func_plot)
    # axes[2].set_ylim((-0.06, 0.06))
    axes[2].set_ylim((-0.5, 0.5)) # counting
    axes[0].get_xaxis().set_visible(False)
    axes[1].axes.get_xaxis().set_visible(False)
    axes[2].axes.get_xaxis().set_visible(False)

    writer.add_figure(prefix + 'gt_vs_pred', fig, global_step=total_steps)


# root_path = os.path.join(opt.logging_root, opt.experiment_name)
# utils.cond_mkdir(root_path)


# Define the loss
# loss_fn = loss_functions.function_mse
summary_fn = partial(utils.write_audio_summary, root_path)
summary_fn(model)


# training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
#                steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
#                model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)