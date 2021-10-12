# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions
from torch.utils.data import DataLoader
import configargparse
from functools import partial
import torch
import numpy
import modules as modules
torch.manual_seed(1)
torch.cuda.manual_seed(1)
numpy.random.seed(1)

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--fig', type=str, help='')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=2000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=9999,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=10,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

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

if opt.fig == 'camera':
    img_dataset = dataio.Camera()
elif opt.fig == 'astronaut':
    img_dataset = dataio.Astronaut()
elif opt.fig == 'cat':
    img_dataset = dataio.Cat()
elif opt.fig == 'coin':
    img_dataset = dataio.coin()

coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=256, compute_diff='all')
image_resolution = (256, 256)

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type in ['sine', 'sin_poly2_params', 'sin_gaussian_params_uniform', 'sin_params', 'sin_poly2_gaussian_params']:
    model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', sidelength=image_resolution, a=opt.a_mean, a1=opt.a_std, b=opt.b_mean, b1=opt.b_std, c=opt.c_mean, c1=opt.c_std, d=opt.d_mean, d1=opt.d_std)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode=opt.model_type, sidelength=image_resolution)
else:
    raise NotImplementedError

model.cuda()
root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
loss_fn = partial(loss_functions.image_mse, None)
summary_fn = partial(utils.write_image_summary, image_resolution)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)
