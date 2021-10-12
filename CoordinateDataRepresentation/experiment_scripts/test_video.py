'''Reproduces Supplement Sec. 7'''

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial
import skvideo.datasets
import torch
import numpy as np

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=400,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--dataset', type=str, default='bikes',
               help='Video dataset; one of (cat, bikes)', choices=['cat', 'bikes'])
p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu)')
p.add_argument('--sample_frac', type=float, default=38e-4,
               help='What fraction of video pixels to sample in each batch (default is all)')

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

if opt.dataset == 'cat':
    video_path = './data/video_512.npy'
elif opt.dataset == 'bikes':
    video_path = skvideo.datasets.bikes()

vid_dataset = dataio.Video(video_path)
coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape, sample_fraction=opt.sample_frac)
dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)
print(vid_dataset.shape)
# Define the model.
if opt.model_type in ['sine', 'sin_gaussian_params_uniform', 'sin_poly2_gaussian_params', 'gaussian_params_uniform']:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3, out_features=vid_dataset.channels,
                                 mode='mlp', hidden_features=400, num_hidden_layers=3, a=opt.a_mean, a1=opt.a_std, b=opt.b_mean, b1=opt.b_std, c=opt.c_mean, c1=opt.c_std, d=opt.d_mean, d1=opt.d_std)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', in_features=3, out_features=vid_dataset.channels, mode=opt.model_type)
else:
    raise NotImplementedError
model.cuda()
model.load_state_dict(torch.load(opt.checkpoint_path))
root_path = os.path.join(opt.logging_root, opt.experiment_name)

def write_video_summary(vid_dataset, model):
    resolution = vid_dataset.shape
    frames = [0, 50, 100, 150, 200]
    # frames = np.arange(250).tolist()
    Nslice = 10
    with torch.no_grad():
        coords = [dataio.get_mgrid((1, resolution[1], resolution[2]), dim=3)[None,...].cuda() for f in frames]
        for idx, f in enumerate(frames):
            coords[idx][..., 0] = (f / (resolution[0] - 1) - 0.5) * 2
        coords = torch.cat(coords, dim=0)

        output = torch.zeros(coords.shape)
        split = int(coords.shape[1] / Nslice)
        for i in range(Nslice):
            pred = model({'coords':coords[:, i*split:(i+1)*split, :]})['model_out']
            output[:, i*split:(i+1)*split, :] =  pred.cpu()

    pred_vid = output.view(len(frames), resolution[1], resolution[2], 3) / 2 + 0.5
    pred_vid = torch.clamp(pred_vid, 0, 1)
    gt_vid = torch.from_numpy(vid_dataset.vid[frames, :, :, :])
    psnr = 10*torch.log10(1 / torch.mean((gt_vid - pred_vid)**2))
    print('{} PSNR {}'.format(opt.checkpoint_path ,psnr))

    pred_vid = pred_vid.permute(0, 3, 1, 2)
    gt_vid = gt_vid.permute(0, 3, 1, 2)

    output_vs_gt = torch.cat((gt_vid, pred_vid), dim=-2)
    from torchvision.utils import make_grid
    make_grid(output_vs_gt, scale_each=False, normalize=True)
    # writer.add_image(prefix + 'output_vs_gt', make_grid(output_vs_gt, scale_each=False, normalize=True),
    #                  global_step=total_steps)
    # writer.add_scalar(prefix + "psnr", psnr, total_steps)
    from PIL import Image
    grid = make_grid(output_vs_gt, scale_each=False, normalize=True)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(os.path.dirname(opt.checkpoint_path)+'/'+opt.checkpoint_path.split('/')[2]+'_img.png')

def write_video_summary_onebyone(vid_dataset, model, frame):
    resolution = vid_dataset.shape
    # frames = [0, 60, 120, 200]
    frames = [frame]
    Nslice = 10
    with torch.no_grad():
        coords = [dataio.get_mgrid((1, resolution[1], resolution[2]), dim=3)[None,...].cuda() for f in frames]
        for idx, f in enumerate(frames):
            coords[idx][..., 0] = (f / (resolution[0] - 1) - 0.5) * 2
        coords = torch.cat(coords, dim=0)

        output = torch.zeros(coords.shape)
        split = int(coords.shape[1] / Nslice)
        for i in range(Nslice):
            pred = model({'coords':coords[:, i*split:(i+1)*split, :]})['model_out']
            output[:, i*split:(i+1)*split, :] =  pred.cpu()

    pred_vid = output.view(len(frames), resolution[1], resolution[2], 3) / 2 + 0.5
    pred_vid = torch.clamp(pred_vid, 0, 1)
    gt_vid = torch.from_numpy(vid_dataset.vid[frames, :, :, :])
    psnr = 10*torch.log10(1 / torch.mean((gt_vid - pred_vid)**2))
    print('{} frame {} PSNR {}'.format(opt.checkpoint_path, frame, psnr))

    pred_vid = pred_vid.permute(0, 3, 1, 2)
    gt_vid = gt_vid.permute(0, 3, 1, 2)

    output_vs_gt = torch.cat((gt_vid, pred_vid), dim=-2)
    from torchvision.utils import make_grid
    make_grid(output_vs_gt, scale_each=False, normalize=True)
    return psnr
    # writer.add_image(prefix + 'output_vs_gt', make_grid(output_vs_gt, scale_each=False, normalize=True),
    #                  global_step=total_steps)
    # writer.add_scalar(prefix + "psnr", psnr, total_steps)
    # from PIL import Image
    # grid = make_grid(output_vs_gt, scale_each=False, normalize=True)
    # ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # im = Image.fromarray(ndarr)
    # im.save(os.path.join(os.path.dirname(opt.checkpoint_path), 'img.png'))

# Define the loss
loss_fn = partial(loss_functions.image_mse, None)
summary_fn = partial(write_video_summary, vid_dataset)
f = open(os.path.dirname(opt.checkpoint_path)+'/'+opt.checkpoint_path.split('/')[2]+'_log_1.txt', "a")
print('logdir: ', os.path.dirname(opt.checkpoint_path)+'/'+opt.checkpoint_path.split('/')[2]+'_log_1.txt')
summary_fn(model)

for i in range(250):
    psnr_ = write_video_summary_onebyone(vid_dataset, model, i)
    f.write('{} \n'.format(psnr_))
f.close()

# training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
#                steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
#                model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)
