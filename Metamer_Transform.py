import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import math
import time

import net
from function import adaptive_instance_normalization, coral

# Example code:
# python Metamer_Transform.py --image 4751.png --output output_Stimuli

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--image', type=str, help='File path to the content image')
parser.add_argument('--image_dir', type=str, help='Directory path to a batch of content images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder-content-similar.pth') # Notice that this decoder is different than the classically used "decoder.pth"!

# Additional options
parser.add_argument('--image_size', type=int, default=512, help='New (minimum) size for the content image, keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true', help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.png', help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output', help='Directory to save the output image(s)')
parser.add_argument('--scale',type=str,default='0.4',help='Rate of growth of the Log-Polar Receptive Fields')
parser.add_argument('--verbose',type=int,default=0,help='Print several hyper-parameters as we run the rendering scheme. Default should be 0, should only be set to 1 for debugging.')
parser.add_argument('--reference',type=int,default=0,help='Compute the reference image')

args = parser.parse_args()

# List of potentially different rate of growth of receptive fields
# assuming a center fixation.
scale_in = ['0.25','0.3','0.4','0.5','0.6','0.7']
scale_out = [377,301,187,126,103,91]

Pooling_Region_Map = dict(zip(scale_in,scale_out))

verb = args.verbose

resize_output = transforms.Compose([transforms.Resize((256,256))])
to_pil_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


#function that loads receptive fields:
def load_receptive_fields():
    d = 1.281 # a value that was fitted via psychophysical experiments assuming 26 deg of visual angle maps to 512 pixels on a screen.
    mask_total = torch.zeros(Pooling_Region_Map[args.scale],64,64)
    alpha_matrix = torch.zeros(Pooling_Region_Map[args.scale])
    for i in range(Pooling_Region_Map[args.scale]):
        i_str = str(i)
        #mask_str = './Receptive_Fields/MetaWindows_clean_s0.4/' + i_str + '.png'
        mask_str = './Receptive_Fields/MetaWindows_clean_s' + args.scale + '/' + i_str + '.png'
        mask_temp = mask_tf(Image.open(str(mask_str)))
        mask_total[i,:,:] = mask_temp
        mask_regular = mask_regular_tf(Image.open(str(mask_str)))
        mask_size = torch.sum(torch.sum(mask_regular>0.5))
        recep_size = np.sqrt(mask_size/3.14)*26.0/512.0
        if i == 0:
            alpha_matrix[i] = 0
        else:
            alpha_matrix[i] = -1 + 2.0 / (1.0+math.exp(-recep_size*d))
        if verb == 1:
            print(alpha_matrix[i])
    return mask_total, alpha_matrix


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def mask_transform():
    transform = transforms.Compose([transforms.Resize(64),transforms.Grayscale(1),transforms.ToTensor()])
    return transform

def mask_transform_regular():
    transform = transforms.Compose([transforms.Resize(512),transforms.Grayscale(1),transforms.ToTensor()])
    return transform

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def foveated_style_transfer(vgg,decoder, content, mask_total, alpha_matrix, reference):
    noise = torch.randn(3,512,512)
    noise = coral(noise,content)
    # Move images to GPU:
    style = content.to(device).unsqueeze(0) # Remember we are doing something like "Auto-Style Transfer"
    content = content.to(device).unsqueeze(0)
    noise = noise.to(device).unsqueeze(0)
    # Create Empty Foveated Feature Vector to which we will allocate the latent crowded feature vectors:
    foveated_f = torch.zeros(1,512,64,64).to(device)
    # Create Content Feature Vector (post VGG):
    content_f = vgg(content)
    # Create Style Feature Vector (post VGG):
    style_f = vgg(style)
    # Create Noise Feature Vector (post VGG):
    noise_f = vgg(noise)
    # assume alpha_i = 0.5
    if reference == 1:
        return decoder(content_f)
    else:
        for i in range(Pooling_Region_Map[args.scale]): # Loop over all the receptive fields (pooling regions)
            alpha_i = alpha_matrix[i]
            mask = mask_total[i,:,:]
            mask = mask.unsqueeze(0)
            mask = tile(mask,0,512)
            mask_binary = mask>0.001
            if verb == 1:
                print(np.shape(content_f))
                print(np.shape(mask_binary[0,:,:]))
            content_f_mask = content_f[:,:,mask_binary[0,:,:]] # 0 was 0th prefix before
            style_f_mask = style_f[:,:,mask_binary[0,:,:]]
            noise_f_mask = noise_f[:,:,mask_binary[0,:,:]]
            #
            if verb == 1:
                print(np.shape(noise_f_mask.unsqueeze(3)))
                print(np.shape(style_f_mask.unsqueeze(3)))
            content_f_mask = content_f_mask.unsqueeze(3)
            noise_f_mask = noise_f_mask.unsqueeze(3)
            style_f_mask = style_f_mask.unsqueeze(3)
            if verb == 1:
                print(np.shape(content_f_mask))
            # Perform the Crowding Operation and Localized Auto Style-Transfer
            texture_f_mask = adaptive_instance_normalization(noise_f_mask,style_f_mask)
            alpha_mixture = (1-alpha_i)*content_f_mask + (alpha_i)*texture_f_mask
            if verb == 1:
                print(np.shape(alpha_mixture))
            foveated_f[:,:,mask_binary[0,:,:]] = alpha_mixture.squeeze(3)
        # Run the now foveated image in the latent space through the decoder to render the metamer
        return decoder(foveated_f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --image or --imageDir should be given.
assert (args.image or args.image_dir)
if args.image:
    image_paths = [Path(args.image)]
else:
    image_dir = Path(args.image_dir)
    image_paths = [f for f in image_dir.glob('*')]

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

reference = args.reference

vgg.to(device)
decoder.to(device)

image_tf = test_transform(args.image_size, args.crop)

# Define Masked Transforms (for localized style transfer that is the basis 
# of the foveated style transform operation for each pooling region
mask_tf = mask_transform()
mask_regular_tf = mask_transform_regular()

if verb == 1:
    print(image_paths)
    print(image_paths[0])
    print(image_paths[0].stem)

print(len(image_paths))

with torch.no_grad():
    mask_total, alpha_matrix = load_receptive_fields()
    for z in range(len(image_paths)):
        image_path = image_paths[z]
        image = image_tf(Image.open(str(image_path)).convert('RGB'))
        start_time = time.time()
        output = foveated_style_transfer(vgg,decoder,image,mask_total,alpha_matrix,reference)
        output = output.cpu()
        output2 = to_pil_image(torch.clamp(output.squeeze(0),0,1))
        output = torch.clamp(to_tensor(resize_output(output2)),0,1)
        end_time = time.time()
        # Move from GPU to CPU
        #output = output.cpu()
        # Move Output
        if reference == 0:
            output_name = output_dir / '{:s}_s{:s}{:s}'.format(image_path.stem, args.scale, args.save_ext)
        elif reference == 1:
            output_name = output_dir / '{:s}_Reference{:s}'.format(image_path.stem, args.save_ext)
        # Save Image
        save_image(output, str(output_name))
        # Display Compute Time
        print(['Total Rendering time: ' + str(end_time-start_time) + ' seconds'])
        if z % 50 == 1:
             time.sleep(10)
