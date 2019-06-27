import argparse
import os
import torch
import vgg
# from net.quantization import apply_weight_sharing, check_module_structure
from net.quantization import *
import util
from torchsummary import summary

parser = argparse.ArgumentParser(description='This program quantizes weight by using weight sharing')
parser.add_argument('--model', type=str, required=False, help='path to saved pruned model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--output', default='saves/model_after_weight_sharing.ptmodel', type=str,
                    help='path to model output')
parser.add_argument('--state_dict', default='saves/model_best_state_dict.pth', type=str,
                    help='path to saved_dict')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()

# model_orig = vgg.vgg19()
# model_orig.features = torch.nn.DataParallel(model_orig.features)
# model_orig = model_orig.cuda()
# model_orig.load_state_dict(torch.load(args.state_dict))
# model_orig.features = model_orig.features.module

model = vgg.vgg19().cuda()
model.features = torch.nn.DataParallel(model.features)
model = model.cuda()
model.load_state_dict(torch.load(args.state_dict))
model.features = model.features.module


model_flat = flatten_model(model)
sd_formatted = get_formatted_state_dict(model)
model_flat.load_state_dict(sd_formatted)


# model = model.module
# Define the model
# model = torch.load(args.model)
print('accuracy before weight sharing')
util.test(model, use_cuda)
util.test(model_flat, use_cuda)
# # Weight sharing
# apply_weight_sharing(model, bits = 1)
# print('accuacy after weight sharing')
# util.test(model, use_cuda)

# # Save the new model
# os.makedirs('saves', exist_ok=True)
# torch.save(model, args.output)
