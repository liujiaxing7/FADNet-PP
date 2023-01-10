import argparse
import os
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn


from networks.FADNet import FADNet

parser = argparse.ArgumentParser(description='FADNet')
# parser.add_argument('--crop_height', type=int, required=True, help="crop height")
# parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--sceneflow', type=int, default=0, help='sceneflow dataset? Default=False')
parser.add_argument('--kitti2012', type=int, default=0, help='kitti 2012? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--middlebury', type=int, default=0, help='Middlebury? Default=False')
parser.add_argument('--eth3d', type=int, default=0, help='ETH3D? Default=False')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='data path')
parser.add_argument('--list', default='lists/middeval_test.list',
                    help='list of stereo images')
parser.add_argument('--loadmodel', default="finetune_0_1.tar",
                    help='loading model')
parser.add_argument('--savepath', default='results/',
                    help='path to save the results.')
parser.add_argument('--model', default='fadnet',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
opt = parser.parse_args()
print(opt)

torch.backends.cudnn.benchmark = True

opt.cuda = not opt.no_cuda and torch.cuda.is_available()

if not os.path.exists(opt.savepath):
    os.makedirs(opt.savepath)

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)


model = FADNet(maxdisp=opt.maxdisp)

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if opt.loadmodel is not None:
    state_dict = torch.load(opt.loadmodel)
    model.load_state_dict(state_dict['state_dict'],strict=False)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

model.eval()

onnx_input_L = torch.rand(1, 3, 384, 1280)
onnx_input_R = torch.rand(1, 3, 384, 1280)
onnx_input_L = onnx_input_L.to("cuda:0")
onnx_input_R = onnx_input_R.to("cuda:0")
# onnx_input = torch.cat((onnx_input_L, onnx_input_R), 1)
torch.onnx.export(model.module,
                  (onnx_input_L,onnx_input_R),
                  "{}.onnx".format(opt.model),
                  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=12,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['left', 'right'],  # the model's input names
                  output_names=['output'])