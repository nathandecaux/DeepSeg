from os.path import expanduser
import sys
home = expanduser("~")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import monai
import matplotlib.pyplot as plt
import argparse
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo TorchIO inference')
  parser.add_argument('-i', '--input', help='Input image', type=str, required=True)
  parser.add_argument('-m', '--model', help='Pytorch model', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output image', type=str, required=True)
  parser.add_argument('-f', '--fuzzy', help='Output fuzzy image', type=str, required=False)
  parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = 64)
  parser.add_argument('--patch_overlap', help='Patch overlap', type=int, required=False, default = 16)
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 2)
  parser.add_argument('-g', '--ground_truth', help='Ground truth for metric computation', type=str, required=False)
  parser.add_argument('-t', '--test_time', help='Number of inferences for test-time augmentation', type=int, required=False, default=1)
  parser.add_argument('-c', '--channels', help='Number of channels', type=int, required=False, default=16)
  parser.add_argument('--classes', help='Number of classes', type=int, required=False, default=1)
  parser.add_argument('-s', '--scales', help='Scaling factor (test-time augmentation)', type=float, required=False, default=0.05)
  parser.add_argument('-d', '--degrees', help='Rotation degrees (test-time augmentation)', type=int, required=False, default=10)
  args = parser.parse_args()
  #%%
  # unet = monai.networks.nets.UNet(
  #   dimensions=3,
  #   in_channels=1,
  #   out_channels=args.classes,
  #   channels=(args.channels, args.channels*2, args.channels*4),
  #   strides=(2, 2, 2),
  #   num_res_units=2,
  # )
  # unet.load_state_dict(torch.load(args.model))
  # class Model(pl.LightningModule):
  #     def __init__(self, net):
  #         super().__init__()
  #         self.net = net
  #     def forward(self,x):
  #         return self.net(x)
  class Unet(pl.LightningModule):
    def __init__(self, dataset, n_channels = 1, n_classes = 1, n_features = 16):
      super(Unet, self).__init__()
      #self.lr = learning_rate
      #self.net = net
      # self.criterion = criterion
      # self.optimizer_class = optimizer_class
      self.n_channels = n_channels
      self.n_classes = n_classes
      self.n_features = n_features
      self.dataset = dataset
      def double_conv(in_channels, out_channels):
        return nn.Sequential(
          nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
          nn.BatchNorm3d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
          nn.BatchNorm3d(out_channels),
          nn.ReLU(inplace=True),
        )
      self.dc1 = double_conv(self.n_channels, self.n_features)
      self.dc2 = double_conv(self.n_features, self.n_features*2)
      self.dc3 = double_conv(self.n_features*2, self.n_features*4)
      self.dc4 = double_conv(self.n_features*6, self.n_features*2)
      self.dc5 = double_conv(self.n_features*3, self.n_features)
      self.mp = nn.MaxPool3d(2)
      self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
      self.out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)
    def forward(self, x):
      x1 = self.dc1(x)
      x2 = self.mp(x1)
      x2 = self.dc2(x2)
      x3 = self.mp(x2)
      x3 = self.dc3(x3)
      x4 = self.up(x3)
      x4 = torch.cat([x4,x2], dim=1)
      x4 = self.dc4(x4)
      x5 = self.up(x4)
      x5 = torch.cat([x5,x1], dim=1)
      x5 = self.dc5(x5)
      return self.out(x5)
  net = Unet(dataset='dhcp')
  net.eval()
  #%%
  subject = tio.Subject(
        image=tio.ScalarImage(args.input),
    )
  #%%
  #normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
  normalization = tio.ZNormalization()
  onehot = tio.OneHot()
  spatial = tio.RandomAffine(scales=args.scales,degrees=args.degrees,translation=5,image_interpolation='bspline',p=1)
  flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)
  print('Inference')
  patch_overlap = args.patch_overlap
  patch_size = args.patch_size
  batch_size = args.batch_size
  output_tensors = []
  for i in range(args.test_time):
    if i == 0:
      augment = normalization
    else:
      augment = tio.Compose((normalization,flip, spatial))
    subj = augment(subject)
    #print(subj.get_composed_history())
    grid_sampler = tio.inference.GridSampler(
      subj,
      patch_size,
      patch_overlap,
      )
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
    with torch.no_grad():
      for patches_batch in patch_loader:
        input_tensor = patches_batch['image'][tio.DATA]
        locations = patches_batch[tio.LOCATION]
        outputs = net(input_tensor)
        aggregator.add_batch(outputs, locations)
    output_tensor = aggregator.get_output_tensor()
    #output_tensor = torch.sigmoid(output_tensor)
    tmp = tio.ScalarImage(tensor=output_tensor, affine=subj.image.affine)
    subj.add_image(tmp, 'label')
    back = subj.apply_inverse_transform(image_interpolation='linear')
    output_tensors.append(torch.unsqueeze(back.label.data,0))
  output_tensor = torch.squeeze(torch.stack(output_tensors, dim=0).mean(dim=0))
  print(output_tensor.shape)
  print('Saving images')
  output_seg = tio.ScalarImage(tensor=output_tensor.unsqueeze(0), affine=subject['image'].affine)
  # print(type(output_seg))
  # print(output_seg)
  output_seg.save(args.output)
  if args.fuzzy is not None:
    output_seg = tio.ScalarImage(tensor=output_tensor, affine=subject['image'].affine)
    output_seg.save(args.fuzzy)
  if args.ground_truth is not None:
    gt_image= onehot(tio.ScalarImage(args.ground_truth))
    pred_image = onehot(output_seg)
    print(gt_image.data.shape)
    print(pred_image.data.shape)
    psnr_val = monai.metrics.PSNRMetric(torch.unsqueeze(pred_image.data,0), torch.unsqueeze(gt_image.data,0))
    print("PSNR :")
    print(psnr_val)