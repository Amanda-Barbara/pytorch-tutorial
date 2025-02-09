import torch
import torchvision
def fuse_conv_and_bn(conv, bn):
	#
	# init
	fusedconv = torch.nn.Conv2d(
		conv.in_channels,
		conv.out_channels,
		kernel_size=conv.kernel_size,
		stride=conv.stride,
		padding=conv.padding,
		bias=True
	)
	#
	# prepare filters
	print("conv.out_channels:",conv.out_channels)
	w_conv = conv.weight.clone().view(conv.out_channels, -1)
	print("w_conv:",w_conv.shape)
	w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
	print("w_bn:", w_bn.shape)
	fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
	#
	# prepare spatial bias
	if conv.bias is not None:
		b_conv = conv.bias
	else:
		b_conv = torch.zeros( conv.weight.size(0) )
	b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
	fusedconv.bias.copy_( b_conv + b_bn )
	#
	# we're done
	return fusedconv

#以下代码片段在ResNet18的前两层测试上述函数:

torch.set_grad_enabled(False)
x = torch.randn(16, 3, 256, 256)
rn18 = torchvision.models.resnet18(pretrained=True)
rn18.eval()
net = torch.nn.Sequential(
	rn18.conv1,
	rn18.bn1
)
y1 = net.forward(x)
fusedconv = fuse_conv_and_bn(net[0], net[1])
y2 = fusedconv.forward(x)
d = (y1 - y2).norm().div(y1.norm()).item()
print("error: %.8f" % d)

