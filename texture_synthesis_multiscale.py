import torch
import my_utils
import torchvision
import torch.nn as nn

# Choose image size and image file 
imsize0 = 512 # use 256 for raddish
N_optim_iter = 250
save_every = 50
N_scales = 1

image_folder = 'Images/'
image_name = 'bones.jpg' #raddish.jpg


data_file = image_folder + image_name
y0 = 2*my_utils.image_loader(data_file).cuda()-1

# Change this to directory of previously downloaded model, or re download model
model_folder = '../pytorch-image-synthesis-labs/Models/'

# Exercise 2: synthesize bones using different feature maps
# out_keys determine which intermediate tensors in VGG-19 are used in the gram matrix loss
# see vgg.py and my_utils.py
out_keys = ['r11','r21','r31','r41','r51']
loss_alpha = 1e-2 # NOTE: you might have to modify this value if the loss doesn't change

# out_keys = ['r11','r21','r31'] 
# loss_alpha = 1e-2 # Also uncomment this when trying this setting

#out_keys = ['r51'] 
#loss_alpha = 1e-2 # Also uncomment this when trying this setting

# Load image and VGG-19 network
out_folder = 'my_synthesis/'
import os
if os.path.exists(out_folder)==False:
    os.mkdir(out_folder)
aspect_ratio = y0.size(3)/y0.size(2)
imsize1 = int(imsize0*aspect_ratio)
y0 = nn.AdaptiveAvgPool2d((imsize0,imsize1))(y0)


# Load network
L2_loss = nn.MSELoss()
gram_matrix = my_utils.GramMatrix()
vgg_net = my_utils.get_vgg_net(model_folder,out_keys)
loss_lambda = loss_alpha*torch.tensor([1,1,1,1,1.0]).cuda()


im_folder = out_folder + image_name + '_'+'_'.join(out_keys) +'_' 'N_scales=%d'%(N_scales) + '/'

if os.path.exists(im_folder)==False:
    os.mkdir(im_folder)
torchvision.utils.save_image(y0,im_folder + 'input_image.jpg',normalize=True)


# Exercise 1: Print size of each tensor in output of vgg_net(y)
# Note that the outputs are decided by out_keys
#y = y0.detach()
#y_activ = [out.detach() for out in vgg_net(y)]
#y_feats = [gram_matrix(out) for out in y_activ]

#input('Comment after completion of exercise 1')


# Exercise 3: part 1: replaces imsizes0, imsizes1 with resolutions imsize0/2^(N_scales-1)... imsize0/2^(0)
imsizes0 = [imsize0]*N_scales
imsizes1 = [imsize1]*N_scales

for s,cur_imsize0 in enumerate(imsizes0):
    
    cur_imsize1 = imsizes1[s]

    # Replace this scale layer with nn.AdaptiveAvgPool2d and the correct resolution
    scale_layer = lambda x: x
    
    y = scale_layer(y0.detach())
    y_feats = [gram_matrix(out.detach()) for out in vgg_net(y)]
    
    if s==0:
        x = my_utils.get_input_noise(cur_imsize0,cur_imsize1)

    else:
        alpha = .9
        x = scale_layer(x.detach().cpu())
        x = my_utils.smooth_image(x)
    
    optimizer = torch.optim.LBFGS([x.requires_grad_()],max_iter=1)

    for i in range(N_optim_iter):
        def closure():
    
            optimizer.zero_grad()
            x_feats = vgg_net(x.cuda().clamp(-1,1))
            
            loss = 0
            for ll,x_feat,y_feat in zip(loss_lambda,x_feats,y_feats):
                loss += ll*L2_loss(gram_matrix(x_feat),y_feat)
    
            loss.backward()
            print('iter=%.d, loss=%e'%(i,loss.item()))
            return loss
        optimizer.step(closure)
        if i%save_every==0:
            torchvision.utils.save_image(x.cpu().detach().clamp(-1,1),im_folder + 'scale_%d_iteration_%d.jpg'%(s,i),normalize=True)
        
    torchvision.utils.save_image(x.cpu().detach().clamp(-1,1),im_folder + 'final_synthesis.jpg',normalize=True)
    

