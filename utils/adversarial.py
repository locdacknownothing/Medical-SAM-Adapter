import datetime
import warnings
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch import autograd
from torch.autograd import Variable
from tqdm import tqdm
from models.discriminator import Discriminator

from utils import args, device
from utils.visualization import tensor_to_img_array, view, export

def para_image(w, h=None, img = None, mode = 'multi', seg = None, sd=None, batch=None,
          fft = False, channels=None, init = None):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    param_f = fft_image if fft else pixel_image
    if init is not None:
        param_f = init_image
        params, maps_f = param_f(init)
    else:
        params, maps_f = param_f(shape, sd=sd)
    if mode == 'multi':
        output = to_valid_out(maps_f,img,seg)
    elif mode == 'seg':
        output = gene_out(maps_f,img)
    elif mode == 'raw':
        output = raw_out(maps_f,img)
    return params, output

def to_valid_out(maps_f,img,seg): #multi-rater
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        maps = torch.nn.Softmax(dim = 1)(maps)
        final_seg = torch.multiply(seg,maps).sum(dim = 1, keepdim = True)
        return torch.cat((img,final_seg),1)
        # return torch.cat((img,maps),1)
    return inner

def gene_out(maps_f,img): #pure seg
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        # maps = torch.nn.Sigmoid()(maps)
        return torch.cat((img,maps),1)
        # return torch.cat((img,maps),1)
    return inner

def raw_out(maps_f,img): #raw
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        # maps = torch.nn.Sigmoid()(maps)
        return maps
        # return torch.cat((img,maps),1)
    return inner

class CompositeActivation(torch.nn.Module):

    def forward(self, x):
        x = torch.atan(x)
        return torch.cat([x/0.67, (x*x)/0.6], 1)
        # return x

def cppn(args, size, img = None, seg = None, batch=None, num_output_channels=1, num_hidden_channels=128, num_layers=8,
         activation_fn=CompositeActivation, normalize=False, device = "cuda:0"):

    r = 3 ** 0.5

    coord_range = torch.linspace(-r, r, size)
    x = coord_range.view(-1, 1).repeat(1, coord_range.size(0))
    y = coord_range.view(1, -1).repeat(coord_range.size(0), 1)

    input_tensor = torch.stack([x, y], dim=0).unsqueeze(0).repeat(batch,1,1,1).to(device)

    layers = []
    kernel_size = 1
    for i in range(num_layers):
        out_c = num_hidden_channels
        in_c = out_c * 2 # * 2 for composite activation
        if i == 0:
            in_c = 2
        if i == num_layers - 1:
            out_c = num_output_channels
        layers.append(('conv{}'.format(i), torch.nn.Conv2d(in_c, out_c, kernel_size)))
        if normalize:
            layers.append(('norm{}'.format(i), torch.nn.InstanceNorm2d(out_c)))
        if i < num_layers - 1:
            layers.append(('actv{}'.format(i), activation_fn()))
        else:
            layers.append(('output', torch.nn.Sigmoid()))

    # Initialize model
    net = torch.nn.Sequential(OrderedDict(layers)).to(device)
    # Initialize weights
    def weights_init(module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, 0, np.sqrt(1/module.in_channels))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    net.apply(weights_init)
    # Set last conv2d layer's weights to 0
    torch.nn.init.zeros_(dict(net.named_children())['conv{}'.format(num_layers - 1)].weight)
    outimg = raw_out(lambda: net(input_tensor),img) if args.netype == 'raw' else to_valid_out(lambda: net(input_tensor),img,seg)
    return net.parameters(), outimg

def siren(args, wrapper, vae, img = None, seg = None, batch=None, num_output_channels=1, num_hidden_channels=128, num_layers=8,
         activation_fn=CompositeActivation, normalize=False, device = "cuda:0"):
    vae_img = torchvision.transforms.Resize(64)(img)
    latent = vae.encoder(vae_img).view(-1).detach()
    outimg = raw_out(lambda: wrapper(latent = latent),img) if args.netype == 'raw' else to_valid_out(lambda: wrapper(latent = latent),img,seg)
    # img = torch.randn(1, 3, 256, 256)
    # loss = wrapper(img)
    # loss.backward()

    # # after much training ...
    # # simply invoke the wrapper without passing in anything

    # pred_img = wrapper() # (1, 3, 256, 256)
    return wrapper.parameters(), outimg

def render_vis(
    args,
    model,
    objective_f,
    real_img,
    param_f=None,
    optimizer=None,
    transforms=None,
    thresholds=(256,),
    verbose=True,
    preprocess=True,
    progress=True,
    show_image=True,
    save_image=False,
    image_name=None,
    show_inline=False,
    fixed_image_size=None,
    label = 1,
    raw_img = None,
    prompt = None
):
    if label == 1:
        sign = 1
    elif label == 0:
        sign = -1
    else:
        print('label is wrong, label is',label)
    if args.reverse:
        sign = -sign
    if args.multilayer:
        sign = 1

    '''prepare'''
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y, %H:%M:%S")

    netD, optD = pre_d()
    '''end'''

    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()
    
    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=5e-1)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = []
    transforms = transforms.copy()

    # Upsample images smaller than 224
    image_shape = image_f().shape

    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    hook = hook_model(model, image_f)
    objective_f = objectives.as_objective(objective_f)

    if verbose:
        model(transform_f(image_f()))
        print("Initial loss of ad: {:.3f}".format(objective_f(hook)))

    images = []
    try:
        for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
            optimizer.zero_grad()
            try:
                model(transform_f(image_f()))
            except RuntimeError as ex:
                if i == 1:
                    # Only display the warning message
                    # on the first iteration, no need to do that
                    # every iteration
                    warnings.warn(
                        "Some layers could not be computed because the size of the "
                        "image is not big enough. It is fine, as long as the non"
                        "computed layers are not used in the objective function"
                        f"(exception details: '{ex}')"
                    )
            if args.disc:
                '''dom loss part'''
                # content_img = raw_img
                # style_img = raw_img
                # precpt_loss = run_precpt(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, transform_f(image_f()))
                for p in netD.parameters():
                    p.requires_grad = True
                for _ in range(args.drec):
                    netD.zero_grad()
                    real = real_img
                    fake = image_f()
                    # for _ in range(6):
                    #     errD, D_x, D_G_z1 = update_d(args, netD, optD, real, fake)

                    # label = torch.full((args.b,), 1., dtype=torch.float, device=device)
                    # label.fill_(1.)
                    # output = netD(fake).view(-1)
                    # errG = nn.BCELoss()(output, label)
                    # D_G_z2 = output.mean().item()
                    # dom_loss = err
                    one = torch.tensor(1, dtype=torch.float)
                    mone = one * -1
                    one = one.cuda(args.gpu_device)
                    mone = mone.cuda(args.gpu_device)

                    d_loss_real = netD(real)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)

                    d_loss_fake = netD(fake)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)

                    # Train with gradient penalty
                    gradient_penalty = calculate_gradient_penalty(netD, real.data, fake.data)
                    gradient_penalty.backward()


                    d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    Wasserstein_D = d_loss_real - d_loss_fake
                    optD.step()

                # Generator update
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation

                fake_images = image_f()
                g_loss = netD(fake_images)
                g_loss = -g_loss.mean()
                dom_loss = g_loss
                g_cost = -g_loss

                if i% 5 == 0:
                    print(f' loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
                    print(f'Generator g_loss: {g_loss}')
                '''end'''



            '''ssim loss'''

            '''end'''

            if args.disc:
                loss = sign * objective_f(hook) + args.pw * dom_loss
                # loss = args.pw * dom_loss
            else:
                loss = sign * objective_f(hook)
                # loss = args.pw * dom_loss

            loss.backward()

            # #video the images
            # if i % 5 == 0:
            #     print('1')
            #     image_name = image_name[0].split('\\')[-1].split('.')[0] + '_' + str(i) + '.png'
            #     img_path = os.path.join(args.path_helper['sample_path'], str(image_name))
            #     export(image_f(), img_path)
            # #end
            # if i % 50 == 0:
            #     print('Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #       % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            optimizer.step()
            if i in thresholds:
                image = tensor_to_img_array(image_f())
                # if verbose:
                #     print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                if save_image:
                    na = image_name[0].split('\\')[-1].split('.')[0] + '_' + str(i) + '.png'
                    na = date_time + na
                    outpath = args.quickcheck if args.quickcheck else args.path_helper['sample_path']
                    img_path = os.path.join(outpath, str(na))
                    export(image_f(), img_path)
                
                images.append(image)
    except KeyboardInterrupt:
        print("Interrupted optimization at step {:d}.".format(i))
        if verbose:
            print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
        images.append(tensor_to_img_array(image_f()))

    if save_image:
        na = image_name[0].split('\\')[-1].split('.')[0] + '.png'
        na = date_time + na
        outpath = args.quickcheck if args.quickcheck else args.path_helper['sample_path']
        img_path = os.path.join(outpath, str(na))
        export(image_f(), img_path)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())
    return image_f()

class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None


    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output


    def close(self):
        self.hook.remove()

def hook_model(model, image_f):
    features = OrderedDict()
    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook

def dot_compare(layer, batch=1, cossim_pow=0):
  def inner(T):
    dot = (T(layer)[batch] * T(layer)[0]).sum()
    mag = torch.sqrt(torch.sum(T(layer)[0]**2))
    cossim = dot/(1e-6 + mag)
    return -dot * cossim ** cossim_pow
  return inner

def init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def pre_d():
    netD = Discriminator(3).to(device)
    # netD.apply(init_D)
    beta1 = 0.5
    dis_lr = 0.00002
    optimizerD = optim.Adam(netD.parameters(), lr=dis_lr, betas=(beta1, 0.999))
    return netD, optimizerD

def update_d(args, netD, optimizerD, real, fake):
    criterion = nn.BCELoss()

    label = torch.full((args.b,), 1., dtype=torch.float, device=device)
    output = netD(real).view(-1)
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    label.fill_(0.)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    return errD, D_x, D_G_z1

def calculate_gradient_penalty(netD, real_images, fake_images):
    eta = torch.FloatTensor(args.b,1,1,1).uniform_(0,1)
    eta = eta.expand(args.b, real_images.size(1), real_images.size(2), real_images.size(3)).to(device = device)

    interpolated = (eta * real_images + ((1 - eta) * fake_images)).to(device = device)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(
                                prob_interpolated.size()).to(device = device),
                            create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty

