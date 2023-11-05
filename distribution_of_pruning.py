import torch
import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import time
import argparse
from compressai_modify_ssd.datasets import ImageFolder
from torch.utils.data import DataLoader
from compressai_modify_ssd.utils.eval_model.__main__ import inference
from torchvision import transforms
import math
from compressai_modify_ssd.zoo import bmshj2018_hyperprior




from sys import exit



##############################################
# GETTING LAYER FOR PRUING
############################################
def get_encoder_deocder_layer(model, layer_name, network_part):

    """
    @model: CODEC
    @layer_name: encoder or decoder layer name [e.g. "g_a" or "g_s"]
    @network_part: which part's layer we need [e.g. "encoder" or "decoder"]
    @output: [all encoder layer except last layer], [encoder/decoder all layers], [layer id]

    #What is the functionality of this function?
    In this function works for getting layer from the network. Here I mention
    network_part, that's mean if you need only encoder or decoder then you
    can mention the part.
    """


    layers = []
    layer_id = []


    if str(network_part)== "encoder":
        for name, layer in model.named_modules():
            if str(layer_name) in name:
                if isinstance(layer, nn.Conv2d):
                    layers.append(layer)
                    layer_id.append(int(name[-1:]))

    if str(network_part)== "decoder":
            if str(layer_name) in name:
                if isinstance(layer, nn.ConvTranspose2d):
                    layers.append(layer)
                    layer_id.append(int(name[-1:]))


    return layers[:-1], layers, layer_id



def get_full_network_layer(model, encoder_layer, decoder_layer, network_part):

    """
    @model: CODEC
    @encoder_layer: encoder layer name [e.g. "g_a"]
    @decoder_layer: decoder layer name [e.g. "g_s"]
    @network_part: which part's layer we want [e.g. "encoder" or "decoder"]
    @output: [all encoder layer except last layer], [encoder all layers], [all decoder layer except last layer], [decoder all layers], [layer id]

    #What is the functionality of this function?
    In this function works for getting the full network ( encoder and decoder ) layers togather.
    """

    encoder = []
    decoder = []
    layer_id = []

    for name, layer in model.named_modules():
        if str(encoder_layer) in name:
            if isinstance(layer, nn.Conv2d):
                encoder.append(layer)
                layer_id.append(int(name[-1:]))

        if str(decoder_layer) in name:
            if isinstance(layer, nn.ConvTranspose2d):
                decoder.append(layer)
                layer_id.append(int(name[-1:]))

    return encoder[:-1], encoder, decoder[:-1], decoder, layer_id




def get_full_network_layer_weight(model, encoder_layer, decoder_layer, network_part):

    """
    @model: CODEC
    @encoder_layer: encoder layer name [e.g. "g_a"]
    @decoder_layer: decoder layer name [e.g. "g_s"]
    @network_part: which part's layer we want [e.g. "encoder" or "decoder"]
    @output:  [encoder all layers weights], [decoder all layers weights], [layer id]

    #What is the functionality of this function?
    In this function works for getting the full network ( encoder and decoder ) layers weights together.
    """


    layer_id = []
    encoder_weight =[]
    decoder_weight=[]

    for name, layer in model.named_modules():
        if str(encoder_layer) in name:
            if isinstance(layer, nn.Conv2d):
                encoder_weight.append(layer.weight)
                layer_id.append(int(name[-1:]))

        if str(decoder_layer) in name:
            if isinstance(layer, nn.ConvTranspose2d):
                decoder_weight.append(layer.weight)
                layer_id.append(int(name[-1:]))

    return encoder_weight, decoder_weight, layer_id



##############################################
# GROUP REGULARIZE
############################################
def __grouplasso_reg(groups, strength, dim):

    """
    @groups: Input groups where groups is either "layer" or "filters"
    @strength: Regularizer Strength ( Alpha values)
    @dim: dimension of input
    """
    if dim == -1:
    # We only have single group
        return groups.norm(2) * strength
    return groups.norm(2, dim=dim).sum().mul_(strength)

def __3d_filterwise_reg(layer_weights, strength):

    """Group Lasso with group = 3D weights filter
    @layer_weights: CNN layer weights
    @strength: Reglarizer Strength
    """
    assert layer_weights.dim() == 4, "This regularization is only supported for 4D weights"

    # create a filter structure
    filters_view = layer_weights.view(layer_weights.size(0), -1)
    return __grouplasso_reg(filters_view, strength, dim=1)


def _channels_l2(layer_weights,strength):
        epsilon=sys.float_info.epsilon

        """Compute the L2-norm of convolution input channels weights.

        A weights input channel is composed of all the kernels that are applied to the
        same activation input channel.  Each kernel belongs to a different weights filter.
        """
        # Now, for each group, we want to select a specific channel from all of the filters
        num_filters = layer_weights.size(0)
        #print("\nNumber of Filters", num_filters)
        num_kernels_per_filter = layer_weights.size(1)
        #print("\nNumber of Filters", num_kernels_per_filter)

        # First, reshape the weights tensor such that each channel (kernel) in the original
        # tensor, is now a row in the 2D tensor.
        view_2d = layer_weights.view(-1, layer_weights.size(2) * layer_weights.size(3))
        # Next, compute the sum of the squares (of the elements in each row/kernel)
        k_sq_sums = view_2d.pow(2).sum(dim=1)
        # Now we have a long vector in which the first num_kernels_per_filter elements
        # hold the sum-of-squares of channels 1..num_kernels_per_filter of the 1st filter,
        # the second num_kernels_per_filter hold the sum-of-squares of channels
        # 1..num_kernels_per_filter of the 2nd filter, and so on.
        # Let's reshape this vector into a matrix, in which each row holds
        # the sum-of-squares of the channels of some filter
        k_sq_sums_mat = k_sq_sums.view(num_filters, num_kernels_per_filter).t()

        # Now it's easy, just do Group Lasso on groups=rows
        channels_l2 = k_sq_sums_mat.sum(dim=1).add(epsilon).pow(1/2.)
        #print("\n channels_l2", channels_l2)

        assert layer_weights.dim() == 4, "This regularization is only supported for 4D weights"

        # Sum of all channel L2s * regulization_strength
        layer_channels_l2 =channels_l2.sum().mul_(strength)
        #print("\n layer_channels_l2", layer_channels_l2)


        return layer_channels_l2

##############################################
# L1 / L2 REGULARIZE
############################################
def regularizer(model, alpha):
    '''
    @model: CODEC
    @alpha: Regularizer Strength
    '''
    l1=sum(abs(p).sum() for p in model.parameters())
    g= l1 * alpha
    return g


def group_regularizer(model, alpha, encoder_layer, decoder_layer, network_part, group_type):

    """
    @model: CODEC
    @alpha: Regularizer Strength ( A value between 0 to 1 )
    @encoder_layer: encoder layer name [e_g. "g_a"]
    @decoder_layer: decoder layer name [e_g. "g_s"]
    @network_part: which part's layer we want [e.g. "encoder" or "decoder"]
    @group_type: which regularizer we want to use [ "channel", "filter", "l1", "l2"]
    @output: regularizered values

    #What is the functionality of this function?
    This is the most important function in the pruning algorithm
    This function turns the network's weights smaller, so that we
    can apply a threshold to pruning out some filters in the network.
    """

    encoder_, decoder_, _ = get_full_network_layer_weight(model, encoder_layer,decoder_layer,  network_part)

    if str(group_type)=="channel":

        if str(network_part)=="encoder":
            g = torch.zeros([1], requires_grad=True).cuda()
            for idx, conv in enumerate(encoder_):
                channel=_channels_l2(conv, alpha)
                g = g + channel

        elif str(network_part)=="decoder":
            g = torch.zeros([1], requires_grad=True).cuda()
            for idx, conv in enumerate(decoder_):
                channel=_channels_l2(conv, alpha)
                g = g + channel

        elif str(network_part)=="full_network":
            g = torch.zeros([1], requires_grad=True).cuda()
            full= encoder_ + decoder_
            for idx, conv in enumerate(full):

                channel=_channels_l2(conv, alpha)
                g = g + channel

    if str(group_type)=="filter":

        if str(network_part)=="encoder":
            g = torch.zeros([1], requires_grad=True).cuda()
            for idx, conv in enumerate(encoder_):
                filter=_channels_l2(conv, alpha)
                g = g + filter

        elif str(network_part)=="decoder":
            g = torch.zeros([1], requires_grad=True).cuda()
            for idx, conv in enumerate(decoder_):
                filter=_channels_l2(conv, alpha)
                g = g + filter

        elif str(network_part)=="full_network":
            g = torch.zeros([1], requires_grad=True).cuda()
            full= encoder_ + decoder_
            for idx, conv in enumerate(full):

                filter=_channels_l2(conv, alpha)
                g = g + filter

    if str(group_type)=="l1":
        l1=sum(abs(p).sum() for p in model.parameters())
        g = l1 * alpha



    if str(group_type)=="l2":

        l2=sum(p.pow(2).sum() for p in model.parameters())
        g= l2 * alpha

    if str(group_type)=="both":

        l2=sum(p.pow(2).sum() for p in model.parameters())
        l1=sum(abs(p).sum() for p in model.parameters())
        #l2=sum(p.pow(2).sum() for p in model.parameters())
        g = abs(l2-l1) * alpha


    return g




##############################################
# FILTER COUNTING
############################################

def get_all_layers(model):
    """
    @model:CODEC
    @output: get the all layer in the CODEC
    """
    layer = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            layer.append(m)
        if isinstance(m, nn.ConvTranspose2d):
            layer.append(m)
    return layer


def count_total_filters(model):
    """
    @model: CODEC
    @output: Count the all filters in the network
    """

    conv_layer=get_all_layers(model)

    total=0
    for i in range(len(conv_layer)):
        for j in range (conv_layer[i].weight.size(0)): #weight.size(0) it's bring the number of filters.
                total+=1
    return total



def count_pruned_filters(model, encoder_layer_name, decoder_layer_name, pruning_part):

    """
    @model: CODEC
    @encoder_layer: encoder layer name [e.g. "g_a"]
    @decoder_layer: decoder layer name [e.g. "g_s"]
    @pruning_part: which part's layer we want [e.g. "encoder" or "decoder"]
    @output: number of pruned filters ( it's mean number of zeroed filters of output channel)
    """
    encoder_layer,_,decoder_layer,_,_= get_full_network_layer(model, encoder_layer_name, decoder_layer_name, pruning_part)

    if str(pruning_part)== "encoder":

        prune_filter=0
        for i in range(len(encoder_layer)):
            for j in range (encoder_layer[i].weight.size(0)): #encoder_layer[i].weight.size(0), it gives number of out_channel in encoder layer

                if abs(encoder_layer[i].weight[j]).sum() == 0: #sum the weight of the filters, if sum is zero then count it
                    prune_filter+=1

    if str(pruning_part)== "decoder":
        prune_filter=0
        for i in range(len(decoder_layer)):
            for j in range (decoder_layer[i].weight[0].size(0)): #decoder_layer[i].weight[0].size(0), it gives number of out_channel in decoder layer
                #print(f'non zero filters:\n {decoder_layer[i].weight[0][j]}')
                if abs(decoder_layer[i].weight[0][j]).sum() == 0: #sum the weight of the filters, if sum is zero then count it
                    #print(f'zero filters:\n {decoder_layer[i].weight[0][j]}')
                    prune_filter+=1



    return prune_filter


def count_alive_filters(model, encoder_layer_name, decoder_layer_name, pruning_part):
    '''
    @model: CODEC
    @encoder_layer: encoder layer name [e.g. "g_a"]
    @decoder_layer: decoder layer name [e.g. "g_s"]
    @network_part: which part's layer we want [e.g. "encoder" or "decoder"]
    @output: [all encoder layer except last layer], [encoder all layers], [all decoder layer except last layer],
     [decoder all layers], [layer id]
    '''
    encoder_layer,_,decoder_layer,_,_= get_full_network_layer(model, encoder_layer_name, decoder_layer_name, pruning_part)

    if str(pruning_part)== "encoder":
        alive_filter=0
        for i in range(len(encoder_layer)):
            for j in range (encoder_layer[i].weight.size(0)):
                if abs(encoder_layer[i].weight[j]).sum() != 0:
                    alive_filter+=1

    if str(pruning_part)== "decoder":
        alive_filter=0
        for i in range(len(decoder_layer)):
            for j in range (decoder_layer[i].weight[0].size(0)):
                if abs(decoder_layer[i].weight[0][j]).sum() != 0:
                    alive_filter+=1

    if str(pruning_part)== "full_network":

        dalive_filter=0
        ealive_filter=0


        for i in range(len(decoder_layer)):
            for j in range (decoder_layer[i].weight[0].size(0)):
                if abs(decoder_layer[i].weight[0][j]).sum() > 0:
                    dalive_filter+=1

        for i in range(len(encoder_layer)):
            for j in range (encoder_layer[i].weight.size(0)):
                if abs(encoder_layer[i].weight[j]).sum() > 0:
                    ealive_filter+=1

        alive_filter=ealive_filter + dalive_filter

    return alive_filter




def count_prunable_filters(model, encoder_layer_name, decoder_layer_name, pruning_part):
    '''
    @model: CODEC
    @encoder_layer: encoder layer name [e.g. "g_a"]
    @decoder_layer: decoder layer name [e.g. "g_s"]
    @network_part: which part's layer we want [e.g. "encoder" or "decoder"]
    @output: [all encoder layer except last layer], [encoder all layers], [all decoder layer except last layer],
     [decoder all layers], [layer id]
    '''
    encoder_layer,_,decoder_layer,_,_= get_full_network_layer(model, encoder_layer_name, decoder_layer_name, pruning_part)

    if str(pruning_part)== "encoder":

        filters=0
        for i in range(len(encoder_layer)):
            for j in range (encoder_layer[i].weight.size(0)):
                    filters+=1

    if str(pruning_part)== "decoder":
        filters=0
        for i in range(len(decoder_layer)):
            for j in range (decoder_layer[i].weight[0].size(0)):
                    filters+=1

    if str(pruning_part)== "full_network":

        efilters=0
        dfilters=0


        for i in range(len(decoder_layer)):
            for j in range (decoder_layer[i].weight[0].size(0)):
                    dfilters+=1

        for i in range(len(encoder_layer)):
            for j in range (encoder_layer[i].weight.size(0)):
                    efilters+=1

        filters= efilters+dfilters

    return filters


def percentage_of_pruning(model, encoder_layer_name, decoder_layer_name, pruning_part):
    '''
    @model: CODEC
    '''
    return float(count_pruned_filters(model, encoder_layer_name, decoder_layer_name,
                                      pruning_part)/count_prunable_filters(model, encoder_layer_name,
                                                                           decoder_layer_name, pruning_part))*100

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


##############################################
# THRESHOLD APPLY
############################################

def weight_threshold(model, thr, encoder_layer_name, decoder_layer_name, pruning_part):
    '''
    @model: CODEC
    @thr: threshold for making weights zero
    @encoder_layer: encoder layer name [e.g. "g_a"]
    @decoder_layer: decoder layer name [e.g. "g_s"]
    @network_part: which part's layer we want [e.g. "encoder" or "decoder"]
    @output: [all encoder layer except last layer], [encoder all layers], [all decoder layer except last layer],
    [decoder all layers], [layer id]
    '''
    encoder_layer,_,decoder_layer,_,_= get_full_network_layer(model, encoder_layer_name, decoder_layer_name, pruning_part)


    if str(pruning_part)== "encoder":
        for idx, conv in enumerate(encoder_layer):
            weights = conv[idx].weight.data.cpu().numpy()
            idx_out = np.abs(weights) < thr
            weights[idx_out] = 0
            conv[idx].weight.data = torch.from_numpy(weights).cuda()

    if str(pruning_part)== "decoder":
        for idx, deconv in enumerate(decoder_layer):
            weights = deconv.weight.data.cpu().numpy()
            idx_out = np.abs(weights) < thr
            weights[idx_out] = 0
            deconv.weight.data = torch.from_numpy(weights).cuda()




    '''
    if str(pruning_part)== "full_network":
        for idx, conv in enumerate(encoder_layer):
            weights = conv.weight.data.cpu().numpy()
            idx_out = np.abs(weights) < thr
            weights[idx_out] = 0
            conv.weight.data = torch.from_numpy(weights).cuda()

        for idx, deconv in enumerate(decoder_layer):
            weights = deconv.weight.data.cpu().numpy()
            idx_out = np.abs(weights) < thr
            weights[idx_out] = 0
            deconv.weight.data = torch.from_numpy(weights).cuda()
            
    '''



##############################################
# SAVE CHECKPOINT
############################################

def save_checkpoint(state, is_best, filename='checkpoint'):
    if is_best:
        torch.save(state, '{}_best.pth.tar'.format(filename))



class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""


    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * pow(255,2) * out["mse_loss"] + out["bpp_loss"]


        return out



def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""


    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.lr,
    )


    return optimizer, aux_optimizer


##############################################
# TESTING
############################################
def performance_test(model, loader):
    '''
    @model: CODEC
    @loader: Testing dataset
    '''
    model.eval()
    psnr=[]
    bpp=[]
    for i, batch in enumerate(loader):
        batch= batch.to('cuda')
        res=inference(model, batch)

        psnr.append(res['psnr(y)'])
        bpp.append(res['bpp'])

    return np.mean(psnr), np.mean(bpp)

##############################################
# TRAIN ONE EPOCH
############################################

def pruning_train_one_epoch(model, optimizer, criterion, loader, alpha=None, encoder_layer_name=None,
                            decoder_layer_name=None, pruning_part=None, regularizer=None):
    '''
    @model: CODEC
    @optim: optimizer
    @criterion: loss function
    @loader: training dataset
    @alpha: Regularizer strength
    '''

    model.train()
    step_count=0


    time_start = time.time()

    train_loss = 0
    reg= 0
    reg_loss = 0

    for i, img in enumerate(loader):

        for batch in img:
            optimizer.zero_grad()
            batch=batch.to(device)
            #batch.unsqueeze(0)
            out = model(batch)

            group_penalty_term=group_regularizer(model,alpha, encoder_layer_name, decoder_layer_name,  pruning_part, regularizer)
            #group_penalty_term=group_regularizer(model,0.7, encoder_layer_name, decoder_layer_name,  pruning_part, "l2")

            out_criterion = criterion(out, batch)

            loss= out_criterion["loss"] + group_penalty_term
            loss.backward()
            #out_criterion["loss"].backward()
            optimizer.step()

            #apply threshold
            #weight_threshold(model, 0.001, encoder_layer_name, decoder_layer_name, pruning_part)

            #calculate training loss
            #train_loss += out_criterion["loss"].item() * batch.size(0)
            #train_loss += loss.item() * batch.size(0)

            #calculate regularizer
            #reg += group_penalty_term.item() * batch.size(0)

            # calculate loss with regularizer
            #reg_loss+= loss.item() * batch.size(0)

            step_count+=1
            model.update()


        if (i % 10 == 0) or (i == len(loader)-1):
            print('Train | Batch {}/{} --> {:.2f}% '.format(i, len(loader), ((i)/len(loader))*100))



    #train_loss_avg = float(f"{train_loss / (len(img) * len(loader.dataset)):.5f}")
    #reg_avg = float(f"{reg/(len(img) * len(loader.dataset)):.5f}")
    #reg_loss_avg = float(f"{reg_loss /(len(img) * len(loader.dataset)):.5f}")

    #print(f'\n\n train avg loss:', train_loss_avg)
    #print(f'\n Reg avg :', reg_avg)
    #print(f'\n Reg to loss avg :', reg_loss_avg)

    time_end=time.time()
    total_time = time_end - time_start
    print("\n----------- TIME ------------------------------------")
    print(f'Per epoch training time: {(total_time/60):.2f} MINUTEs |  {(total_time/60)*60:.2f} SECONDs \n')

    #print("----------- STEPS -----------------")
    #print(f'STEPS Per Epoch: {step_count} \n\n')

    #return train_loss_avg, reg_avg, reg_loss_avg



    return model



def test_epoch(test_dataloader, model, criterion):
    t0=time.time()
    model.eval().to('cuda')
    device = next(model.parameters()).device
    loss_count=0

    with torch.no_grad():

        for i,d in enumerate(test_dataloader):

            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            loss_count+=out_criterion["loss"].item() * d.size(0)

    avg_loss= float(f"{loss_count/len(test_dataloader.dataset):.5f}")


    return avg_loss

##############################################
# TRAINING PHASE
############################################

def train_for_pruning(model, train_loader, test_dataloader, val_loader, lbda=None, epochs=None,
                      lr=None, alpha=None, encoder_layer_name=None, decoder_layer_name=None,
                      pruning_part=None, regularizer=None):

    '''
    @model: CODEC
    @train_loader: training dataset
    @lbda: codec quality control parameter
    @epochs: number of iteration
    @lr: learning rate
    '''

    model = model.to('cuda')
    model.train()


    optimizer, _ = configure_optimizers(model,args)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(epochs*0.3), int(epochs*0.6), int(epochs*0.8)], gamma=0.2)
    criterion = RateDistortionLoss(lbda) # lambda value for loss function 0.2 collected from paper "computationally effificient Neural Image compression"

    # importing os module
    import os

    # Directory
    directory = str(lbda).replace('.', '_')

    # Parent Directory path
    parent_dir = "./cp/regularized/"

    # Path
    path = os.path.join(parent_dir, directory)

    os.makedirs(path, exist_ok=True)


    for e in range(epochs):
        print()
        print(f'Epoch: {e+1} #LR: {lr}| #Lambda: {lbda} | #Alpha: {alpha}')
        print()

        net=pruning_train_one_epoch(model, optimizer, criterion, train_loader, alpha=alpha,
                                encoder_layer_name=encoder_layer_name, decoder_layer_name=decoder_layer_name,
                                pruning_part=pruning_part, regularizer=regularizer)

        scheduler.step()
        model.update()

        torch.save(model, './cp/regularized/'+directory+'/{}_best.tar.pth'.format("pruned_model_alpha_" + str(alpha)+"_"+str(args.lbda)))

    psnr, bpp = performance_test(model, test_dataloader)
    print(f'PSNR: {psnr:.2f} | BPP: {bpp:.2f} \n\n')


    return model







def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='/data')
    #parser.add_argument("--dataset", type=str, default='torchvision.datasets.CIFAR10')
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--name", type=str, default='tahsir_pruned_codec')
    parser.add_argument("--reg", type=str, default='l1') #Regularizer[l1, l1, channel, filter regularizer]
    #parser.add_argument("--model", type=str, default='ft_mbnetv2')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4) # working lr: 1e-1 # Flop reducing parameter [1e- less, reduce a lot and 1e-more, reduce a bit]
    parser.add_argument("--alpha", type=float, default= 0.01) # Regularizer Stringth
    parser.add_argument("--lbda", type=float, default= 0.0130) # Quality controler for the codec
    #parser.add_argument("--prune_away", type=float, default=0.1, help='The constraint level in portion to the original network, e.g. 0.5 is prune away 50%')
    parser.add_argument("--pruning_part", type=str, default='encoder')
    parser.add_argument("--threshold", type=float, default= 0.001) #Threshold for Pruning
    parser.add_argument("--no_grow", action='store_true', default=True)
    #parser.add_argument("--pruner", type=str, default='FilterPrunerResNet', help='Different network require differnt pruner implementation')
    args = parser.parse_args()
    return args

def NRandomCrop(img,size, N):
  '''
  @img : image for cropping
  @size: cropping size
  @N: number of cropping

  '''
  tf = transforms.Compose(
        [transforms.RandomCrop(size), transforms.ToTensor()])
  lst_img=[] #list of cropped images as tensor
  for _ in range(N):
    tf_img= tf(img)
    lst_img.append(tf_img)

  return lst_img

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    args = get_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("-------------------------")
    print(device)


    # N number of random crops
    train_set = transforms.Compose(
        [(lambda crops: NRandomCrop(crops, 256, args.crop))

         ]
    )


    #train_set = transforms.Compose(
    #    [transforms.RandomCrop(256), transforms.ToTensor()]
    #)


    val_set = transforms.Compose(
        [transforms.CenterCrop(256), transforms.ToTensor()]
    )




    train_dataset = ImageFolder(args.datapath, split="train",transform=train_set)
    val_dataset = ImageFolder(args.datapath, split="val", transform=val_set)
    test_dataset = ImageFolder(args.datapath, split="test", transform=val_set)


    print("-------------------------")
    print("Train image:",len(train_dataset))
    print("Train with cropped (total image):",len(train_dataset)*args.crop)
    print("Validation:",len(val_dataset))
    print("Test:",len(test_dataset))
    print("Batch Size:",args.batch_size)
    print("Cropped:",args.crop)
    print("Pruning Part:",args.pruning_part)
    print("Regularizer Strength:", args.alpha)
    print("Number of Epochs:", args.epoch)
    #print("Threshold:", args.threshold)

    print("-------------------------\n\n\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=True,
        pin_memory=(device == "cuda"))

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=6,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )


    if args.lbda==0.0018:
        net = bmshj2018_hyperprior(quality=1, pretrained=False).eval().to(device)
        net.load_state_dict(torch.load("./cp/baseline/model_1_bmshj2018-hyperprior_0-dc9957c5.0018_mse_e120_jpegai5k_cropped_30_batch_16_lr1e-4_checkpoint.pth.tar_checkpoint_best_loss.pth.tar"))
        print(f'\n model= {1} | lambda: {args.lbda} \n')
       


    if args.lbda==0.0067:
        net = bmshj2018_hyperprior(quality=3, pretrained=False).eval().to(device)
        net.load_state_dict(torch.load("./cp/baseline/model_3_bmshj2018-hyperprior_0-3c35bb58.0067_mse_e120_jpegai5k_cropped_30_batch_16_lr1e-4_checkpoint.pth.tar_checkpoint_best_loss.pth.tar"))
        print(f'\n model= {3} | lambda: {args.lbda} \n')


    if args.lbda==0.0483:
        net = bmshj2018_hyperprior(quality=6, pretrained=False).eval().to(device)
        net.load_state_dict(torch.load("./cp/baseline/model_6_bmshj2018-hyperprior_0-db71d757.0483_mse_e120_jpegai5k_cropped_30_batch_16_lr1e-4_checkpoint.pth.tar_checkpoint_best_loss.pth.tar"))
        print(f'\n model= {6} | lambda: {args.lbda} \n')


    if args.lbda==0.18:
        net = bmshj2018_hyperprior(quality=8, pretrained=False).eval().to(device)
        net.load_state_dict(torch.load("./cp/baseline/model_8_bmshj2018-hyperprior_0-ba115683.18_mse_e120_jpegai5k_cropped_30_batch_16_lr1e-4_checkpoint.pth.tar_checkpoint_best_loss.pth.tar"))
        print(f'\n model= {8} | lambda: {args.lbda} \n')


    from matplotlib import pyplot as plt


    