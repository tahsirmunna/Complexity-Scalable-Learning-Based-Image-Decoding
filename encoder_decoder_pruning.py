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

    if str(pruning_part)== "full_network":

        eprune_filter=0
        dprune_filter=0

        for i in range(len(decoder_layer)):
            for j in range (decoder_layer[i].weight[0].size(0)):
                if abs(decoder_layer[i].weight[0][j]).sum() == 0:
                    dprune_filter+=1

        for i in range(len(encoder_layer)):
            for j in range (encoder_layer[i].weight.size(0)):
                if abs(encoder_layer[i].weight[j]).sum() == 0:
                    eprune_filter+=1
        prune_filter=eprune_filter+dprune_filter

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
            weights = conv.weight.data.cpu().numpy()
            idx_out = np.abs(weights) < thr
            weights[idx_out] = 0
            conv.weight.data = torch.from_numpy(weights).cuda()

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
            train_loss += loss.item() * batch.size(0)

            #calculate regularizer
            #reg += group_penalty_term.item() * batch.size(0)

            # calculate loss with regularizer
            #reg_loss+= loss.item() * batch.size(0)

            step_count+=1
            model.update()



        #weight_threshold(model, 0.001, encoder_layer_name, decoder_layer_name, pruning_part)




        if (i % 10 == 0) or (i == len(loader)-1):
            print('Train | Batch ({}/{}) | Prunable Filters: {} | Pruned Filters: {} ({:.2f} %)'.format(
                    i+1, len(loader),
                    count_prunable_filters(model, "g_a", "g_s", pruning_part= args.pruning_part),
                    count_pruned_filters(model, "g_a", "g_s", pruning_part= args.pruning_part),
                    percentage_of_pruning(model, "g_a", "g_s", pruning_part= args.pruning_part)))

    train_loss_avg = float(f"{train_loss / (len(img) * len(loader.dataset)):.5f}")
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
    return model, train_loss_avg



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
                      pruning_part=None, regularizer=None, thr=None):

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

    #while e>=epochs:

    model_param= count_parameters(model)
    train_best_loss = float("inf")
    #reg_train_best_loss = float("inf")
    test_best_loss=float("inf")
    val_best_loss = float("inf")

    #train_best_loss_list =[]
    #reg_train_best_loss_list=[]
    #reg_values=[]
    test_loss_list=[]
    val_loss_list = []

    for e in range(epochs):
        print()
        print(f'Epoch: {e+1} #LR: {lr}| #Lambda: {lbda} | #Alpha: {alpha} | #Threshold: {thr}')
        print()
        '''
        net, train_loss=pruning_train_one_epoch(model, optimizer, criterion, train_loader, alpha=alpha,
                                encoder_layer_name=encoder_layer_name, decoder_layer_name=decoder_layer_name,
                                pruning_part=pruning_part, regularizer=regularizer)
                                
        '''
        ##########################################################################################
        model.train()
        step_count = 0

        time_start = time.time()

        train_loss = 0
        reg_train_loss=0
        reg = 0
        reg_loss = 0

        for i, img in enumerate(train_loader):

            for batch in img:
                optimizer.zero_grad()
                batch = batch.to(device)
                # batch.unsqueeze(0)
                out = model(batch)

                group_penalty_term = group_regularizer(model, alpha, encoder_layer_name, decoder_layer_name,
                                                       pruning_part, regularizer)
                # group_penalty_term=group_regularizer(model,0.7, encoder_layer_name, decoder_layer_name,  pruning_part, "l2")

                out_criterion = criterion(out, batch)

                loss = out_criterion["loss"] + group_penalty_term
                loss.backward()
                # out_criterion["loss"].backward()
                optimizer.step()

                # apply threshold
                # weight_threshold(model, 0.001, encoder_layer_name, decoder_layer_name, pruning_part)

                # calculate training loss
                #train_loss += out_criterion["loss"].item()* batch.size(0)
                #reg_train_loss += loss.item()* batch.size(0)

                # calculate regularizer
                # reg += group_penalty_term.item() * batch.size(0)

                # calculate loss with regularizer
                # reg_loss+= loss.item() * batch.size(0)

                step_count += 1
                model.update()

                if e == epochs-1:
                    #print("ENCODER PRUNING....")
                    if pruning_part=='encoder':

                        weight_threshold(model,thr, encoder_layer_name, decoder_layer_name, "encoder")
                        if (i % 10 == 0) or (i == len(train_loader) - 1):
                            print('ENCODER PRUNING | Prunable Filters: {} | Pruned Filters: {} ({:.2f} %)'.format(
                                count_prunable_filters(model, "g_a", "g_s", pruning_part="encoder"),
                                count_pruned_filters(model, "g_a", "g_s", pruning_part="encoder"),
                                percentage_of_pruning(model, "g_a", "g_s", pruning_part="encoder")))


                    encoder_prunned_channel = layer_wise_encoder_decoder_pruning_filters_count(
                        model, encoder_layer_name="g_a", decoder_layer_name="g_s", pruning_part="encoder")

                    encoder_alive_filters = count_alive_filters(model, "g_a", "g_s", pruning_part="encoder")
                    encoder_prune_percentage = percentage_of_pruning(model, "g_a", "g_s", pruning_part="encoder")
                    encoder_prunable_filters = count_prunable_filters(model, "g_a", "g_s", pruning_part="encoder")
                    encoder_pruned_filter = count_pruned_filters(model, "g_a", "g_s", pruning_part="encoder")

                    if pruning_part=='decoder':

                        weight_threshold(model, thr, encoder_layer_name, decoder_layer_name, "decoder")
                        if (i % 10 == 0) or (i == len(train_loader) - 1):
                            print('DECODER PRUNING | Prunable Filters: {} | Pruned Filters: {} ({:.2f} %)'.format(
                                count_prunable_filters(model, "g_a", "g_s", pruning_part="decoder"),
                                count_pruned_filters(model, "g_a", "g_s", pruning_part="decoder"),
                                percentage_of_pruning(model, "g_a", "g_s", pruning_part="decoder")))

                    decoder_prunned_channel = layer_wise_encoder_decoder_pruning_filters_count(
                        model, encoder_layer_name="g_a", decoder_layer_name="g_s", pruning_part="decoder")

                    decoder_alive_filters = count_alive_filters(model, "g_a", "g_s", pruning_part="decoder")
                    decoder_prune_percentage = percentage_of_pruning(model, "g_a", "g_s", pruning_part="decoder")
                    decoder_prunable_filters = count_prunable_filters(model, "g_a", "g_s", pruning_part="decoder")
                    decoder_pruned_filter = count_pruned_filters(model, "g_a", "g_s", pruning_part="decoder")
                    decoder_param = count_parameters(model)
                    print(decoder_prunned_channel)






            #train_loss_avg = float(f"{train_loss / (len(train_loader.dataset))}")
            #reg_train_loss_avg = float(f"{reg_train_loss / (len(train_loader.dataset))}")

            # weight_threshold(model, 0.001, encoder_layer_name, decoder_layer_name, pruning_part)
            if (i % 10 == 0) or (i == len(train_loader) - 1):
                print('Train | Batch ({}/{}) --> ({:.1f}%)'.format(i, len(train_loader), (i/len(train_loader))*100))


        #train_loss_avg = float(f"{train_loss / (len(img) * len(train_loader.dataset)):.5f}")
        # reg_avg = float(f"{reg/(len(img) * len(loader.dataset)):.5f}")
        # reg_loss_avg = float(f"{reg_loss /(len(img) * len(loader.dataset)):.5f}")

        # print(f'\n\n train avg loss:', train_loss_avg)
        # print(f'\n Reg avg :', reg_avg)
        # print(f'\n Reg to loss avg :', reg_loss_avg)

        time_end = time.time()
        total_time = time_end - time_start
        print("\n----------- TIME ------------------------------------")
        print(f'Per epoch training time: {(total_time / 60):.2f} MINUTEs |  {(total_time / 60) * 60:.2f} SECONDs \n')

        ###########################################################################################


        scheduler.step()

        val_avg = test_epoch(val_loader, model, criterion)
        test_avg = test_epoch(test_dataloader, model, criterion)

        # train
        #train_best_loss = min(train_loss, train_best_loss)
        #train_best_loss_list.append(train_best_loss)
        #train_best_loss_list.append(train_loss)

        # reg_train_loss
        #reg_train_best_loss = min(reg_loss, reg_train_best_loss)
        #reg_train_best_loss_list.append(reg_train_best_loss)
        #reg_train_best_loss_list.append(reg_loss)

        #regularizer values
        #reg_values.append(reg)

        #test loss
        test_best_loss = min(test_avg, test_best_loss)
        test_loss_list.append(test_best_loss)
        #test_loss_list.append(test_avg)

        #val loss
        val_best_loss = min(val_avg, val_best_loss)
        val_loss_list.append(val_best_loss)

        model.update()

        print("\n----------- LOSS MONITORING ----------------------------")
        print(f'\n Test loss: {test_best_loss:.4f}')
        print(f'\n Val loss: {val_best_loss:.4f}')
        #print(f'\n Train loss: {train_best_loss:.4f}')
        print("\n--------------------------------------------------------")
        #np.save('train_loss_prune', train_best_loss_list)
        #np.save('reg_train_prune', reg_train_best_loss_list)
        #np.save('reg_values',reg_values)
        np.save('test_loss_'+str(pruning_part)+"_thr_"+str(thr)+"_alpha_"+str(alpha), test_loss_list)
        np.save('val_loss_'+str(pruning_part)+"_thr_"+str(thr)+"_alpha_"+str(alpha), val_loss_list)



    if pruning_part== "encoder":
        print("---------- SUMMARY -------------")
        print(
            f'Encoder Prunable Filters: {encoder_prunable_filters} \n'
            f'Encoder Pruned Filters: {encoder_pruned_filter}\n'
            f'Encoder Pruned Percentage: {encoder_prune_percentage:.2f} \n'
            f'Encoder Alive Filters: {encoder_alive_filters} \n')

    if pruning_part== "decoder":
        print("---------- SUMMARY -------------")
        print(
            f'Decoder Prunable Filters: {decoder_prunable_filters} \n'
            f'Decoder Pruned Filters: {decoder_pruned_filter}\n'
            f'Decoder Pruned Percentage: {decoder_prune_percentage :.2f} \n'
            f'Decoder Alive Filters: {decoder_alive_filters} \n')

        print(decoder_prunned_channel)



    return model


##############################################
# COUNT PRUNED FILTER LAYER WISE
############################################

def layer_wise_encoder_decoder_pruning_filters_count(model, encoder_layer_name, decoder_layer_name, pruning_part):

    '''
    @model: CODEC
    @layer_name: encoder or decoder layer name [e.g. "g_a", or "g_s"]
    @network_part: which part's layer we want [e.g. "encoder" or "decoder"]
    @output: a dictionaly with layer name and number of pruned filter
    '''
    _,encoder_layer,_,decoder_layer,layer_id=get_full_network_layer(model, encoder_layer_name, decoder_layer_name, pruning_part)

    if (pruning_part)=="encoder":


        zerod_filter_index={}

        for i in range(len(encoder_layer)):
            sum_of_kernel = list(torch.sum(torch.abs(encoder_layer[i].weight.view(encoder_layer[i].weight.size(0), -1)),
                                           dim=1).cpu().detach().numpy())
            count_zero_filter=sum_of_kernel.count(0)
            zerod_filter_index["layer_"+str(i+1)]= count_zero_filter

    if str(pruning_part)=="decoder":
        zerod_filter_index={}

        for i in range(len(decoder_layer)):
            sum_of_kernel = list(torch.sum(torch.abs(decoder_layer[i].weight[0].view(decoder_layer[i].weight[0].size(0), -1)),
                                           dim=1).cpu().detach().numpy())
            count_zero_filter=sum_of_kernel.count(0)
            zerod_filter_index["layer_"+str(i+1)]= count_zero_filter


    return zerod_filter_index


def layer_wise_full_network_pruning_filters_count(model, encoder_layer_name, decoder_layer_name, pruning_part):

    '''
    @model: CODEC
    @encoder_layer_name: encoder layer name [e.g. "g_a"]
    @decoder_layer_name: decoder layer name [e.g. "g_s"]
    @network_part: which part's layer we want [e.g. "encoder" or "decoder"]
    @output: a dictionary for encoder number of pruning filter, a dictionary for decoder number of pruning filter,
    '''


    _,encoder_layer,_,decoder_layer,layer_id=get_full_network_layer(model, encoder_layer_name, decoder_layer_name, pruning_part)

    encoder={}
    for i in range(len(encoder_layer)):
        encoder_sum_of_kernel = list(torch.sum(torch.abs(encoder_layer[i].weight.view(encoder_layer[i].weight.size(0), -1)),
                                               dim=1).cpu().detach().numpy())
        encoder_count_zero_filter=encoder_sum_of_kernel.count(0)
        encoder["layer_"+str(i+1)]= encoder_count_zero_filter


    decoder={}
    for i in range(len(decoder_layer)):
        decoder_sum_of_kernel = list(torch.sum(torch.abs(decoder_layer[i].weight[0].view(decoder_layer[i].weight[0].size(0), -1)),
                                               dim=1).cpu().detach().numpy())
        decoder_count_zero_filter=decoder_sum_of_kernel.count(0)
        decoder["layer_"+str(i+1)]= decoder_count_zero_filter



    return encoder, decoder

##############################################
# GET PRUNED ARCHITECTURE
############################################
def partial_network_pruning_architecture_with_weights(model, number_of_pruned_channel, name_of_part_to_be_pruned, encoder_layer_name,
                                                      decoder_layer_name, pruning_part):

    '''
    @model: CODEC
    @number_of_pruned_channel: a dictionary with layer name and number of pruned filters
    @name_of_part_to_be_pruned: The part where pruned occured [Example: model.g_a or model.g_s]. where g_a and g_s in the encoder and decoder part
    @encoder_layer: encoder layer name [e.g. "g_a"]
    @decoder_layer: decoder layer name [e.g. "g_s"]
    @pruning_part: name of the part that has pruned [e.g. "encoder", "decoder"]
    @output: pruned model ARCHITECTURE with weights
    '''



    channel_pruned_in_layer=[list(number_of_pruned_channel.values())[i] for i in range(len(list(number_of_pruned_channel.values())))]

    _,encoder_layer,_,decoder_layer,layer_id=get_full_network_layer(model, encoder_layer_name, decoder_layer_name, pruning_part)


    if str(pruning_part)== "encoder":
        for i, conv in enumerate(encoder_layer):
            if i==0:
                name_of_part_to_be_pruned[layer_id[i]]= torch.nn.Conv2d(
                in_channels=conv.in_channels,
                out_channels=int(conv.out_channels - channel_pruned_in_layer[i]),
                kernel_size=conv.kernel_size,
                stride=conv.stride, padding=conv.padding)

                name_of_part_to_be_pruned[layer_id[i]].weight.data = conv.weight.data
                name_of_part_to_be_pruned[layer_id[i]].bias.data = conv.bias.data



            elif i== len(encoder_layer)-1:
                name_of_part_to_be_pruned[layer_id[i]] = torch.nn.Conv2d(
                in_channels=int(conv.in_channels - channel_pruned_in_layer[i-1]),
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride, padding=conv.padding)

                name_of_part_to_be_pruned[layer_id[i]].weight.data = conv.weight.data
                name_of_part_to_be_pruned[layer_id[i]].bias.data = conv.bias.data


            else:
                name_of_part_to_be_pruned[layer_id[i]] = torch.nn.Conv2d(
                in_channels=int(conv.in_channels - channel_pruned_in_layer[i-1]),
                out_channels=int(conv.out_channels - channel_pruned_in_layer[i]),
                kernel_size=conv.kernel_size,
                stride=conv.stride, padding=conv.padding)

                name_of_part_to_be_pruned[layer_id[i]].weight.data = conv.weight.data
                name_of_part_to_be_pruned[layer_id[i]].bias.data = conv.bias.data



    if str(pruning_part)== "decoder":


        for i, conv in enumerate(decoder_layer):
            if i==0:

                name_of_part_to_be_pruned[layer_id[i]]= torch.nn.ConvTranspose2d(
                in_channels=conv.in_channels,
                out_channels=int(conv.out_channels - channel_pruned_in_layer[i]),
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                output_padding=conv.output_padding)

                name_of_part_to_be_pruned[layer_id[i]].weight.data = conv.weight.data
                name_of_part_to_be_pruned[layer_id[i]].bias.data = conv.bias.data



            elif i== len(decoder_layer)-1:
                name_of_part_to_be_pruned[layer_id[i]] = torch.nn.ConvTranspose2d(
                in_channels=int(conv.in_channels - channel_pruned_in_layer[i-1]),
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                output_padding=conv.output_padding)

                name_of_part_to_be_pruned[layer_id[i]].weight.data = conv.weight.data
                name_of_part_to_be_pruned[layer_id[i]].bias.data = conv.bias.data


            else:
                name_of_part_to_be_pruned[layer_id[i]] = torch.nn.ConvTranspose2d(
                in_channels=int(conv.in_channels - channel_pruned_in_layer[i-1]),
                out_channels=int(conv.out_channels - channel_pruned_in_layer[i]),
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                output_padding=conv.output_padding)

                name_of_part_to_be_pruned[layer_id[i]].weight.data = conv.weight.data
                name_of_part_to_be_pruned[layer_id[i]].bias.data = conv.bias.data





    return model



def partial_network_pruning_architecture_without_weights(model, number_of_pruned_channel, name_of_part_to_be_pruned,
                                                         encoder_layer_name, decoder_layer_name, pruning_part):

    '''
    @model: CODEC
    @number_of_pruned_channel: a dictionary with layer name and number of pruned filters
    @name_of_part_to_be_pruned: The part where pruned occured [Example: model.g_a or model.g_s]. where g_a and g_s in the encoder and decoder part
    @encoder_layer: encoder layer name [e.g. "g_a"]
    @decoder_layer: decoder layer name [e.g. "g_s"]
    @pruning_part: name of the part that has pruned [e.g. "encoder", "decoder"]
    @output: pruned model ARCHITECTURE without weights
    '''



    channel_pruned_in_layer=[list(number_of_pruned_channel.values())[i] for i in range(len(list(number_of_pruned_channel.values())))]

    _,encoder_layer,_,decoder_layer,layer_id=get_full_network_layer(model, encoder_layer_name, decoder_layer_name, pruning_part)



    if str(pruning_part)== "encoder":



        for i, conv in enumerate(encoder_layer):
            if i==0:
                name_of_part_to_be_pruned[layer_id[i]]= torch.nn.Conv2d(
                    in_channels=conv.in_channels,
                    out_channels=int(conv.out_channels - channel_pruned_in_layer[i]),
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding)



            elif i== len(encoder_layer)-1:
                name_of_part_to_be_pruned[layer_id[i]] = torch.nn.Conv2d(
                in_channels=int(conv.in_channels - channel_pruned_in_layer[i-1]),
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding)



            else:
                name_of_part_to_be_pruned[layer_id[i]] = torch.nn.Conv2d(
                in_channels=int(conv.in_channels - channel_pruned_in_layer[i-1]),
                out_channels=int(conv.out_channels - channel_pruned_in_layer[i]),
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding)




    if str(pruning_part)== "decoder":


        for i, conv in enumerate(decoder_layer):
            if i==0:

                name_of_part_to_be_pruned[layer_id[i]]= torch.nn.ConvTranspose2d(
                in_channels=conv.in_channels,
                out_channels=int(conv.out_channels - channel_pruned_in_layer[i]),
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                output_padding=conv.output_padding)





            elif i== len(decoder_layer)-1:
                name_of_part_to_be_pruned[layer_id[i]] = torch.nn.ConvTranspose2d(
                in_channels=int(conv.in_channels - channel_pruned_in_layer[i-1]),
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                output_padding=conv.output_padding)



            else:
                name_of_part_to_be_pruned[layer_id[i]] = torch.nn.ConvTranspose2d(
                in_channels=int(conv.in_channels - channel_pruned_in_layer[i-1]),
                out_channels=int(conv.out_channels - channel_pruned_in_layer[i]),
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                output_padding=conv.output_padding)


    return model




def full_network_pruning_architecture_with_weights(model, encoder_prunned_channel, decoder_prunned_channel, encoder_part_to_be_pruned,
                                                   decoder_part_to_be_pruned, encoder_layer_name, decoder_layer_name,pruning_part):

    '''
    @model: CODEC
    @encoder_prunned_channel: a dictionary with encoder layer and number of pruned filters
    @decoder_prunned_channel: a dictionary with decoder layer and number of pruned filters
    @encoder_part_to_be_pruned: encoder part will be pruned [Example: model.g_a]. where g_a in the encoder
    @decoder_part_to_be_pruned: decoder part will be pruned [Example: model.g_s]. where g_s in the decoder
    @encoder_layer: encoder layer name [e.g. "g_a"]
    @decoder_layer: decoder layer name [e.g. "g_s"]
    @pruning_part: name of the part that has pruned [e.g. "encoder", "decoder"]
    @output: pruned model ARCHITECTURE with weights
    '''

    en_prunned_channel=[list(encoder_prunned_channel.values())[i] for i in range(len(list(encoder_prunned_channel.values())))]
    de_prunned_channel=[list(decoder_prunned_channel.values())[i] for i in range(len(list(decoder_prunned_channel.values())))]


    _,encoder_layer,_,decoder_layer,layer_id=get_full_network_layer(model, encoder_layer_name, decoder_layer_name, pruning_part)

    #ENCODER PART
    for i, conv in enumerate(encoder_layer):

        if i==0:

            encoder_part_to_be_pruned[layer_id[i]]= torch.nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=int(conv.out_channels - en_prunned_channel[i]),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding)

            encoder_part_to_be_pruned[layer_id[i]].weight.data = conv.weight.data
            encoder_part_to_be_pruned[layer_id[i]].bias.data = conv.bias.data



        elif i== len(encoder_layer)-1:
            encoder_part_to_be_pruned[layer_id[i]] = torch.nn.Conv2d(
            in_channels=int(conv.in_channels - en_prunned_channel[i-1]),
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding)

            encoder_part_to_be_pruned[layer_id[i]].weight.data = conv.weight.data
            encoder_part_to_be_pruned[layer_id[i]].bias.data = conv.bias.data


        else:
            encoder_part_to_be_pruned[layer_id[i]] = torch.nn.Conv2d(
            in_channels=int(conv.in_channels - en_prunned_channel[i-1]),
            out_channels=int(conv.out_channels - en_prunned_channel[i]),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding)

            encoder_part_to_be_pruned[layer_id[i]].weight.data = conv.weight.data
            encoder_part_to_be_pruned[layer_id[i]].bias.data = conv.bias.data


    #DECODER PART
    for i, conv in enumerate(decoder_layer):
        if i==0:

            decoder_part_to_be_pruned[layer_id[i+len(encoder_layer)]]= torch.nn.ConvTranspose2d(
            in_channels=conv.in_channels,
            out_channels=int(conv.out_channels - de_prunned_channel[i]),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            output_padding=conv.output_padding)

            decoder_part_to_be_pruned[layer_id[i+len(encoder_layer)]].weight.data = conv.weight.data
            decoder_part_to_be_pruned[layer_id[i+len(encoder_layer)]].bias.data = conv.bias.data



        elif i== len(decoder_layer)-1:
            decoder_part_to_be_pruned[layer_id[i+len(encoder_layer)]] = torch.nn.ConvTranspose2d(
            in_channels=int(conv.in_channels - de_prunned_channel[i-1]),
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            output_padding=conv.output_padding)

            decoder_part_to_be_pruned[layer_id[i+len(encoder_layer)]].weight.data = conv.weight.data
            decoder_part_to_be_pruned[layer_id[i+len(encoder_layer)]].bias.data = conv.bias.data


        else:
            decoder_part_to_be_pruned[layer_id[i+len(encoder_layer)]] = torch.nn.ConvTranspose2d(
            in_channels=int(conv.in_channels - de_prunned_channel[i-1]),
            out_channels=int(conv.out_channels - de_prunned_channel[i]),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            output_padding=conv.output_padding)

            decoder_part_to_be_pruned[layer_id[i+len(encoder_layer)]].weight.data = conv.weight.data
            decoder_part_to_be_pruned[layer_id[i+len(encoder_layer)]].bias.data = conv.bias.data




    return model



def full_network_pruning_architecture_without_weights(model, encoder_prunned_channel, decoder_prunned_channel,
                                                      encoder_part_to_be_pruned, decoder_part_to_be_pruned, encoder_layer_name,
                                                      decoder_layer_name, pruning_part):

    '''
    @model: CODEC
    @encoder_prunned_channel: a dictionary with encoder layer and number of pruned filters
    @decoder_prunned_channel: a dictionary with decoder layer and number of pruned filters
    @encoder_part_to_be_pruned: encoder part will be pruned [Example: model.g_a]. where g_a in the encoder
    @decoder_part_to_be_pruned: decoder part will be pruned [Example: model.g_s]. where g_s in the decoder
    @encoder_layer: encoder layer name [e.g. "g_a"]
    @decoder_layer: decoder layer name [e.g. "g_s"]
    @pruning_part: name of the part that has pruned [e.g. "encoder", "decoder"]
    @output: pruned model ARCHITECTURE without weights
    '''

    en_prunned_channel=[list(encoder_prunned_channel.values())[i] for i in range(len(list(encoder_prunned_channel.values())))]
    de_prunned_channel=[list(decoder_prunned_channel.values())[i] for i in range(len(list(decoder_prunned_channel.values())))]


    _,encoder_layer,_,decoder_layer,layer_id=get_full_network_layer(model, encoder_layer_name, decoder_layer_name, pruning_part)

    #ENCODER PART
    for i, conv in enumerate(encoder_layer):

        if i==0:

            encoder_part_to_be_pruned[layer_id[i]]= torch.nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=int(conv.out_channels - en_prunned_channel[i]),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding)


        elif i== len(encoder_layer)-1:
            encoder_part_to_be_pruned[layer_id[i]] = torch.nn.Conv2d(
            in_channels=int(conv.in_channels - en_prunned_channel[i-1]),
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding)


        else:
            encoder_part_to_be_pruned[layer_id[i]] = torch.nn.Conv2d(
            in_channels=int(conv.in_channels - en_prunned_channel[i-1]),
            out_channels=int(conv.out_channels - en_prunned_channel[i]),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding)


    #DECODER PART
    for i, conv in enumerate(decoder_layer):
        if i==0:

            decoder_part_to_be_pruned[layer_id[i+len(encoder_layer)]]= torch.nn.ConvTranspose2d(
            in_channels=conv.in_channels,
            out_channels=int(conv.out_channels - de_prunned_channel[i]),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            output_padding=conv.output_padding)


        elif i== len(decoder_layer)-1:
            decoder_part_to_be_pruned[layer_id[i+len(encoder_layer)]] = torch.nn.ConvTranspose2d(
            in_channels=int(conv.in_channels - de_prunned_channel[i-1]),
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            output_padding=conv.output_padding)


        else:
            decoder_part_to_be_pruned[layer_id[i+len(encoder_layer)]] = torch.nn.ConvTranspose2d(
            in_channels=int(conv.in_channels - de_prunned_channel[i-1]),
            out_channels=int(conv.out_channels - de_prunned_channel[i]),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            output_padding=conv.output_padding)


    return model

#######################
#VALIDATION AREA
######################






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
    print("Threshold:", args.threshold)

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


    model= bmshj2018_hyperprior(quality=1, pretrained=False).eval().to(device)
    model.load_state_dict(torch.load("./cp/baseline/model_1_bmshj2018-hyperprior_0-dc9957c5.0018_mse_e120_jpegai5k_cropped_30_batch_16_lr1e-4_checkpoint.pth.tar_checkpoint_best_loss.pth.tar"))
    #net=model


    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    np.random.seed(98)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    num_params=count_parameters(model)
    #num_filters_=count_total_filters(model)
    pruneable_filters_= count_prunable_filters(model, "g_a", "g_s", pruning_part= args.pruning_part)

    print(f'Before {str(args.pruning_part)} Pruning')
    #print(' #Params: {:.3f}M  | #Total out channel Filters: {}| #Prunable out channel Filters: {}'.format( num_params/1000000., num_filters_, count_prunable_filters(model, "g_a", "g_s", pruning_part= args.pruning_part)))
    print(' #Params: {:.3f}M  | #Prunable Filters: {}'.format( num_params/1000000., pruneable_filters_))


    print()

    print("\n #Training for Pruning is running...\n")

    model=train_for_pruning(model, train_loader, test_loader, val_loader, epochs=args.epoch, lr=args.lr, lbda=args.lbda, alpha=args.alpha,
                                   encoder_layer_name="g_a", decoder_layer_name="g_s",
                                   pruning_part=args.pruning_part, regularizer=args.reg, thr=args.threshold)



    if args.pruning_part=="encoder":

        encoder_prunned_channel= layer_wise_encoder_decoder_pruning_filters_count(model,
                                                                                  encoder_layer_name="g_a", decoder_layer_name="g_s",
                                                                                  pruning_part="encoder")
        print()
        print("Encoder Layer:", encoder_prunned_channel)
        print()
        new_encoder_pruned_model_without_weight= partial_network_pruning_architecture_without_weights(
            model, encoder_prunned_channel, model.g_a, encoder_layer_name="g_a",
            decoder_layer_name="g_s",pruning_part="encoder")
            #print(new_encoder_pruned_model_with_weight)


        num_params_pruned=count_parameters(new_encoder_pruned_model_without_weight)
        prunable_left = count_prunable_filters(model, "g_a", "g_s", args.pruning_part)
        print("--------------------")
        print(f'After {str(args.pruning_part)} Pruning')
        print(
            ' #Params: {:.3f}M (Before: {:.3f}M) | #Total Pruned: {} | # Prunable Filters left: {} (Before: {})'.format(
                num_params_pruned / 1000000.,
                num_params / 1000000.,
                pruneable_filters_ - prunable_left,
                prunable_left,
                pruneable_filters_))
        print("---------------------")


    if args.pruning_part=="decoder":


        decoder_prunned_channel= layer_wise_encoder_decoder_pruning_filters_count(
            model, encoder_layer_name="g_a", decoder_layer_name="g_s",pruning_part="decoder")
        print("Decoder layer:", decoder_prunned_channel)
        print()
        new_decoder_pruned_model_without_weight= partial_network_pruning_architecture_without_weights(
            model, decoder_prunned_channel, model.g_s, encoder_layer_name="g_a",
            decoder_layer_name="g_s",pruning_part="decoder")
            #print("new_architecture", new_decoder_pruned_model_with_weight)

        num_params_pruned=count_parameters(new_decoder_pruned_model_without_weight)
        prunable_left=count_prunable_filters(model, "g_a", "g_s", args.pruning_part)
        print("--------------------")
        print(f'After {str(args.pruning_part)} Pruning')
        print(
            ' #Params: {:.3f}M (Before: {:.3f}M) | #Total Pruned: {} | # Prunable Filters left: {} (Before: {})'.format(
                num_params_pruned / 1000000.,
                num_params / 1000000.,
                pruneable_filters_ -  prunable_left,
                prunable_left,
                pruneable_filters_))
        print("---------------------")



