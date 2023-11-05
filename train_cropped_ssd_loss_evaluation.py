# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import math
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai_modify_ssd.datasets import ImageFolder
from compressai_modify_ssd.zoo import image_models
from compressai_modify_ssd.utils.eval_model.__main__ import inference


import time

#import matplotlib.pyplot as plt
from sys import exit

from datetime import datetime
start_time = datetime.now()



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
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        #out["loss"] = out["mse_loss"] * self.lmbda + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    #t0=time.time()

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
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )


    return optimizer, aux_optimizer




def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, alpha
):
    t0=time.time()


    model.train().to('cuda')
    device = next(model.parameters()).device


    train_one_epoch.stpes_count=0

    last_time = time.time()


    train_one_epoch.t_sum = 0
    train_one_epoch.opt_sum=0

    full_train_loss = 0
    train_loss = 0
    for i, img in enumerate(train_dataloader):


        for j, d in enumerate(img):
            #print("this is j", j)

            d = d.to(device)

            t = time.time()

            t_diff = t- last_time

            last_time = time.time()
            train_one_epoch.t_sum += t_diff


            opt_t= time.time()

            optimizer.zero_grad()
            aux_optimizer.zero_grad()


            out_net = model(d)

            out_criterion = criterion(out_net, d)
            out_criterion["loss"].backward()


            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

            optimizer.step()

            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            # calculate training loss
            train_loss += out_criterion["loss"].item() * d.size(0)

            opt_end = time.time()
            opt_diff = opt_end- opt_t


            last_time = time.time()
            train_one_epoch.opt_sum += opt_diff



            #counting the updating steps
            train_one_epoch.stpes_count+=1

            model.update()

        if i % 10 == 0:


            print(

                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)} ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tWorking img:{len(img)*len(train_dataloader.dataset)}|'
                    f'\tLoss: {out_criterion["loss"].item():.3f}|'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f}|'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f}|'
                    f"\tAux loss: {aux_loss.item():.2f}"

               )
    #print("i", i)
    #print(len(train_dataloader.dataset))
    train_loss_avg = train_loss / (len(img)*len(train_dataloader.dataset))
    print(f'train avg loss:', train_loss_avg)



    print("======================================================")
    print(f'train_one_epoch function running time (Min): {(time.time()-t0)/60:.3f}')
    print("======================================================")

    return train_loss_avg


def val_epoch(epoch, val_dataloader, model, criterion):
    t0=time.time()
    model.eval().to('cuda')
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():

        for d in val_dataloader:

            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])


    print("++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        f"Validation epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    val_epoch.test_time= time.time()-t0


    return loss.avg

def test_epoch(epoch, test_dataloader, model, criterion):
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

    avg_loss=loss_count/len(test_dataloader.dataset)

    print("++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {avg_loss:.3f} |"
    )

    test_epoch.test_time= time.time()-t0


    return avg_loss

def test(model, loader):
    model.eval()

    psnr=[]
    bpp=[]

    for i, batch in enumerate(loader):
        batch= batch.to('cuda')
        res=inference(model, batch)

        psnr.append(res['psnr(y)'])
        bpp.append(res['bpp'])

    return np.mean(psnr), np.mean(bpp)




def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )

    parser.add_argument(
        "--alpha",
        dest="alpha",
        type=float,
        default=1e-9,
        help="Regularizer Strength (default: %(default)s)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )

    parser.add_argument(
        "--cropped",
        default=16,
        type=int,
        help="Number of random cropped you need per images (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def save_checkpoint(state, is_best, filename="pruned_model_8_mse_e120_jpegai5k_cropped_30_batch_16_lr1e-4_checkpoint.pth.tar"):

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+"_checkpoint_best_loss.pth.tar")

# =============================================================================
# This function cropped N number of times a single image.
# =============================================================================
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


def main(argv):

    st_time=time.time()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    '''
    #TenCrop by Pytorch
    train_transforms = transforms.Compose(
        [transforms.TenCrop(args.patch_size),
         (lambda crops: [transforms.ToTensor()(crop) for crop in crops])

         ]
    )



    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )


    '''
    # N number of random crops
    train_transforms = transforms.Compose(
        [(lambda crops: NRandomCrop(crops, args.patch_size, args.cropped))

         ]
    )


    val_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )


    train_dataset = ImageFolder(args.dataset, split="train",transform=train_transforms)
    val_dataset = ImageFolder(args.dataset, split="val", transform=val_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=val_transforms)
    test_performance = ImageFolder(args.dataset, split="test", transform=transforms.ToTensor())

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    print("-------------------------")
    print("Train image:", len(train_dataset)*args.cropped)
    print("Validation:",len(val_dataset))
    print("Test:",len(test_dataset))
    print("Batch Size:", args.batch_size)
    print(" Test batch Size:", args.test_batch_size)
    print("-------------------------\n")


    #cuda0 =torch.cuda.caching_allocator_alloc(19327352832, device=None, stream=None)
    cuda0 = torch.cuda.max_memory_allocated(device=None)

    print()
    print("Training with -->", device)


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"))

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )


    test_loader = DataLoader(
        test_performance,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )


# =============================================================================
# Quality | MSE
# 1       2     3       4      5      6      7      8
# 0.0018 0.0035 0.0067 0.0130 0.0250 0.0483 0.0932 0.1800
# =============================================================================


    if args.lmbda==0.0018:

        net = image_models[args.model](quality=1).to(device)
        print(f'CODEC name: {args.model} | model= {1} | lambda: {args.lmbda}')
        param=count_parameters(net)

    if args.lmbda==0.0035:

        net = image_models[args.model](quality=2).to(device)
        print(f'CODEC name: {args.model} | model= {2} | lambda: {args.lmbda}')
        param=count_parameters(net)

    if args.lmbda==0.0067:

        net = image_models[args.model](quality=3).to(device)
        print(args.lmbda)
        print(f'CODEC name: {args.model} | model= {3} | lambda: {args.lmbda}')
        param=count_parameters(net)
    if args.lmbda==0.0130:

        net = image_models[args.model](quality=4).to(device)
        print(f'CODEC name: {args.model} | model= {4} | lambda: {args.lmbda}')
        param=count_parameters(net)
    if args.lmbda==0.0250:

        net = image_models[args.model](quality=5).to(device)
        print(f'CODEC name: {args.model} | model= {5} | lambda: {args.lmbda}')
        param=count_parameters(net)
    if args.lmbda==0.0483:

        net = image_models[args.model](quality=6).to(device)
        print(f'CODEC name: {args.model} | model= {6} | lambda: {args.lmbda}')
        param=count_parameters(net)
    if args.lmbda==0.0932:

        net = image_models[args.model](quality=7).to(device)
        print(f'CODEC name: {args.model} | model= {7} | lambda: {args.lmbda}')
        param=count_parameters(net)
    if args.lmbda==0.1800:
        net = image_models[args.model](quality=8).to(device)
        print(f'CODEC name: {args.model} | model= {8} | lambda: {args.lmbda}')
        param=count_parameters(net)






    #############################################################

    print('\n')

    print('---------------------------------------------------')
    print('Model Starting Time: ',start_time)
    print('---------------------------------------------------')

    ############################################################


    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)



    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)



    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        print("=================================================================================")
        print("\n checkpoint last epoch",checkpoint["epoch"])
        print("\n checkpoint last loss",checkpoint["loss"])
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    train_best_loss= float("inf")
    test_best_loss=float("inf")

    loss_list = []
    train_avg_loss=[]
    test_avg_loss=[]

    save_best=0
    #avg_loss_count=0
    for epoch in range(last_epoch, args.epochs):

        ##########################################################################################
        epoch_start_time = datetime.now()
        print('\n')
        print(f'################################ EPOCH {epoch} / {args.epochs} STARTED ###############################')

        print('\n')
        print("################ TRAINING HYPER-PARAMETER ######################################")
        print(f'CODEC name: {args.model}')
        print()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print('Lambda:',args.lmbda)
        print()
        print('Number of cropped:',args.cropped)
        print("Number of parameter:", param)

        print('----------------------------------------')
        print()

        ##########################################################################################



        train_loss=train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            args.alpha
        )


        val_loss = val_epoch(epoch, val_dataloader, net, criterion)
        test_loss = test_epoch(epoch, test_dataloader, net, criterion)

        lr_scheduler.step(train_loss)
        #lr_scheduler.step(test_loss)

        is_best = train_loss < train_best_loss


        #val_loss
        best_loss = min(val_loss, best_loss).cpu()
        loss_list.append(best_loss)

        #train_loss
        train_best_loss = min(train_loss, train_best_loss)
        train_avg_loss.append(train_best_loss)

        #test_loss
        test_best_loss = min(test_loss, test_best_loss)
        test_avg_loss.append(test_best_loss)

        #Getting test report


        if epoch == args.epochs -1:
            psnr,bpp = test(net, test_loader)

            print("=============== Quality Measure ====================")
            print(f'PSNR(Y): {psnr:.2f} | BPP: {bpp:.2f}')
            print()
            


        ########################

        ##########################################################################################
        np.save('val_loss', loss_list)
        np.save('train_loss',train_avg_loss)
        np.save('test_loss', test_avg_loss)
        ##########################################################################################
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": train_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best, filename=str(args.model)+'_'+str(args.lmbda)+'_mse_e120_jpegai5k_cropped_30_batch_16_lr1e-4_checkpoint.pth.tar'
            )

            torch.cuda.caching_allocator_delete(cuda0)

        ##########################################################################################

        end_epoch = datetime.now()
        epoch_time_min= int(str(end_epoch - epoch_start_time)[2:4])/60
        epoch_time_sec= int(str(end_epoch - epoch_start_time)[5:7])/3600
        epoch_time_hour= epoch_time_min + epoch_time_sec

        ##########################################################################################


        ##########################################################################################
        print('###################### TRAINING TIME COUNTING #################################')

        print('Model Train Starting Time: {}'.format(start_time))
        print('Current Time: {}'.format(datetime.now()))
        print()

        print('Total epoch running time: {}'.format(end_epoch - start_time))
        print('Approximate Total Training Time (H): {}'.format((epoch_time_hour*args.epochs)))
        print()

        print('MODEL FINISHED (%): {} | Epoch left: {} '.format((epoch+1)*100/args.epochs, args.epochs-(epoch+1)))
        print()


        #final time
        print('######################## STEP CALCULATION ######################################')
        print('Total steps:', train_one_epoch.stpes_count*(epoch+1))
        print()

        print("############ DATA AND PROCESSING TIME CALCULATION ##############################")

        endtime= time.time()-st_time
        print(f"{epoch+1} epoch training time: {endtime/60:2f} minute" )
        per_epoch_loading_time=((train_one_epoch.t_sum*100)/endtime)
        print(f"percentage of loading data: {per_epoch_loading_time:.2f}%")

        per_epoch_opt_time=((train_one_epoch.opt_sum*100)/endtime)
        print(f"percentage of optimizing data: {per_epoch_opt_time:.2f}%")

        per_epoch_val_time=((test_epoch.test_time*100)/endtime)
        print(f"percentage of validation: {per_epoch_val_time:.2f}%")
        print()


        print("#################### AVG. LOSS MONITORING ######################################")

        if args.checkpoint:
            print(f'{epoch} epoch best loss value: {save_best:.4f}')
            print()
            #print(f'PSNR(Y): {psnr:.2f} | BPP: {bpp:.2f}')

        else:
            if epoch==0:

                print(f'{epoch} epoch best loss value: {save_best:.4f}')
                print()
                #print(f'PSNR(Y): {psnr:.2f} | BPP: {bpp:.2f}')

            elif epoch==1:

                print(f'After {epoch-1} epoch best loss value: {save_best:.4f}')
                print()
                #print(f'PSNR(Y): {psnr:.2f} | BPP: {bpp:.2f}')
            elif epoch>1:

                print(f'last epoch ({epoch-1}) best loss value was: {loss_list[epoch-1]:.4f}')
                print(f'After {epoch} epoch best loss value is: {save_best:.4f}')
                print()
               # print(f'PSNR(Y): {psnr:.2f} | BPP: {bpp:.2f}')

        print("################ TRAINING LEARNING RATE ########################################")

        print(f"After {epoch} epoch")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print()
        print(f'################################ EPOCH {epoch}/{args.epochs} FINISHED ##############################')
        print()
        print()



        st_time=time.time()

        ##########################################################################################


if __name__ == "__main__":
    main(sys.argv[1:])
