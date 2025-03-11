import argparse
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time

from dataloaders.kitti_loader import load_calib, input_options, KittiDepth
from metrics import AverageMeter, Result
import criteria
import helper
import vis_utils

from model import ENet
from model import PENet_C1_train
from model import PENet_C2_train
#from model import PENet_C4_train (Not Implemented)
from model import PENet_C1
from model import PENet_C2
from model import PENet_C4

from tqdm import tqdm
from config import args

args = args
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

print('\n=== Arguments ===')
cnt = 0
for key in sorted(vars(args)):
    print(key, ':', getattr(args, key), end = '\t\t')
    cnt += 1
    if (cnt + 1) % 1 == 0:
        print('')
# cuda = torch.cuda.is_available() and not args.cpu
cuda = True
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("------------------------------------------------")
print("using '{}' for computation.".format(device))
print("------------------------------------------------")

# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (
    args.criterion == 'l2') else criteria.MaskedL1Loss()

#multi batch
multi_batch_size = 1
def iterate(mode, args, loader, model, optimizer, logger, epoch):   # epoch: [n, 50]
    actual_epoch = epoch - args.start_epoch + args.start_epoch_bias

    block_average_meter = AverageMeter()
    block_average_meter.reset(False)

    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        # lr = helper.adjust_learning_rate_new(args.lr, optimizer, actual_epoch, epoch, args)
        lr = helper.adjust_learning_rate_old(args.lr, optimizer, actual_epoch, args)
    else:
        #  mode = "val", "eval", "test_prediction", "test_completion"
        model.eval()
        lr = 0

    torch.cuda.empty_cache()
    iter_train_loss = 0.0
    tqdm_bar = tqdm(loader)
    for i, batch_data in enumerate(tqdm_bar):
        dstart = time.time()
        # 一个字典  {"rgb": rgb, "d": sparse, "gt": target, "g": gray, 'position': position, 'K': self.K}
        batch_data = { key: val.to(device)
                        if val is not None and not isinstance(val[0], str) else val
                        for key, val in batch_data.items()
        }

        gt = batch_data['gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        batch_size = gt[0]
        data_time = time.time() - dstart

        pred = None
        start = None
        gpu_time = 0

        #start = time.time()
        #pred = model(batch_data)
        #gpu_time = time.time() - start

        #'''
        if(args.network_model == 'e'):
            start = time.time()
            st1_pred, st2_pred, pred = model(batch_data)        # ENet的输出： rgbd_depth, d_depth, output
        else:
            start = time.time()
            pred = model(batch_data)        # PeNet的输出： refined depth

        if(args.evaluate):
            gpu_time = time.time() - start
        #'''

        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None

        # loss的超参
        st1_loss, st2_loss, loss = 0, 0, 0
        w_st1, w_st2 = 0, 0                     # lamda_cd, lamda_dd
        round1, round2, round3 = 1, 3, None
        if(actual_epoch <= round1):
            w_st1, w_st2 = 0.2, 0.2
        elif(actual_epoch <= round2):
            w_st1, w_st2 = 0.05, 0.05
        else:
            w_st1, w_st2 = 0, 0

        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask　=　1 indicates that a pixel does not ground truth labels
            # 一个 batch 中所有图片 loss 的总和
            # l1 + l2
            # depth_loss, depth_loss_l1, depth_loss_l2 = depth_criterion(pred, gt)
            # l2
            depth_loss = depth_criterion(pred, gt)

            if args.network_model == 'e':
                # l1 + l2
                # st1_loss, st1_loss_l1, st1_loss_l2 = depth_criterion(st1_pred, gt)
                # st2_loss, st2_loss_l1, st2_loss_l2 = depth_criterion(st2_pred, gt)
                # l2
                st1_loss = depth_criterion(st1_pred, gt)
                st2_loss = depth_criterion(st2_pred, gt)
                loss = (1 - w_st1 - w_st2) * depth_loss + w_st1 * st1_loss + w_st2 * st2_loss
            else:
                loss = depth_loss

            if i % multi_batch_size == 0:
                optimizer.zero_grad()
            loss.backward()

            if i % multi_batch_size == (multi_batch_size-1) or i==(len(loader)-1):
                optimizer.step()

            # l1 + l2
            # error_str = "train epoch [{}/{}] | lr: {} | loss:{:.3f} | L1_loss: {:.3f} | L2_loss: {:.3f}".format(epoch, args.epochs, lr, loss, depth_loss_l1, depth_loss_l2)
            # l2
            error_str = "train epoch [{}/{}] | lr: {} | loss:{:.3f}".format(epoch, args.epochs, lr, loss)
            tqdm_bar.set_description(error_str)
            # tqdm_bar.desc = "train epoch [{}/{}] | lr: {} | st1_loss loss:{:.3f}".format(epoch, args.epochs, loss)

            iter_train_loss += loss.item()

        if mode == "test_completion":
            str_i = str(i)
            path_i = str_i.zfill(10) + '.png'       # 长度为10，右对齐，前面补零
            path = os.path.join(args.data_folder_save, path_i)
            vis_utils.save_depth_as_uint16png_upload(pred, path)

        if(not args.evaluate):      # 不eval模型时执行
            gpu_time = time.time() - start

        # 计算正确率，记录loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)       # 等于 batch_size
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                # "train", "val", "eval"时执行
                result.evaluate(pred.data, gt.data, photometric_loss)

                [ m.update(result, gpu_time, data_time, mini_batch_size) for m in meters ]

                if mode != 'train':
                    # "val", "eval"时执行
                    logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)

                # logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
                # logger.conditional_save_pred(mode, i, pred, epoch)
                logger.save_img_comparison(mode, i, batch_data, pred, epoch)
                logger.save_pred(mode, batch_data['img_name'][0], pred, batch_data['d'], epoch)

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    iter_train_loss = torch.tensor(iter_train_loss)
    perepoch_train_loss = iter_train_loss / len(loader)

    return avg, is_best, perepoch_train_loss

def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:   # eval模型
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate),
                  end='')
            checkpoint = torch.load(args.evaluate, map_location=device)
            #args = checkpoint['args']
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            is_eval = True
            print("Completed.")
        else:
            is_eval = True
            print("No model found at '{}'".format(args.evaluate))
            #return
    elif args.resume:  # 加载模型继续训练
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume),
                  end='')
            checkpoint = torch.load(args.resume, map_location=device)

            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    # 训练模型
    print("=> creating model and optimizer ... ", end='')
    model = None
    penet_accelerated = False
    if (args.network_model == 'e'):
        model = ENet(args).to(device)
    elif (is_eval == False):                        # 若网络为 penet 且 不评价模型（训练PENet）则不在前两种膨胀卷积使用penet加速训练
        if (args.dilation_rate == 1):
            model = PENet_C1_train(args).to(device)
        elif (args.dilation_rate == 2):
            model = PENet_C2_train(args).to(device)
        elif (args.dilation_rate == 4):
            model = PENet_C4(args).to(device)
            penet_accelerated = True
    else:                                           # 若网络为 penet 且 评价模型，则使用加速训练方法
        if (args.dilation_rate == 1):
            model = PENet_C1(args).to(device)
            penet_accelerated = True
        elif (args.dilation_rate == 2):
            model = PENet_C2(args).to(device)
            penet_accelerated = True
        elif (args.dilation_rate == 4):
            model = PENet_C4(args).to(device)
            penet_accelerated = True

    if (penet_accelerated == True):     # 如果加速训练，第3, 5, 7个encoder不需要计算梯度
        model.encoder3.requires_grad = False
        model.encoder5.requires_grad = False
        model.encoder7.requires_grad = False

    model_named_params = None
    model_bone_params = None
    model_new_params = None
    optimizer = None

    if checkpoint is not None:      # 如果加载了预训练模型
        #print(checkpoint.keys())
        if (args.freeze_backbone == True):      #　若为 True，即为训练 DA-CSPN++，加载的模型为ENet　# model.backbone = ENet
            model.backbone.load_state_dict(checkpoint['model'])
        else:                                   #　若为 False，则继续训练 ENet 或者 训练 PENet
            model.load_state_dict(checkpoint['model'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    logger = helper.logger(args)

    if checkpoint is not None:      # 如果加载了预训练模型
        logger.best_result = checkpoint['best_result']
        del checkpoint
    print("=> logger created.")

    test_dataset = None
    test_loader = None

    # 若只test，则只需输入rgb, depth，估计gt depth，测试完直接返回
    if (args.test):
        test_dataset = KittiDepth('test_completion', args)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)
        iterate("test_completion", args, test_loader, model, None, logger, 0)
        # 测试完直接返回
        return

    val_dataset = KittiDepth('val', args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    # 若为val，则需输入rgb, depth, gt depth，估计gt depth，测试完直接返回
    if is_eval == True:
        for p in model.parameters():
            p.requires_grad = False

        result, is_best, _ = iterate("val", args, val_loader, model, None, logger, args.start_epoch - 1)
        # 测试完直接返回
        return

    ################## 下面为Train部分：需输入rgb, depth, gt depth，估计gt depth ###########
    if (args.freeze_backbone == True):
        # 需冻结ENet的权重，仅训练 DA-CSPN++。model.backbone = ENet
        for p in model.backbone.parameters():
            p.requires_grad = False

        # named_parameters()：输出梯度传播的 网络层名字 和 参数的迭代器
        # 将需要进行梯度传播的 参数的迭代器 存入 list
        model_named_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
        # optimizer = RAdam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        # optimizer = torch.optim.AdamW(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    elif (args.network_model == 'pe'):
        # 训练完整的 PENet，model.backbone = ENet
        # 将ENet需要进行梯度传播的 参数的迭代器 存入 list
        model_bone_params = [
            p for _, p in model.backbone.named_parameters() if p.requires_grad
        ]
        # 将PENet需要进行梯度传播的 参数的迭代器 存入 list
        model_new_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        model_new_params = list(set(model_new_params) - set(model_bone_params))     # 只将PENet中除去ENet需要进行梯度传播的 参数的迭代器 存入 list
        optimizer = torch.optim.Adam([{'params': model_bone_params, 'lr': args.lr / 10}, {'params': model_new_params}], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
        # optimizer = RAdam([{'params': model_bone_params, 'lr': args.lr / 10}, {'params': model_new_params}], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        # optimizer = torch.optim.AdamW([{'params': model_bone_params, 'lr': args.lr / 10}, {'params': model_new_params}],
        #                              lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    else:
        # 训练 ENet，model = ENet
        model_named_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
        # optimizer = RAdam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        # optimizer = torch.optim.AdamW(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    print("completed.")

    model = torch.nn.DataParallel(model)        # 多张GPU并行计算

    #--------------------------------------------------------------------------------------
    #--------------------- Data loading code ----------------------------------------------
    #-------------------------------------------------------------------------------------
    print("=> creating data loaders ... ")
    if not is_eval:     # 如果不评价
        train_dataset = KittiDepth('train', args)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   sampler=None)
        print("\t==> train_loader size:{}".format(len(train_loader)))

    print("=> starting main loop ...")

    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))

        _, _ , perepoch_train_loss = iterate("train", args, train_loader, model, optimizer, logger, epoch)  # 一个epoch的训练

        # validation memory reset
        for p in model.parameters():        # 冻结参数，以进行验证测试
            p.requires_grad = False

        result, is_best, _ = iterate("val", args, val_loader, model, None, logger, epoch)  # 在验证集上做测试

        for p in model.parameters():        # 解冻参数
            p.requires_grad = True


        if (args.freeze_backbone == True):  #　若为 True，需训练 DA-CSPN++，冻结ENet的权重
            for p in model.module.backbone.parameters():
                p.requires_grad = False

        if (penet_accelerated == True):     # 若加速训练，则冻结encoder的部分层
            model.module.encoder3.requires_grad = False
            model.module.encoder5.requires_grad = False
            model.module.encoder7.requires_grad = False

        # 保存模型
        helper.save_checkpoint({ 'model': model.module.state_dict(),
                                 'best_result': logger.best_result,
                                 'optimizer' : optimizer.state_dict(),
                                 'epoch': epoch,
                                 'args' : args,},
                is_best, epoch, logger.output_directory)


if __name__ == '__main__':
    main()
