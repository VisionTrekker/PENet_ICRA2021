import math
import os, time
import shutil
import torch
import csv
import vis_utils
import numpy as np
from metrics import Result

fieldnames = [
    'epoch', 'rmse', 'photo', 'mae', 'irmse', 'imae', 'mse', 'absrel', 'lg10',
    'silog', 'squared_rel', 'delta1', 'delta2', 'delta3', 'data_time',
    'gpu_time'
]


class logger:
    def __init__(self, args, prepare=True):
        self.args = args
        output_directory = get_folder_name(args)
        self.output_directory = output_directory
        self.best_result = Result()
        self.best_result.set_to_worst()

        if not prepare:
            return
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        self.train_csv = os.path.join(output_directory, 'train.csv')
        self.val_csv = os.path.join(output_directory, 'val.csv')
        self.best_txt = os.path.join(output_directory, 'best.txt')

        # backup the source code
        if args.resume == '':
            print("=> creating source code backup ...")
            backup_directory = os.path.join(output_directory, "code_backup")
            self.backup_directory = backup_directory
            backup_source_code(backup_directory)
            # create new csv files with only header
            with open(self.train_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            with open(self.val_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            print("=> finished creating source code backup.")

    def conditional_print(self, split, i, epoch, lr, n_set, blk_avg_meter,
                          avg_meter):
        if (i + 1) % self.args.print_freq == 0:
            avg = avg_meter.average()
            blk_avg = blk_avg_meter.average()
            print('=> output: {}'.format(self.output_directory))
            print(
                '{split} Epoch: {0} [{1}/{2}]\tlr={lr} '
                't_Data={blk_avg.data_time:.3f}({average.data_time:.3f}) '
                't_GPU={blk_avg.gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                'RMSE={blk_avg.rmse:.2f}({average.rmse:.2f}) '
                'MAE={blk_avg.mae:.2f}({average.mae:.2f}) '
                'iRMSE={blk_avg.irmse:.2f}({average.irmse:.2f}) '
                'iMAE={blk_avg.imae:.2f}({average.imae:.2f})\n\t'
                'silog={blk_avg.silog:.2f}({average.silog:.2f}) '
                'squared_rel={blk_avg.squared_rel:.2f}({average.squared_rel:.2f}) '
                'Delta1={blk_avg.delta1:.3f}({average.delta1:.3f}) '
                'REL={blk_avg.absrel:.3f}({average.absrel:.3f})\n\t'
                'Lg10={blk_avg.lg10:.3f}({average.lg10:.3f}) '
                'Photometric={blk_avg.photometric:.3f}({average.photometric:.3f}) '
                .format(epoch,
                        i + 1,
                        n_set,
                        lr=lr,
                        blk_avg=blk_avg,
                        average=avg,
                        split=split.capitalize()))
            blk_avg_meter.reset(False)

    def conditional_save_info(self, split, average_meter, epoch):
        avg = average_meter.average()
        if split == "train":
            csvfile_name = self.train_csv
        elif split == "val":
            csvfile_name = self.val_csv
        elif split == "eval":
            eval_filename = os.path.join(self.output_directory, 'eval.txt')
            self.save_single_txt(eval_filename, avg, epoch)
            return avg
        elif "test" in split:
            return avg
        else:
            raise ValueError("wrong split provided to logger")
        with open(csvfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch,
                'rmse': avg.rmse,
                'photo': avg.photometric,
                'mae': avg.mae,
                'irmse': avg.irmse,
                'imae': avg.imae,
                'mse': avg.mse,
                'silog': avg.silog,
                'squared_rel': avg.squared_rel,
                'absrel': avg.absrel,
                'lg10': avg.lg10,
                'delta1': avg.delta1,
                'delta2': avg.delta2,
                'delta3': avg.delta3,
                'gpu_time': avg.gpu_time,
                'data_time': avg.data_time
            })
        return avg

    def save_single_txt(self, filename, result, epoch):
        with open(filename, 'w') as txtfile:
            txtfile.write(
                ("rank_metric={}\n" + "epoch={}\n" + "rmse={:.3f}\n" +
                 "mae={:.3f}\n" + "silog={:.3f}\n" + "squared_rel={:.3f}\n" +
                 "irmse={:.3f}\n" + "imae={:.3f}\n" + "mse={:.3f}\n" +
                 "absrel={:.3f}\n" + "lg10={:.3f}\n" + "delta1={:.3f}\n" +
                 "t_gpu={:.4f}").format(self.args.rank_metric, epoch,
                                        result.rmse, result.mae, result.silog,
                                        result.squared_rel, result.irmse,
                                        result.imae, result.mse, result.absrel,
                                        result.lg10, result.delta1,
                                        result.gpu_time))

    def save_best_txt(self, result, epoch):
        self.save_single_txt(self.best_txt, result, epoch)

    def _get_img_comparison_name(self, mode, epoch, is_best=False):
        if mode == 'eval':
            return self.output_directory + '/comparison_eval.png'
        if mode == 'val':
            if is_best:
                return self.output_directory + '/comparison_best.png'
            else:
                return self.output_directory + '/comparison_' + str(epoch) + '.png'

    def conditional_save_img_comparison(self, mode, i, ele, pred, epoch, predrgb=None, predg=None, extra=None, extra2=None, extrargb=None):
        if mode == 'val' or mode == 'eval':
            skip = 100
            if i == 0:
                self.img_merge = vis_utils.merge_into_row(ele, pred, predrgb, predg, extra, extra2, extrargb)
            elif i % skip == 0 and i < 8 * skip:
                row = vis_utils.merge_into_row(ele, pred, predrgb, predg, extra, extra2, extrargb)
                self.img_merge = vis_utils.add_row(self.img_merge, row)
            elif i == 8 * skip: # 共保存 8 张
                filename = self._get_img_comparison_name(mode, epoch)
                vis_utils.save_image(self.img_merge, filename)

    def save_img_comparison(self, mode, i, ele, pred, epoch, predrgb=None, predg=None, extra=None, extra2=None, extrargb=None):
        """
            mode: "val"
            i: 图片id
            ele: batch_data
            pred: 预测的深度图
            epoch: 第多少个epoch
        """
        # 共保存 8 张
        if mode == 'val' or mode == 'eval':
            skip = 20      # 每 100 张取出 1 张
            if i == 0:
                self.img_merge = vis_utils.merge_into_row(ele, pred, predrgb, predg, extra, extra2, extrargb)
            elif i % skip == 0 and i < 8 * skip:
                row = vis_utils.merge_into_row(ele, pred, predrgb, predg, extra, extra2, extrargb)
                self.img_merge = vis_utils.add_row(self.img_merge, row)
            elif i == 8 * skip:
                filename = self.args.data_folder_save + '/comparison_' + str(epoch) + '.png'
                vis_utils.save_image(self.img_merge, filename)

    def save_img_comparison_as_best(self, mode, epoch):
        if mode == 'val':
            filename = self._get_img_comparison_name(mode, epoch, is_best=True)
            vis_utils.save_image(self.img_merge, filename)

    def get_ranking_error(self, result):
        return getattr(result, self.args.rank_metric)       # 返回 result 中 rmse 的值

    def rank_conditional_save_best(self, mode, result, epoch):      # 传入参数为 (mode, avg, epoch)
        error = self.get_ranking_error(result)
        best_error = self.get_ranking_error(self.best_result)
        is_best = error < best_error
        if is_best and mode == "val":
            self.old_best_result = self.best_result
            self.best_result = result
            self.save_best_txt(result, epoch)
        return is_best

    def conditional_save_pred(self, mode, i, pred, epoch):
        if ("test" in mode or mode == "eval") and self.args.save_pred:

            # save images for visualization/ testing
            image_folder = os.path.join(self.output_directory,
                                        mode + "_output")
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            img = torch.squeeze(pred.data.cpu()).numpy()
            filename = os.path.join(image_folder, '{0:010d}.png'.format(i))
            vis_utils.save_depth_as_uint16png(img, filename)

    def save_pred(self, mode, img_name, pred, sparse, epoch):
        if ("test" in mode or mode == "val"):
            # save images for visualization/ testing
            image_folder = os.path.join(self.args.data_folder_save, mode + "_output")
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            img = torch.squeeze(pred.data.cpu()).numpy()
            np.save(os.path.join(image_folder, '{}.npy'.format(img_name)), img) # 保存为npy格式
            filename = os.path.join(image_folder, '{}.png'.format(img_name))
            vis_utils.save_depth_as_uint16png(img, filename)
            # "/media/liuzhi/b4608ade-d2e0-430d-a40b-f29a8b22cb8c/3DGS_code/PENet_3dgs/results"
            image_folder = os.path.join(self.args.data_folder_save, "sparse_output")
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            sparse = torch.squeeze(sparse.data.cpu()).numpy()
            np.save(os.path.join(image_folder, '{}.npy'.format(img_name)), sparse)  # 保存为npy格式

    def conditional_summarize(self, mode, avg, is_best):
        print("\n*\nSummary of ", mode, "round")
        print(''
              'RMSE={average.rmse:.3f}\n'
              'MAE={average.mae:.3f}\n'
              'Photo={average.photometric:.3f}\n'
              'iRMSE={average.irmse:.3f}\n'
              'iMAE={average.imae:.3f}\n'
              'squared_rel={average.squared_rel}\n'
              'silog={average.silog}\n'
              'Delta1={average.delta1:.3f}\n'
              'REL={average.absrel:.3f}\n'
              'Lg10={average.lg10:.3f}\n'
              't_GPU={time:.3f}'.format(average=avg, time=avg.gpu_time))
        if is_best and mode == "val":
            print("New best model by %s (was %.3f)" %
                  (self.args.rank_metric,
                   self.get_ranking_error(self.old_best_result)))
        elif mode == "val":
            print("(best %s is %.3f)" %
                  (self.args.rank_metric,
                   self.get_ranking_error(self.best_result)))
        print("*\n")


ignore_hidden = shutil.ignore_patterns(".", "..", ".git*", "*pycache*",
                                       "*build", "*.fuse*", "*_drive_*")


def backup_source_code(backup_directory):
    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)
    shutil.copytree('.', backup_directory, ignore=ignore_hidden)


def adjust_learning_rate_new(lr_init, optimizer, actual_epoch, epoch, args):    # epoch: [n, 50] actual_epoch: [0, 50-n]
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    #lr = lr_init * (0.5**(epoch // 5))
    #'''
    lr = lr_init
    if args.network_model == 'pe':  # 完整训练 或 中断后继续训练PENet，或 训练DA-CSPN++
        if args.freeze_backbone == False:   # 完整训练 或 中断后继续训练PENet
            if args.resume:                 # 中断后继续训练PENet
                if (epoch >= 10):
                    lr = lr_init * 0.5
                if (epoch >= 20):
                    lr = lr_init * 0.1
                if (epoch >= 30):
                    lr = lr_init * 0.01
                if (epoch >= 40):
                    lr = lr_init * 0.0005
                if (epoch >= 50):
                    lr = lr_init * 0.00001
            else:                           # 完整训练PENet
                if (actual_epoch >= 10):
                    lr = lr_init * 0.5
                if (actual_epoch >= 20):
                    lr = lr_init * 0.1
                if (actual_epoch >= 30):
                    lr = lr_init * 0.01
                if (actual_epoch >= 40):
                    lr = lr_init * 0.0005
                if (actual_epoch >= 50):
                    lr = lr_init * 0.00001
        else:
            if args.resume:             # 训练DA-CSPN++   # 2个epoch
                lr = lr_init

    else:         # 完整训练 或 中断后继续训练ENet
        if args.resume:                 # 中断后继续训练ENet
            if (epoch >= 10):
                lr = lr_init * 0.5
            if (epoch >= 15):
                lr = lr_init * 0.2
            if (epoch >= 25):
                lr = lr_init * 0.1
        else:                           # 完整训练ENet
            if (actual_epoch >= 10):
                lr = lr_init * 0.5
            if (actual_epoch >= 15):
                lr = lr_init * 0.2
            if (actual_epoch >= 25):
                lr = lr_init * 0.1
    #'''

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_old(lr_init, optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    #lr = lr_init * (0.5**(epoch // 5))
    #'''
    lr = lr_init
    if (args.network_model == 'pe' and args.freeze_backbone == False):
        if (epoch >= 10):
            lr = lr_init * 0.5
        if (epoch >= 20):
            lr = lr_init * 0.1
        if (epoch >= 30):
            lr = lr_init * 0.01
        if (epoch >= 40):
            lr = lr_init * 0.0005
        if (epoch >= 50):
            lr = lr_init * 0.00001
    else:
        if (epoch >= 10):
            lr = lr_init * 0.5
        if (epoch >= 15):
            lr = lr_init * 0.1
        if (epoch >= 25):
            lr = lr_init * 0.01
    #'''

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    print("saved: {}".format(checkpoint_filename))
    # torch 1.4版本以下加载高版本torch保存的模型
    # torch.save(state, checkpoint_filename, _use_new_zipfile_serialization=False)

    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(
            output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


def get_folder_name(args):
    current_time = time.strftime('%Y-%m-%d@%H-%M')
    return os.path.join(args.result,
        'input={}.criterion={}.lr={}.he={}.w={}.bs={}.wd={}.jitter={}.time={}'.
        format(args.input, args.criterion, \
            args.lr, args.random_crop_height, args.random_crop_width, args.batch_size, args.weight_decay, \
            args.jitter, current_time
            ))


avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2).cuda()


def multiscale(img):
    img1 = avgpool(img)
    img2 = avgpool(img1)
    img3 = avgpool(img2)
    img4 = avgpool(img3)
    img5 = avgpool(img4)
    return img5, img4, img3, img2, img1
