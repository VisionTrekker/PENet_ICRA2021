import torch
import math
import numpy as np

lg_e_10 = math.log(10)


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / lg_e_10


class Result(object):
    def __init__(self):
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0             # 绝对相对误差
        self.squared_rel = 0        # 平方相对误差
        self.lg10 = 0               # log MAE,                      clg10 = 1/n * sum(|log(pred) - log(gt)|)
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0

        self.silog = 0  # Scale invariant logarithmic error [log(m)*100],　d = log(pred) - log(gt), silog = 1/n * sum(d_i ** 2) + 1/n^2 * ( sum(d_i) ) ** 2
        self.photometric = 0

    def set_to_worst(self):
        self.irmse = np.inf     # np.inf 为一个无限大的正整数
        self.imae = np.inf
        self.mse = np.inf
        self.rmse = np.inf
        self.mae = np.inf
        self.absrel = np.inf
        self.squared_rel = np.inf
        self.lg10 = np.inf
        self.silog = np.inf
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10, delta1, delta2, delta3, gpu_time, data_time, silog, photometric=0):
        self.irmse = irmse
        self.imae = imae
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.absrel = absrel
        self.squared_rel = squared_rel
        self.lg10 = lg10
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.data_time = data_time
        self.gpu_time = gpu_time

        self.silog = silog
        self.photometric = photometric

    def evaluate(self, output, target, photometric=0):
        valid_mask = target > 0.1

        # 将单位从　m　换算到　mm
        output_mm = 1e3 * output[valid_mask]        # 变成了一维数组
        target_mm = 1e3 * target[valid_mask]

        abs_diff = (output_mm - target_mm).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float(( log10(output_mm) - log10(target_mm) ).abs().mean())     # log MAE
        self.absrel = float((abs_diff / target_mm).mean())                          # 绝对相对误差
        self.squared_rel = float(((abs_diff / target_mm)**2).mean())

        maxRatio = torch.max(output_mm / target_mm, target_mm / output_mm)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25**2).float().mean())
        self.delta3 = float((maxRatio < 1.25**3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        # silog 使用　m 为单位                    d = log(pred) - log(gt),            silog = 1/n * sum(d_i ** 2) + [1/n * ( sum(d_i) )] ** 2
        err_log = torch.log(target[valid_mask]) - torch.log(output[valid_mask])
        normalized_squared_log = (err_log**2).mean()
        log_mean = err_log.mean()
        self.silog = math.sqrt(normalized_squared_log - log_mean * log_mean) * 100

        # 将单位从　m　换算到　km
        inv_output_km = (1e-3 * output[valid_mask])**(-1)
        inv_target_km = (1e-3 * target[valid_mask])**(-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

        self.photometric = float(photometric)


class AverageMeter(object):
    def __init__(self):
        self.reset(time_stable=True)

    def reset(self, time_stable):
        self.count = 0.0
        self.sum_irmse = 0
        self.sum_imae = 0
        self.sum_mse = 0
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_absrel = 0
        self.sum_squared_rel = 0
        self.sum_lg10 = 0
        self.sum_delta1 = 0
        self.sum_delta2 = 0
        self.sum_delta3 = 0
        self.sum_data_time = 0
        self.sum_gpu_time = 0
        self.sum_photometric = 0
        self.sum_silog = 0
        self.time_stable = time_stable
        self.time_stable_counter_init = 10
        self.time_stable_counter = self.time_stable_counter_init

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_squared_rel += n * result.squared_rel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        if self.time_stable == True and self.time_stable_counter > 0:
            self.time_stable_counter = self.time_stable_counter - 1
        else:
            self.sum_gpu_time += n * gpu_time
        self.sum_silog += n * result.silog
        self.sum_photometric += n * result.photometric

    def average(self):
        avg = Result()
        if self.time_stable == True:
            if self.count > 0 and self.count - self.time_stable_counter_init > 0:
                avg.update(
                    self.sum_irmse / self.count, self.sum_imae / self.count,
                    self.sum_mse / self.count, self.sum_rmse / self.count,
                    self.sum_mae / self.count, self.sum_absrel / self.count,
                    self.sum_squared_rel / self.count, self.sum_lg10 / self.count,
                    self.sum_delta1 / self.count, self.sum_delta2 / self.count,
                    self.sum_delta3 / self.count, self.sum_gpu_time / (self.count - self.time_stable_counter_init),
                    self.sum_data_time / self.count, self.sum_silog / self.count,
                    self.sum_photometric / self.count)
            elif self.count > 0:
                avg.update(
                    self.sum_irmse / self.count, self.sum_imae / self.count,
                    self.sum_mse / self.count, self.sum_rmse / self.count,
                    self.sum_mae / self.count, self.sum_absrel / self.count,
                    self.sum_squared_rel / self.count, self.sum_lg10 / self.count,
                    self.sum_delta1 / self.count, self.sum_delta2 / self.count,
                    self.sum_delta3 / self.count, 0,
                    self.sum_data_time / self.count, self.sum_silog / self.count,
                    self.sum_photometric / self.count)
        elif self.count > 0:
            avg.update(
                self.sum_irmse / self.count, self.sum_imae / self.count,
                self.sum_mse / self.count, self.sum_rmse / self.count,
                self.sum_mae / self.count, self.sum_absrel / self.count,
                self.sum_squared_rel / self.count, self.sum_lg10 / self.count,
                self.sum_delta1 / self.count, self.sum_delta2 / self.count,
                self.sum_delta3 / self.count, self.sum_gpu_time / self.count,
                self.sum_data_time / self.count, self.sum_silog / self.count,
                self.sum_photometric / self.count)
        return avg
