import os
import os.path
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
from dataloaders import transforms
import CoordConv
import torch.nn.functional as F
import struct
import collections

input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def load_calib_(path):
    # cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    # cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    # cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    # cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    # for idx, key in enumerate(cam_extrinsics):
    #     extr = cam_extrinsics[key]
    #     intr = cam_intrinsics[extr.camera_id]
    #
    # if intr.model == "PINHOLE":
    #     K = np.array([[intr.params[0], 0, intr.params[2]],
    #               [0, intr.params[1], intr.params[3]],
    #               [0, 0, 1]]).astype(np.float32)
    # elif intr.model == "SIMPLE_RADIAL" or "SIMPLE_PINHOLE":
    #     K = np.array([[intr.params[0], 0, intr.params[1]],
    #                   [0, intr.params[0], intr.params[2]],
    #                   [0, 0, 1]]).astype(np.float32)

    K = np.array([[554.256, 0, 960],
                  [0, 554.256, 540],
                  [0, 0, 1]]).astype(np.float32)
    # Kitti使用了中心裁剪，而调整了K的值
    # 1080 - 640 / 2
    K[1, 2] = K[1, 2] - 220

    return K

def load_calib():
    """
        Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25] # P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:] # ['7.215377e+02', '0.000000e+00', '6.095593e+02', '4.485728e+01', '0.000000e+00', '7.215377e+02', '1.728540e+02', '2.163791e-01', '0.000000e+00', '0.000000e+00', '1.000000e+00', '2.745884e-03\n']
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                      (3, 4)).astype(np.float32)        # [[7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01]
                                                        #   [0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01]
                                                        #   [0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03]]
    K = Proj[:3, :3]    # camera matrix
                        # [[721.5377   0.     609.5593]
                        #  [  0.     721.5377 172.854 ]
                        #  [  0.       0.       1.    ]]

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    # K[0, 2] = K[0, 2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    # K[1, 2] = K[1, 2] - 11.5  # from heigth = 375 to 352, with a 11.5-pixel cut on both sides
    K[0, 2] = K[0, 2] - 13;     # [[721.5377   0.     596.5593]
                                #  [  0.     721.5377 172.854 ]
                                #  [  0.       0.       1.    ]]

    K[1, 2] = K[1, 2] - 11.5;   # [[721.5377   0.     596.5593]
                                #  [  0.     721.5377 161.354 ]
                                #  [  0.       0.       1.    ]]
    return K


def get_paths_and_transform(split, args):
    assert (args.use_d or args.use_rgb or args.use_g), 'no proper input selected'

    if split == "train":
        transform = train_transform
        # transform = val_transform
        glob_d = os.path.join(
            args.data_folder,
            'mvs/stereo/depth_maps/*.png.geometric.bin'
        )
        glob_gt = None
        def get_gt_depth_paths(p):
            idx = p.split('/')[-1].split('.')[0]
            idx_name = str(idx) + '.npy'
            idx_path = os.path.join(args.data_folder, 'depth', idx_name)
            return idx_path

        def get_rgb_paths(p):
            idx = p.split('/')[-1].split('.')[0]
            idx_name = str(idx) + '.png'
            idx_path = os.path.join(args.data_folder, 'images', idx_name)
            return idx_path

    elif split == "val":
        if args.val == "full":
            transform = val_transform

            glob_d = os.path.join(
                args.data_folder,
                'mvs/stereo/depth_maps/*.png.geometric.bin'
            )
            glob_gt = None
            def get_gt_depth_paths(p):
                idx = p.split('/')[-1].split('.')[0]
                idx_name = str(idx) + '.npy'
                idx_path = os.path.join(args.data_folder, 'depth', idx_name)
                return idx_path
            def get_rgb_paths(p):
                idx = p.split('/')[-1].split('.')[0]
                idx_name = str(idx) + '.png'
                idx_path = os.path.join(args.data_folder, 'images', idx_name)
                return idx_path

    elif split == "test_completion":
        transform = no_transform

        glob_d = os.path.join(
            args.data_folder,
            'mvs/stereo/depth_maps/*.png.geometric.bin'
        )
        glob_gt = None
        def get_rgb_paths(p):
            idx = p.split('/')[-1].split('.')[0]
            idx_name = str(idx) + '.png'
            idx_path = os.path.join(args.data_folder, 'images', idx_name)
            return idx_path
    else:
        raise ValueError("Unrecognized split " + str(split))


    if split == "train":
        paths_d = sorted(glob.glob(glob_d)) # 所有 depth .png路径
        paths_gt = [get_gt_depth_paths(p) for p in paths_d]   # 所有 gt depth groundtruth .png路径
        paths_rgb = [get_rgb_paths(p) for p in paths_d]    # 所有 rgb .png路径
    elif split == "val":
        paths_d = sorted(glob.glob(glob_d))
        paths_gt = [get_gt_depth_paths(p) for p in paths_d]
        paths_rgb = [get_rgb_paths(p) for p in paths_d]
    elif split == "test_completion":
        paths_d = sorted(glob.glob(glob_d))
        paths_gt = [None] * len(paths_d)
        paths_rgb = [get_rgb_paths(p) for p in paths_d]


    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0 and args.use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and args.use_rgb:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and args.use_g:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        print(len(paths_rgb), len(paths_d), len(paths_gt))
        # for i in range(999):
        #    print("#####")
        #    print(paths_rgb[i])
        #    print(paths_d[i])
        #    print(paths_gt[i])
        # raise (RuntimeError("Produced different sizes for datasets"))
    print(len(paths_rgb), len(paths_d), len(paths_gt))

    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths, transform


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')
    if rgb_png.shape[2] == 4:
        rgb_png = rgb_png[:, :, :3]# in the range [0,255]
    img_file.close()
    return rgb_png


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def resize_torch(width, height, width_, height_, pre_depth_torch):
    pad_height = height - height_
    pad_width = width - width_

    # 使用 pad 函数对 pre_depth_torch 进行填充
    pre_depth_torch_padded = F.pad(pre_depth_torch, (0, pad_width, 0, pad_height))
    return pre_depth_torch_padded

def colmap_depth_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    try:
        pre_depth = read_array(filename)
    except:
        print(f"Bad File: {filename}")
        return None

    # print("pre_depth: max={}, min={}, path={}".format(np.max(pre_depth), np.min(pre_depth), filename))
    target_height, target_width = 1080, 1920
    depth_height, depth_width = pre_depth.shape[:2]
    pad_height = target_height - depth_height
    pad_width = target_width - depth_width
    pre_depth_ = np.pad(pre_depth, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    pre_depth_ = np.clip(pre_depth_, 0, 1000)
    pre_depth_ = np.expand_dims(pre_depth_, -1)
    # print("pre_depth_after: max={}, min={}, path={}".format(np.max(pre_depth_), np.min(pre_depth_), filename))
    return pre_depth_

def gt_depth_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    try:
        gt_depth = np.load(filename)
    except:
        print(f"Bad File: {filename}")
        return None
    # print("gt_depth: max={}, min={}, path={}".format(np.max(gt_depth), np.min(gt_depth), filename))
    scale_gt2colmap = 0.07191530157289869
    gt_depth_ = gt_depth * scale_gt2colmap
    gt_depth_ = np.expand_dims(gt_depth_, -1) # H W 1
    # print("gt_depth_after: max={}, min={}, path={}".format(np.max(gt_depth_), np.min(gt_depth_), filename))

    return gt_depth_

def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)    # 返回一个数组（与深度图的大小一样）, 其值为 0, 1 (出现 1 的概率 = prob_keep)
    depth *= mask
    return depth

def train_transform(rgb, sparse, target, position, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling # 均匀分布
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    oheight = args.val_h    # 352
    owidth = args.val_w     # 1216

    do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip
                                                # True or False

    transforms_list = [
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.CenterCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ]

    # if small_training == True:
    # transforms_list.append(transforms.RandomCrop((rheight, rwidth)))

    transform_geometric = transforms.Compose(transforms_list)

    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)                             # 0.9 <= brightness <= 1.1  亮度
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)      # 0.9 <= contrast <= 1.1    对比度
        saturation = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)                             # 0.9 <= saturation <= 1.1  饱和度
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)
    # sparse = drop_depth_measurements(sparse, 0.9)

    if position is not None:
        bottom_crop_only = transforms.Compose([transforms.BottomCrop((oheight, owidth))])
        position = bottom_crop_only(position)

    # random crop
    #if small_training == True:
    if args.not_random_crop == False:
        h = oheight     # 352
        w = owidth      # 1216
        rheight = args.random_crop_height   # 320
        rwidth = args.random_crop_width     # 1216
        # randomlize
        i = np.random.randint(0, h - rheight + 1)       # [0, 33)
        j = np.random.randint(0, w - rwidth + 1)        # [0, 1) => 0

        if rgb is not None:
            if rgb.ndim == 3:
                rgb = rgb[i:i + rheight, j:j + rwidth, :]
            elif rgb.ndim == 2:
                rgb = rgb[i:i + rheight, j:j + rwidth]

        if sparse is not None:
            if sparse.ndim == 3:
                sparse = sparse[i:i + rheight, j:j + rwidth, :]
            elif sparse.ndim == 2:
                sparse = sparse[i:i + rheight, j:j + rwidth]

        if target is not None:
            if target.ndim == 3:
                target = target[i:i + rheight, j:j + rwidth, :]
            elif target.ndim == 2:
                target = target[i:i + rheight, j:j + rwidth]

        if position is not None:
            if position.ndim == 3:
                position = position[i:i + rheight, j:j + rwidth, :]
            elif position.ndim == 2:
                position = position[i:i + rheight, j:j + rwidth]

    return rgb, sparse, target, position

def val_transform(rgb, sparse, target, position, args):
    oheight = args.val_h
    owidth = args.val_w

    transform = transforms.Compose([
        transforms.CenterCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    if position is not None:
        position = transform(position)

    return rgb, sparse, target, position


def no_transform(rgb, sparse, target, position, args):
    return rgb, sparse, target, position


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img_gray = np.array(Image.fromarray(rgb).convert('L'))
        img_gray = np.expand_dims(img_gray, -1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img_gray


def get_rgb_near(path, args):
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, _ = os.path.split(filename)
        new_filename = os.path.join(head, '%010d.png' % new_id)
        return new_filename

    head, number = extract_frame_id(path)
    count = 0
    max_frame_diff = 3
    candidates = [
        i - max_frame_diff for i in range(max_frame_diff * 2 + 1)
        if i - max_frame_diff != 0
    ]
    while True:
        random_offset = choice(candidates)
        path_near = get_nearby_filename(path, number + random_offset)
        if os.path.exists(path_near):
            break
        assert count < 20, "cannot find a nearby frame in 20 trials for {}".format(path_near)

    return rgb_read(path_near)


class KittiDepth(data.Dataset):
    """
        A data loader for the Kitti dataset
    """

    def __init__(self, split, args):
        self.args = args
        self.split = split  # 'train', 'val'
        paths, transform = get_paths_and_transform(split, args) # 获取训练集或验证集的 数据集图像路径 和对应的transform
        self.paths = paths          # is a dict, paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
        self.transform = transform  # train_transform or val_transform or no_transform
        self.K = load_calib_(None)
        self.threshold_translation = 0.1

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index]) if \
            (self.paths['rgb'][index] is not None and (self.args.use_rgb or self.args.use_g)) else None
        sparse = colmap_depth_read(self.paths['d'][index]) if \
            (self.paths['d'][index] is not None and self.args.use_d) else None
        target = gt_depth_read(self.paths['gt'][index]) if \
            self.paths['gt'][index] is not None else None
        img_name = self.paths['rgb'][index].split('/')[-1].split('.')[0]
        return rgb, sparse, target, img_name

    def __getitem__(self, index):
        rgb, sparse, target, img_name = self.__getraw__(index)
        position = CoordConv.AddCoordsNp(self.args.val_h, self.args.val_w)
        position = position.call()  # (val_h, val_w, 2)
        rgb, sparse, target, position = self.transform(rgb, sparse, target, position, self.args)        # train_transform:  (320, 1216, *)
                                                                                                        # val_transform:    (352, 1216, *)
                                                                                                        # no_transform:     (352, 1216, *)

        rgb, gray = handle_gray(rgb, self.args)
        # candidates = {"rgb": rgb, "d": sparse, "gt": target, \
        #              "g": gray, "r_mat": r_mat, "t_vec": t_vec, "rgb_near": rgb_near}
        candidates = {"rgb": rgb, "d": sparse, "gt": target, \
                      "g": gray, 'position': position, 'K': self.K}

        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }
        items['img_name'] = img_name

        return items

    def __len__(self):
        return len(self.paths['gt'])
