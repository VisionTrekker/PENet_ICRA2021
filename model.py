import math
import warnings
from torch import Tensor
import torch
import numpy as np
from basic import *

class ENet(nn.Module):
    def __init__(self, args):
        super(ENet, self).__init__()
        self.args = args
        self.geofeature = None
        self.geoplanes = 3
        if self.args.convolutional_layer_encoding == "xyz":
            self.geofeature = GeometryFeature()
        elif self.args.convolutional_layer_encoding == "std":
            self.geoplanes = 0
        elif self.args.convolutional_layer_encoding == "uv":
            self.geoplanes = 2
        elif self.args.convolutional_layer_encoding == "z":
            self.geoplanes = 1

        # rgb encoder                                                                                           # input: [b, 4, 320, 1216]
        self.rgb_conv_init = convbnrelu(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)     # [b, 32, 320, 1216]

        self.rgb_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer3 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer4 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer5 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer6 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer7 = BasicBlockGeo(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer8 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer9 = BasicBlockGeo(inplanes=512, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer10 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)

        # rgb decoder
        self.rgb_decoder_layer8 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer6 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer4 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer2 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_output = deconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)

        # depth encoder
        self.depth_conv_init = convbnrelu(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.depth_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.depth_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.depth_layer3 = BasicBlockGeo(inplanes=128, planes=128, stride=2, geoplanes=self.geoplanes)
        self.depth_layer4 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.depth_layer5 = BasicBlockGeo(inplanes=256, planes=256, stride=2, geoplanes=self.geoplanes)
        self.depth_layer6 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.depth_layer7 = BasicBlockGeo(inplanes=512, planes=512, stride=2, geoplanes=self.geoplanes)
        self.depth_layer8 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.depth_layer9 = BasicBlockGeo(inplanes=1024, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.depth_layer10 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)

        # depth decoder
        self.decoder_layer1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)

        self.decoder_layer6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

        weights_init(self)

    def forward(self, input):
        #independent input
        rgb = input['rgb']              # [b, 3, 320, 1216]
        d = input['d']                  # [b, 1, 320, 1216]
        position = input['position']    # [b, 2, 320, 1216]
        K = input['K']                  # [b, 3, 3]

        # 获取 原depth map及其下采样 中各点的 -位置坐标-
        unorm = position[:, 0:1, :, :]  # 取出 x 坐标   [b, 1, 320, 1216]
        vnorm = position[:, 1:2, :, :]  # 取出 y 坐标   [b, 1, 320, 1216]

        vnorm_s2 = self.pooling(vnorm)          # [b, 1, 160, 608]
        vnorm_s3 = self.pooling(vnorm_s2)       # [b, 1, 80, 304]
        vnorm_s4 = self.pooling(vnorm_s3)       # [b, 1, 40, 152]
        vnorm_s5 = self.pooling(vnorm_s4)       # [b, 1, 20, 76]
        vnorm_s6 = self.pooling(vnorm_s5)       # [b, 1, 10, 38]

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)

        # 获取原depth map及其下采样map，以及它们相对应的valid mask
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))     # torch.full_like(d, 1.0) 生成一个大小与 d [b, 1, 320, 1216] 相同，　值为 1 的tensor
                                                                                            # torch.where(condition, a, b) 按照条件condition合并两个tensor，如果该位置满足条件，选择a的值，否则选择b的值作为输出
        d_s2, vm_s2 = self.sparsepooling(d, valid_mask)     # [b, 1, 160, 608]  d_s2: maxpool depth        vm_s2: maxpool valid_mask
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)       # [b, 1, 80, 304]
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)       # [b, 1, 40, 152]
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)       # [b, 1, 20, 76]
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)       # [b, 1, 10, 38]

        # 获取照相机内参
        # fy
        f352 = K[:, 1, 1]               # [b]
        f352 = f352.unsqueeze(1)        # [b, 1]
        f352 = f352.unsqueeze(2)        # [b, 1, 1]
        f352 = f352.unsqueeze(3)        # [b, 1, 1, 1]
        # cy
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        # fx
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        # cx
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        # 获取各层depth map的x, y, z层　或　u, v 层　或　z 层
        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
        geo_s6 = None

        # if self.args.convolutional_layer_encoding == "xyz":
        #     geo_s1 = self.geofeature(d, vnorm, unorm, 352, 1216, c352, c1216, f352, f1216)                      # [b, 3, 320, 1216]
        #     geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, 352 / 2, 1216 / 2, c352, c1216, f352, f1216)     # [b, 3, 160, 608]
        #     geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, 352 / 4, 1216 / 4, c352, c1216, f352, f1216)     # [b, 3, 80, 304]
        #     geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, 352 / 8, 1216 / 8, c352, c1216, f352, f1216)     # [b, 3, 40, 152]
        #     geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, 352 / 16, 1216 / 16, c352, c1216, f352, f1216)   # [b, 3, 20, 76]
        #     geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, 352 / 32, 1216 / 32, c352, c1216, f352, f1216)   # [b, 3, 10, 38]
        if self.args.convolutional_layer_encoding == "xyz":
            geo_s1 = self.geofeature(d, vnorm, unorm, 1080, 1920, c352, c1216, f352, f1216)                      # [b, 3, 320, 1216]
            geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, 1080 / 2, 1920 / 2, c352, c1216, f352, f1216)     # [b, 3, 160, 608]
            geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, 1080 / 4, 1920 / 4, c352, c1216, f352, f1216)     # [b, 3, 80, 304]
            geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, 1080 / 8, 1920 / 8, c352, c1216, f352, f1216)     # [b, 3, 40, 152]
            geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, 1080 / 16, 1920 / 16, c352, c1216, f352, f1216)   # [b, 3, 20, 76]
            geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, 1080 / 32, 1920 / 32, c352, c1216, f352, f1216)   # [b, 3, 10, 38]
        elif self.args.convolutional_layer_encoding == "uv":
            geo_s1 = torch.cat((vnorm, unorm), dim=1)
            geo_s2 = torch.cat((vnorm_s2, unorm_s2), dim=1)
            geo_s3 = torch.cat((vnorm_s3, unorm_s3), dim=1)
            geo_s4 = torch.cat((vnorm_s4, unorm_s4), dim=1)
            geo_s5 = torch.cat((vnorm_s5, unorm_s5), dim=1)
            geo_s6 = torch.cat((vnorm_s6, unorm_s6), dim=1)
        elif self.args.convolutional_layer_encoding == "z":
            geo_s1 = d
            geo_s2 = d_s2
            geo_s3 = d_s3
            geo_s4 = d_s4
            geo_s5 = d_s5
            geo_s6 = d_s6

        # -----------------------------------------------------------------------
        # rgb encoder                                                           # input: [b, 4, 320, 1216]
        rgb_feature = self.rgb_conv_init(torch.cat((rgb, d), dim=1))            # b 32 320 1216
        rgb_feature1 = self.rgb_encoder_layer1(rgb_feature, geo_s1, geo_s2)     # b 64 160 608
        rgb_feature2 = self.rgb_encoder_layer2(rgb_feature1, geo_s2, geo_s2)    # b 64 160 608
        rgb_feature3 = self.rgb_encoder_layer3(rgb_feature2, geo_s2, geo_s3)    # b 128 80 304
        rgb_feature4 = self.rgb_encoder_layer4(rgb_feature3, geo_s3, geo_s3)    # b 128 80 304
        rgb_feature5 = self.rgb_encoder_layer5(rgb_feature4, geo_s3, geo_s4)    # b 256 40 152
        rgb_feature6 = self.rgb_encoder_layer6(rgb_feature5, geo_s4, geo_s4)    # b 256 40 152
        rgb_feature7 = self.rgb_encoder_layer7(rgb_feature6, geo_s4, geo_s5)    # b 512 20 76
        rgb_feature8 = self.rgb_encoder_layer8(rgb_feature7, geo_s5, geo_s5)    # b 512 20 76
        rgb_feature9 = self.rgb_encoder_layer9(rgb_feature8, geo_s5, geo_s6)    # b 1024 10 38
        rgb_feature10 = self.rgb_encoder_layer10(rgb_feature9, geo_s6, geo_s6)  # b 1024 10 38
                                                                                # 输入为 [b, 3, 320, 1216] --> [b, 1024, 10, 38]
                                                                                # 输入为 [b, 3, 352, 1216] --> [b, 1024, 11, 38]
                                                                                # 输入为 [b, 3, 160, 576]  --> [b, 1024, 5, 18]
                                                                                # 输入为 [b, 3, 160, 608]  --> [b, 1024, 5, 19]

        # rgb decoder                                                         # input: [b, 1024, 10, 38]
        rgb_feature_decoder8 = self.rgb_decoder_layer8(rgb_feature10)
        rgb_feature8_plus = rgb_feature_decoder8 + rgb_feature8               # b 512 20 76

        rgb_feature_decoder6 = self.rgb_decoder_layer6(rgb_feature8_plus)
        rgb_feature6_plus = rgb_feature_decoder6 + rgb_feature6               # b 256 40 152

        rgb_feature_decoder4 = self.rgb_decoder_layer4(rgb_feature6_plus)
        rgb_feature4_plus = rgb_feature_decoder4 + rgb_feature4               # b 128 80 304

        rgb_feature_decoder2 = self.rgb_decoder_layer2(rgb_feature4_plus)
        rgb_feature2_plus = rgb_feature_decoder2 + rgb_feature2               # b 64 160 608

        rgb_feature_decoder0 = self.rgb_decoder_layer0(rgb_feature2_plus)
        rgb_feature0_plus = rgb_feature_decoder0 + rgb_feature                # b 32 320 1216

        rgb_output = self.rgb_decoder_output(rgb_feature0_plus)               # b 2 320 1216
        rgb_depth = rgb_output[:, 0:1, :, :]
        rgb_conf = rgb_output[:, 1:2, :, :]

        # -----------------------------------------------------------------------

        # depth encoder                                                                 # input: [b, 2, 320, 1216]
        sparsed_feature = self.depth_conv_init(torch.cat((d, rgb_depth), dim=1))        # b 32 320 1216

        sparsed_feature1 = self.depth_layer1(sparsed_feature, geo_s1, geo_s2)           # b 64 160 608
        sparsed_feature2 = self.depth_layer2(sparsed_feature1, geo_s2, geo_s2)          # b 64 160 608
        sparsed_feature2_plus = torch.cat([rgb_feature2_plus, sparsed_feature2], 1)     # b 128 160 608

        sparsed_feature3 = self.depth_layer3(sparsed_feature2_plus, geo_s2, geo_s3)     # b 128 80 304
        sparsed_feature4 = self.depth_layer4(sparsed_feature3, geo_s3, geo_s3)          # b 128 80 304
        sparsed_feature4_plus = torch.cat([rgb_feature4_plus, sparsed_feature4], 1)     # b 256 80 304

        sparsed_feature5 = self.depth_layer5(sparsed_feature4_plus, geo_s3, geo_s4)     # b 256 40 152
        sparsed_feature6 = self.depth_layer6(sparsed_feature5, geo_s4, geo_s4)          # b 256 40 152
        sparsed_feature6_plus = torch.cat([rgb_feature6_plus, sparsed_feature6], 1)     # b 512 40 152

        sparsed_feature7 = self.depth_layer7(sparsed_feature6_plus, geo_s4, geo_s5)     # b 512 20 76
        sparsed_feature8 = self.depth_layer8(sparsed_feature7, geo_s5, geo_s5)          # b 512 20 76
        sparsed_feature8_plus = torch.cat([rgb_feature8_plus, sparsed_feature8], 1)     # b 1024 20 76

        sparsed_feature9 = self.depth_layer9(sparsed_feature8_plus, geo_s5, geo_s6)     # b 1024 10 38
        sparsed_feature10 = self.depth_layer10(sparsed_feature9, geo_s6, geo_s6)        # b 1024 10 38

        # depth decoder
        fusion1 = rgb_feature10 + sparsed_feature10             # b 1024 10 38
        decoder_feature1 = self.decoder_layer1(fusion1)         # b 512 20 76

        fusion2 = sparsed_feature8 + decoder_feature1           # b 512 20 76
        decoder_feature2 = self.decoder_layer2(fusion2)         # b 256 40 152

        fusion3 = sparsed_feature6 + decoder_feature2           # b 256 40 152
        decoder_feature3 = self.decoder_layer3(fusion3)         # b 128 80 304

        fusion4 = sparsed_feature4 + decoder_feature3           # b 128 80 304
        decoder_feature4 = self.decoder_layer4(fusion4)         # b 64 160 608

        fusion5 = sparsed_feature2 + decoder_feature4           # b 64 160 608
        decoder_feature5 = self.decoder_layer5(fusion5)         # b 32 320 1216

        depth_output = self.decoder_layer6(decoder_feature5)    # b 2 320 1216

        d_depth, d_conf = torch.chunk(depth_output, 2, dim=1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf, d_conf), dim=1)), 2, dim=1)
        output = rgb_conf*rgb_depth + d_conf*d_depth

        # -----------------------------------------------------------------------------------------

        if(self.args.network_model == 'e'):
            return rgb_depth, d_depth, output
        elif(self.args.dilation_rate == 1):
            return torch.cat((rgb_feature0_plus, decoder_feature5),1), output
        elif (self.args.dilation_rate == 2):
            return torch.cat((rgb_feature0_plus, decoder_feature5), 1), torch.cat((rgb_feature2_plus, decoder_feature4),1), output
        elif (self.args.dilation_rate == 4):
            return torch.cat((rgb_feature0_plus, decoder_feature5), 1), torch.cat((rgb_feature2_plus, decoder_feature4),1),\
                   torch.cat((rgb_feature4_plus, decoder_feature3), 1), output

class PENet_C1(nn.Module):
    def __init__(self, args):
        super(PENet_C1, self).__init__()

        self.backbone = ENet(args)
        #self.backbone = Bone()
        self.mask_layer = convbn(64, 3)

        self.kernel_conf_layer = convbn(64, 3)
        self.iter_conf_layer = convbn(64, 12)
        self.iter_guide_layer3 = CSPNGenerateAccelerate(64, 3)
        self.iter_guide_layer5 = CSPNGenerateAccelerate(64, 5)
        self.iter_guide_layer7 = CSPNGenerateAccelerate(64, 7)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPNAccelerate(3)
        self.CSPN5 = CSPNAccelerate(5, padding=2)
        self.CSPN7 = CSPNAccelerate(7, padding=3)

        # CSPN new
        ks = 3
        encoder3 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder3[index] = 1
        self.encoder3 = nn.Parameter(encoder3, requires_grad=False)

        ks = 5
        encoder5 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder5[index] = 1
        self.encoder5 = nn.Parameter(encoder5, requires_grad=False)

        ks = 7
        encoder7 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder7[index] = 1
        self.encoder7 = nn.Parameter(encoder7, requires_grad=False)

        weights_init(self)

    def forward(self, input):
        #rgb = input['rgb']
        d = input['d']
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        feature, coarse_depth= self.backbone(input)

        mask = self.mask_layer(feature)
        mask = torch.sigmoid(mask)

        mask = mask*valid_mask
        mask3 = mask[:, 0:1, :, :]
        mask5 = mask[:, 1:2, :, :]
        mask7 = mask[:, 2:3, :, :]

        kernel_conf = self.kernel_conf_layer(feature)
        kernel_conf = self.softmax(kernel_conf)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        conf = self.iter_conf_layer(feature)
        conf3 = conf[:, 0:4, :, :]
        conf5 = conf[:, 4:8, :, :]
        conf7 = conf[:, 8:12, :, :]
        conf3 = self.softmax(conf3)
        conf5 = self.softmax(conf5)
        conf7 = self.softmax(conf7)

        guide3 = self.iter_guide_layer3(feature)
        guide5 = self.iter_guide_layer5(feature)
        guide7 = self.iter_guide_layer7(feature)

        #init
        depth = coarse_depth
        depth3 = depth
        depth5 = depth
        depth7 = depth

        d3_list = [i for i in range(4)]
        d5_list = [i for i in range(4)]
        d7_list = [i for i in range(4)]

        #prop
        guide3 = kernel_trans(guide3, self.encoder3)
        guide5 = kernel_trans(guide5, self.encoder5)
        guide7 = kernel_trans(guide7, self.encoder7)

        for i in range(12):
            depth3 = self.CSPN3(guide3, depth3, depth)
            depth3 = mask3*d + (1-mask3)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth)
            depth5 = mask5*d + (1-mask5)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth)
            depth7 = mask7*d + (1-mask7)*depth7

            if(i==2):
                d3_list[0] = depth3
                d5_list[0] = depth5
                d7_list[0] = depth7

            if(i==5):
                d3_list[1] = depth3
                d5_list[1] = depth5
                d7_list[1] = depth7

            if(i==8):
                d3_list[2] = depth3
                d5_list[2] = depth5
                d7_list[2] = depth7

            if(i==11):
                d3_list[3] = depth3
                d5_list[3] = depth5
                d7_list[3] = depth7

        refined_depth = \
        d3_list[0] * (kernel_conf3 * conf3[:, 0:1, :, :]) + \
        d3_list[1] * (kernel_conf3 * conf3[:, 1:2, :, :]) + \
        d3_list[2] * (kernel_conf3 * conf3[:, 2:3, :, :]) + \
        d3_list[3] * (kernel_conf3 * conf3[:, 3:4, :, :]) + \
        d5_list[0] * (kernel_conf5 * conf5[:, 0:1, :, :]) + \
        d5_list[1] * (kernel_conf5 * conf5[:, 1:2, :, :]) + \
        d5_list[2] * (kernel_conf5 * conf5[:, 2:3, :, :]) + \
        d5_list[3] * (kernel_conf5 * conf5[:, 3:4, :, :]) + \
        d7_list[0] * (kernel_conf7 * conf7[:, 0:1, :, :]) + \
        d7_list[1] * (kernel_conf7 * conf7[:, 1:2, :, :]) + \
        d7_list[2] * (kernel_conf7 * conf7[:, 2:3, :, :]) + \
        d7_list[3] * (kernel_conf7 * conf7[:, 3:4, :, :])

        return refined_depth

class PENet_C2(nn.Module):
    def __init__(self, args):
        super(PENet_C2, self).__init__()

        self.backbone = ENet(args)

        self.kernel_conf_layer = convbn(64, 3)
        self.mask_layer = convbn(64, 1)
        self.iter_guide_layer3 = CSPNGenerateAccelerate(64, 3)
        self.iter_guide_layer5 = CSPNGenerateAccelerate(64, 5)
        self.iter_guide_layer7 = CSPNGenerateAccelerate(64, 7)

        self.kernel_conf_layer_s2 = convbn(128, 3)
        self.mask_layer_s2 = convbn(128, 1)
        self.iter_guide_layer3_s2 = CSPNGenerateAccelerate(128, 3)
        self.iter_guide_layer5_s2 = CSPNGenerateAccelerate(128, 5)
        self.iter_guide_layer7_s2 = CSPNGenerateAccelerate(128, 7)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.nnupsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.downsample = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPNAccelerate(kernel_size=3, dilation=1, padding=1, stride=1)
        self.CSPN5 = CSPNAccelerate(kernel_size=5, dilation=1, padding=2, stride=1)
        self.CSPN7 = CSPNAccelerate(kernel_size=7, dilation=1, padding=3, stride=1)
        self.CSPN3_s2 = CSPNAccelerate(kernel_size=3, dilation=2, padding=2, stride=1)
        self.CSPN5_s2 = CSPNAccelerate(kernel_size=5, dilation=2, padding=4, stride=1)
        self.CSPN7_s2 = CSPNAccelerate(kernel_size=7, dilation=2, padding=6, stride=1)

        # CSPN
        ks = 3
        encoder3 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder3[index] = 1
        self.encoder3 = nn.Parameter(encoder3, requires_grad=False)

        ks = 5
        encoder5 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder5[index] = 1
        self.encoder5 = nn.Parameter(encoder5, requires_grad=False)

        ks = 7
        encoder7 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder7[index] = 1
        self.encoder7 = nn.Parameter(encoder7, requires_grad=False)

        weights_init(self)

    def forward(self, input):

        d = input['d']
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        feature_s1, feature_s2, coarse_depth = self.backbone(input)
        depth = coarse_depth

        d_s2, valid_mask_s2 = self.downsample(d, valid_mask)
        mask_s2 = self.mask_layer_s2(feature_s2)
        mask_s2 = torch.sigmoid(mask_s2)
        mask_s2 = mask_s2*valid_mask_s2

        kernel_conf_s2 = self.kernel_conf_layer_s2(feature_s2)
        kernel_conf_s2 = self.softmax(kernel_conf_s2)
        kernel_conf3_s2 = self.nnupsample(kernel_conf_s2[:, 0:1, :, :])
        kernel_conf5_s2 = self.nnupsample(kernel_conf_s2[:, 1:2, :, :])
        kernel_conf7_s2 = self.nnupsample(kernel_conf_s2[:, 2:3, :, :])

        guide3_s2 = self.iter_guide_layer3_s2(feature_s2)
        guide5_s2 = self.iter_guide_layer5_s2(feature_s2)
        guide7_s2 = self.iter_guide_layer7_s2(feature_s2)

        depth_s2 = self.nnupsample(d_s2)
        mask_s2 = self.nnupsample(mask_s2)
        depth3 = depth5 = depth7 = depth

        mask = self.mask_layer(feature_s1)
        mask = torch.sigmoid(mask)
        mask = mask * valid_mask

        kernel_conf = self.kernel_conf_layer(feature_s1)
        kernel_conf = self.softmax(kernel_conf)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        guide3 = self.iter_guide_layer3(feature_s1)
        guide5 = self.iter_guide_layer5(feature_s1)
        guide7 = self.iter_guide_layer7(feature_s1)

        guide3 = kernel_trans(guide3, self.encoder3)
        guide5 = kernel_trans(guide5, self.encoder5)
        guide7 = kernel_trans(guide7, self.encoder7)

        guide3_s2 = kernel_trans(guide3_s2, self.encoder3)
        guide5_s2 = kernel_trans(guide5_s2, self.encoder5)
        guide7_s2 = kernel_trans(guide7_s2, self.encoder7)

        guide3_s2 = self.nnupsample(guide3_s2)
        guide5_s2 = self.nnupsample(guide5_s2)
        guide7_s2 = self.nnupsample(guide7_s2)

        for i in range(6):
            depth3 = self.CSPN3_s2(guide3_s2, depth3, coarse_depth)
            depth3 = mask_s2*depth_s2 + (1-mask_s2)*depth3
            depth5 = self.CSPN5_s2(guide5_s2, depth5, coarse_depth)
            depth5 = mask_s2*depth_s2 + (1-mask_s2)*depth5
            depth7 = self.CSPN7_s2(guide7_s2, depth7, coarse_depth)
            depth7 = mask_s2*depth_s2 + (1-mask_s2)*depth7

        depth_s2 = kernel_conf3_s2*depth3 + kernel_conf5_s2*depth5 + kernel_conf7_s2*depth7
        refined_depth_s2 = depth_s2

        depth3 = depth5 = depth7 = refined_depth_s2

        #prop
        for i in range(6):
            depth3 = self.CSPN3(guide3, depth3, depth_s2)
            depth3 = mask*d + (1-mask)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth_s2)
            depth5 = mask*d + (1-mask)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth_s2)
            depth7 = mask*d + (1-mask)*depth7

        refined_depth = kernel_conf3*depth3 + kernel_conf5*depth5 + kernel_conf7*depth7

        return refined_depth

class PENet_C4(nn.Module):
    def __init__(self, args):
        super(PENet_C4, self).__init__()

        self.backbone = ENet(args)

        self.kernel_conf_layer = convbn(64, 3)
        self.mask_layer = convbn(64, 1)
        self.prop_mask_layer = convbn(64, 1)
        self.iter_guide_layer3 = CSPNGenerateAccelerate(64, 3)
        self.iter_guide_layer5 = CSPNGenerateAccelerate(64, 5)
        self.iter_guide_layer7 = CSPNGenerateAccelerate(64, 7)

        self.kernel_conf_layer_s2 = convbn(128, 3)
        self.mask_layer_s2 = convbn(128, 1)
        self.prop_mask_layer_s2 = convbn(128, 1)
        self.iter_guide_layer3_s2 = CSPNGenerateAccelerate(128, 3)
        self.iter_guide_layer5_s2 = CSPNGenerateAccelerate(128, 5)
        self.iter_guide_layer7_s2 = CSPNGenerateAccelerate(128, 7)

        self.kernel_conf_layer_s3 = convbn(256, 3)
        self.mask_layer_s3 = convbn(256, 1)
        self.prop_mask_layer_s3 = convbn(256, 1)
        self.iter_guide_layer3_s3 = CSPNGenerateAccelerate(256, 3)
        self.iter_guide_layer5_s3 = CSPNGenerateAccelerate(256, 5)
        self.iter_guide_layer7_s3 = CSPNGenerateAccelerate(256, 7)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.nnupsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.nnupsample4 = nn.UpsamplingNearest2d(scale_factor=4)
        self.downsample = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPNAccelerate(kernel_size=3, dilation=1, padding=1, stride=1)
        self.CSPN5 = CSPNAccelerate(kernel_size=5, dilation=1, padding=2, stride=1)
        self.CSPN7 = CSPNAccelerate(kernel_size=7, dilation=1, padding=3, stride=1)
        self.CSPN3_s2 = CSPNAccelerate(kernel_size=3, dilation=2, padding=2, stride=1)
        self.CSPN5_s2 = CSPNAccelerate(kernel_size=5, dilation=2, padding=4, stride=1)
        self.CSPN7_s2 = CSPNAccelerate(kernel_size=7, dilation=2, padding=6, stride=1)
        self.CSPN3_s3 = CSPNAccelerate(kernel_size=3, dilation=4, padding=4, stride=1)
        self.CSPN5_s3 = CSPNAccelerate(kernel_size=5, dilation=4, padding=8, stride=1)
        self.CSPN7_s3 = CSPNAccelerate(kernel_size=7, dilation=4, padding=12, stride=1)

        # CSPN
        ks = 3
        encoder3 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder3[index] = 1
        self.encoder3 = nn.Parameter(encoder3, requires_grad=False)

        ks = 5
        encoder5 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder5[index] = 1
        self.encoder5 = nn.Parameter(encoder5, requires_grad=False)

        ks = 7
        encoder7 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder7[index] = 1
        self.encoder7 = nn.Parameter(encoder7, requires_grad=False)

        weights_init(self)

    def forward(self, input):
        #rgb = input['rgb']
        d = input['d']
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        feature_s1, feature_s2, feature_s3, coarse_depth = self.backbone(input)
        depth = coarse_depth

        d_s2, valid_mask_s2 = self.downsample(d, valid_mask)
        d_s3, valid_mask_s3 = self.downsample(d_s2, valid_mask_s2)

        #s3
        mask_s3 = self.mask_layer_s3(feature_s3)
        mask_s3 = torch.sigmoid(mask_s3)
        mask_s3 = mask_s3 * valid_mask_s3
        prop_mask_s3 = self.prop_mask_layer_s3(feature_s3)
        prop_mask_s3 = torch.sigmoid(prop_mask_s3)

        kernel_conf_s3 = self.kernel_conf_layer_s3(feature_s3)
        kernel_conf_s3 = self.softmax(kernel_conf_s3)
        kernel_conf3_s3 = self.nnupsample4(kernel_conf_s3[:, 0:1, :, :])
        kernel_conf5_s3 = self.nnupsample4(kernel_conf_s3[:, 1:2, :, :])
        kernel_conf7_s3 = self.nnupsample4(kernel_conf_s3[:, 2:3, :, :])

        guide3_s3 = self.iter_guide_layer3_s3(feature_s3)
        guide5_s3 = self.iter_guide_layer5_s3(feature_s3)
        guide7_s3 = self.iter_guide_layer7_s3(feature_s3)

        guide3_s3 = kernel_trans(guide3_s3, self.encoder3)
        guide5_s3 = kernel_trans(guide5_s3, self.encoder5)
        guide7_s3 = kernel_trans(guide7_s3, self.encoder7)

        guide3_s3 = prop_mask_s3*guide3_s3
        guide5_s3 = prop_mask_s3*guide5_s3
        guide7_s3 = prop_mask_s3*guide7_s3

        guide3_s3 = self.nnupsample4(guide3_s3)
        guide5_s3 = self.nnupsample4(guide5_s3)
        guide7_s3 = self.nnupsample4(guide7_s3)

        depth_s3 = self.nnupsample4(d_s3)
        mask_s3 = self.nnupsample4(mask_s3)
        depth3 = depth5 = depth7 = depth

        for i in range(4):
            depth3 = self.CSPN3_s3(guide3_s3, depth3, coarse_depth)
            depth3 = mask_s3 * depth_s3 + (1 - mask_s3) * depth3
            depth5 = self.CSPN5_s3(guide5_s3, depth5, coarse_depth)
            depth5 = mask_s3 * depth_s3 + (1 - mask_s3) * depth5
            depth7 = self.CSPN7_s3(guide7_s3, depth7, coarse_depth)
            depth7 = mask_s3 * depth_s3 + (1 - mask_s3) * depth7

        depth_s3 = kernel_conf3_s3 * depth3 + kernel_conf5_s3 * depth5 + kernel_conf7_s3 * depth7
        refined_depth_s3 = depth_s3

        #s2
        mask_s2 = self.mask_layer_s2(feature_s2)
        mask_s2 = torch.sigmoid(mask_s2)
        mask_s2 = mask_s2*valid_mask_s2
        prop_mask_s2 = self.prop_mask_layer_s2(feature_s2)
        prop_mask_s2 = torch.sigmoid(prop_mask_s2)

        kernel_conf_s2 = self.kernel_conf_layer_s2(feature_s2)
        kernel_conf_s2 = self.softmax(kernel_conf_s2)
        kernel_conf3_s2 = self.nnupsample(kernel_conf_s2[:, 0:1, :, :])
        kernel_conf5_s2 = self.nnupsample(kernel_conf_s2[:, 1:2, :, :])
        kernel_conf7_s2 = self.nnupsample(kernel_conf_s2[:, 2:3, :, :])

        guide3_s2 = self.iter_guide_layer3_s2(feature_s2)
        guide5_s2 = self.iter_guide_layer5_s2(feature_s2)
        guide7_s2 = self.iter_guide_layer7_s2(feature_s2)

        guide3_s2 = kernel_trans(guide3_s2, self.encoder3)
        guide5_s2 = kernel_trans(guide5_s2, self.encoder5)
        guide7_s2 = kernel_trans(guide7_s2, self.encoder7)

        guide3_s2 = prop_mask_s2*guide3_s2
        guide5_s2 = prop_mask_s2*guide5_s2
        guide7_s2 = prop_mask_s2*guide7_s2

        guide3_s2 = self.nnupsample(guide3_s2)
        guide5_s2 = self.nnupsample(guide5_s2)
        guide7_s2 = self.nnupsample(guide7_s2)

        depth_s2 = self.nnupsample(d_s2)
        mask_s2 = self.nnupsample(mask_s2)
        depth3 = depth5 = depth7 = refined_depth_s3

        for i in range(4):
            depth3 = self.CSPN3_s2(guide3_s2, depth3, depth_s3)
            depth3 = mask_s2*depth_s2 + (1-mask_s2)*depth3
            depth5 = self.CSPN5_s2(guide5_s2, depth5, depth_s3)
            depth5 = mask_s2*depth_s2 + (1-mask_s2)*depth5
            depth7 = self.CSPN7_s2(guide7_s2, depth7, depth_s3)
            depth7 = mask_s2*depth_s2 + (1-mask_s2)*depth7

        depth_s2 = kernel_conf3_s2*depth3 + kernel_conf5_s2*depth5 + kernel_conf7_s2*depth7
        refined_depth_s2 = depth_s2

        #s1
        mask = self.mask_layer(feature_s1)
        mask = torch.sigmoid(mask)
        mask = mask*valid_mask
        prop_mask = self.prop_mask_layer(feature_s1)
        prop_mask = torch.sigmoid(prop_mask)

        kernel_conf = self.kernel_conf_layer(feature_s1)
        kernel_conf = self.softmax(kernel_conf)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        guide3 = self.iter_guide_layer3(feature_s1)
        guide5 = self.iter_guide_layer5(feature_s1)
        guide7 = self.iter_guide_layer7(feature_s1)

        guide3 = kernel_trans(guide3, self.encoder3)
        guide5 = kernel_trans(guide5, self.encoder5)
        guide7 = kernel_trans(guide7, self.encoder7)

        guide3 = prop_mask*guide3
        guide5 = prop_mask*guide5
        guide7 = prop_mask*guide7

        depth3 = depth5 = depth7 = refined_depth_s2

        for i in range(4):
            depth3 = self.CSPN3(guide3, depth3, depth_s2)
            depth3 = mask*d + (1-mask)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth_s2)
            depth5 = mask*d + (1-mask)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth_s2)
            depth7 = mask*d + (1-mask)*depth7

        refined_depth = kernel_conf3*depth3 + kernel_conf5*depth5 + kernel_conf7*depth7
        return refined_depth

class PENet_C1_train(nn.Module):
    def __init__(self, args):
        super(PENet_C1_train, self).__init__()

        self.backbone = ENet(args)
        self.mask_layer = convbn(64, 3)

        self.kernel_conf_layer = convbn(64, 3)
        self.iter_conf_layer = convbn(64, 12)
        self.iter_guide_layer3 = CSPNGenerate(64, 3)
        self.iter_guide_layer5 = CSPNGenerate(64, 5)
        self.iter_guide_layer7 = CSPNGenerate(64, 7)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPN(3)
        self.CSPN5 = CSPN(5)
        self.CSPN7 = CSPN(7)

        weights_init(self)

    def forward(self, input):
        #rgb = input['rgb']
        d = input['d']
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        feature, coarse_depth = self.backbone(input)

        mask = self.mask_layer(feature)
        mask = torch.sigmoid(mask)
        mask = mask*valid_mask
        mask3 = mask[:, 0:1, :, :]
        mask5 = mask[:, 1:2, :, :]
        mask7 = mask[:, 2:3, :, :]

        kernel_conf = self.kernel_conf_layer(feature)
        kernel_conf = self.softmax(kernel_conf)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        conf = self.iter_conf_layer(feature)
        conf3 = conf[:, 0:4, :, :]
        conf5 = conf[:, 4:8, :, :]
        conf7 = conf[:, 8:12, :, :]
        conf3 = self.softmax(conf3)
        conf5 = self.softmax(conf5)
        conf7 = self.softmax(conf7)

        #guide3 = self.iter_guide_layer3(feature)
        #guide5 = self.iter_guide_layer5(feature)
        #guide7 = self.iter_guide_layer7(feature)

        #init
        depth = coarse_depth
        depth3 = depth
        depth5 = depth
        depth7 = depth

        d3_list = [i for i in range(4)]
        d5_list = [i for i in range(4)]
        d7_list = [i for i in range(4)]

        #prop
        guide3 = self.iter_guide_layer3(feature)
        guide5 = self.iter_guide_layer5(feature)
        guide7 = self.iter_guide_layer7(feature)

        for i in range(12):
            depth3 = self.CSPN3(guide3, depth3, depth)
            depth3 = mask3*d + (1-mask3)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth)
            depth5 = mask5*d + (1-mask5)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth)
            depth7 = mask7*d + (1-mask7)*depth7

            if(i==2):
                d3_list[0] = depth3
                d5_list[0] = depth5
                d7_list[0] = depth7

            if(i==5):
                d3_list[1] = depth3
                d5_list[1] = depth5
                d7_list[1] = depth7

            if(i==8):
                d3_list[2] = depth3
                d5_list[2] = depth5
                d7_list[2] = depth7

            if(i==11):
                d3_list[3] = depth3
                d5_list[3] = depth5
                d7_list[3] = depth7

        refined_depth = \
        d3_list[0] * (kernel_conf3 * conf3[:, 0:1, :, :]) + \
        d3_list[1] * (kernel_conf3 * conf3[:, 1:2, :, :]) + \
        d3_list[2] * (kernel_conf3 * conf3[:, 2:3, :, :]) + \
        d3_list[3] * (kernel_conf3 * conf3[:, 3:4, :, :]) + \
        d5_list[0] * (kernel_conf5 * conf5[:, 0:1, :, :]) + \
        d5_list[1] * (kernel_conf5 * conf5[:, 1:2, :, :]) + \
        d5_list[2] * (kernel_conf5 * conf5[:, 2:3, :, :]) + \
        d5_list[3] * (kernel_conf5 * conf5[:, 3:4, :, :]) + \
        d7_list[0] * (kernel_conf7 * conf7[:, 0:1, :, :]) + \
        d7_list[1] * (kernel_conf7 * conf7[:, 1:2, :, :]) + \
        d7_list[2] * (kernel_conf7 * conf7[:, 2:3, :, :]) + \
        d7_list[3] * (kernel_conf7 * conf7[:, 3:4, :, :])

        return refined_depth

class PENet_C2_train(nn.Module):
    def __init__(self, args):
        super(PENet_C2_train, self).__init__()

        self.backbone = ENet(args)

        self.kernel_conf_layer = convbn(64, 3)
        self.mask_layer = convbn(64, 1)
        self.iter_guide_layer3 = CSPNGenerate(64, 3)
        self.iter_guide_layer5 = CSPNGenerate(64, 5)
        self.iter_guide_layer7 = CSPNGenerate(64, 7)

        self.kernel_conf_layer_s2 = convbn(128, 3)
        self.mask_layer_s2 = convbn(128, 1)
        self.iter_guide_layer3_s2 = CSPNGenerate(128, 3)
        self.iter_guide_layer5_s2 = CSPNGenerate(128, 5)
        self.iter_guide_layer7_s2 = CSPNGenerate(128, 7)

        self.dimhalf_s2 = convbnrelu(128, 64, 1, 1, 0)
        self.att_12 = convbnrelu(128, 2)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downsample = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPN(3)
        self.CSPN5 = CSPN(5)
        self.CSPN7 = CSPN(7)

        weights_init(self)

    def forward(self, input):
        d = input['d']
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        # feature_s1 = torch.cat((rgb_feature0_plus, decoder_feature5), 1)      # [b, 64, 320, 1216]
        # feature_s2 = torch.cat((rgb_feature2_plus, decoder_feature4),1)       # [b, 128, 160, 608]
        # coarse_depth = output                                                 # [b, 1, 320, 1216]
        feature_s1, feature_s2, coarse_depth = self.backbone(input)
        depth = coarse_depth

        d_s2, valid_mask_s2 = self.downsample(d, valid_mask)    # [b, 1, 160, 608]
        mask_s2 = self.mask_layer_s2(feature_s2)                # b 128 160 608 --> b 1 160 608
        mask_s2 = torch.sigmoid(mask_s2)                        # b 1 160 608
        mask_s2 = mask_s2*valid_mask_s2                         # b 1 160 608

        kernel_conf_s2 = self.kernel_conf_layer_s2(feature_s2)  # b 3 160 608
        kernel_conf_s2 = self.softmax(kernel_conf_s2)           # b 3 160 608
        kernel_conf3_s2 = kernel_conf_s2[:, 0:1, :, :]          # b 1 160 608
        kernel_conf5_s2 = kernel_conf_s2[:, 1:2, :, :]          # b 1 160 608
        kernel_conf7_s2 = kernel_conf_s2[:, 2:3, :, :]          # b 1 160 608

        mask = self.mask_layer(feature_s1)      # b 64 320 1216 --> b 1 320 1216
        mask = torch.sigmoid(mask)              # b 1 320 1216
        mask = mask*valid_mask                  # b 1 320 1216

        kernel_conf = self.kernel_conf_layer(feature_s1)    # b 64 320 1216 --> b 3 320 1216
        kernel_conf = self.softmax(kernel_conf)             # b 3 320 1216
        kernel_conf3 = kernel_conf[:, 0:1, :, :]            # b 1 320 1216
        kernel_conf5 = kernel_conf[:, 1:2, :, :]            # b 1 320 1216
        kernel_conf7 = kernel_conf[:, 2:3, :, :]            # b 1 320 1216

        feature_12 = torch.cat((feature_s1, self.upsample(self.dimhalf_s2(feature_s2))), 1)     # b 128 320 1216
        att_map_12 = self.softmax(self.att_12(feature_12))          # b 2 320 1216

        guide3_s2 = self.iter_guide_layer3_s2(feature_s2)           # b 9 162 610
        guide5_s2 = self.iter_guide_layer5_s2(feature_s2)           # b 25 164 612
        guide7_s2 = self.iter_guide_layer7_s2(feature_s2)           # b 49 166 614
        guide3 = self.iter_guide_layer3(feature_s1)             # b 9 322 1218
        guide5 = self.iter_guide_layer5(feature_s1)             # b 25 324 1220
        guide7 = self.iter_guide_layer7(feature_s1)             # b 49 326 1222

        depth_s2 = depth                                # b 1 320 1216
        depth_s2_00 = depth_s2[:, :, 0::2, 0::2]        # b 1 160 608   行　列　都取偶数
        depth_s2_01 = depth_s2[:, :, 0::2, 1::2]        # b 1 160 608   行　取偶数，列　取奇数
        depth_s2_10 = depth_s2[:, :, 1::2, 0::2]        # b 1 160 608   行　取奇数，　列　取偶数
        depth_s2_11 = depth_s2[:, :, 1::2, 1::2]        # b 1 160 608   行　列　哦读取奇数

        depth_s2_00_h0 = depth3_s2_00 = depth5_s2_00 = depth7_s2_00 = depth_s2_00       # b 1 160 608
        depth_s2_01_h0 = depth3_s2_01 = depth5_s2_01 = depth7_s2_01 = depth_s2_01       # b 1 160 608
        depth_s2_10_h0 = depth3_s2_10 = depth5_s2_10 = depth7_s2_10 = depth_s2_10       # b 1 160 608
        depth_s2_11_h0 = depth3_s2_11 = depth5_s2_11 = depth7_s2_11 = depth_s2_11       # b 1 160 608

        for i in range(6):
            depth3_s2_00 = self.CSPN3(guide3_s2, depth3_s2_00, depth_s2_00_h0)          # b 1 160 608
            depth3_s2_00 = mask_s2*d_s2 + (1-mask_s2)*depth3_s2_00
            depth5_s2_00 = self.CSPN5(guide5_s2, depth5_s2_00, depth_s2_00_h0)          # b 1 160 608
            depth5_s2_00 = mask_s2*d_s2 + (1-mask_s2)*depth5_s2_00
            depth7_s2_00 = self.CSPN7(guide7_s2, depth7_s2_00, depth_s2_00_h0)          # b 1 160 608
            depth7_s2_00 = mask_s2*d_s2 + (1-mask_s2)*depth7_s2_00

            depth3_s2_01 = self.CSPN3(guide3_s2, depth3_s2_01, depth_s2_01_h0)          # b 1 160 608
            depth3_s2_01 = mask_s2*d_s2 + (1-mask_s2)*depth3_s2_01
            depth5_s2_01 = self.CSPN5(guide5_s2, depth5_s2_01, depth_s2_01_h0)          # b 1 160 608
            depth5_s2_01 = mask_s2*d_s2 + (1-mask_s2)*depth5_s2_01
            depth7_s2_01 = self.CSPN7(guide7_s2, depth7_s2_01, depth_s2_01_h0)          # b 1 160 608
            depth7_s2_01 = mask_s2*d_s2 + (1-mask_s2)*depth7_s2_01

            depth3_s2_10 = self.CSPN3(guide3_s2, depth3_s2_10, depth_s2_10_h0)          # b 1 160 608
            depth3_s2_10 = mask_s2*d_s2 + (1-mask_s2)*depth3_s2_10
            depth5_s2_10 = self.CSPN5(guide5_s2, depth5_s2_10, depth_s2_10_h0)          # b 1 160 608
            depth5_s2_10 = mask_s2*d_s2 + (1-mask_s2)*depth5_s2_10
            depth7_s2_10 = self.CSPN7(guide7_s2, depth7_s2_10, depth_s2_10_h0)          # b 1 160 608
            depth7_s2_10 = mask_s2*d_s2 + (1-mask_s2)*depth7_s2_10

            depth3_s2_11 = self.CSPN3(guide3_s2, depth3_s2_11, depth_s2_11_h0)          # b 1 160 608
            depth3_s2_11 = mask_s2*d_s2 + (1-mask_s2)*depth3_s2_11
            depth5_s2_11 = self.CSPN5(guide5_s2, depth5_s2_11, depth_s2_11_h0)          # b 1 160 608
            depth5_s2_11 = mask_s2*d_s2 + (1-mask_s2)*depth5_s2_11
            depth7_s2_11 = self.CSPN7(guide7_s2, depth7_s2_11, depth_s2_11_h0)          # b 1 160 608
            depth7_s2_11 = mask_s2*d_s2 + (1-mask_s2)*depth7_s2_11

        depth_s2_00 = kernel_conf3_s2*depth3_s2_00 + kernel_conf5_s2*depth5_s2_00 + kernel_conf7_s2*depth7_s2_00        # b 1 160 608
        depth_s2_01 = kernel_conf3_s2*depth3_s2_01 + kernel_conf5_s2*depth5_s2_01 + kernel_conf7_s2*depth7_s2_01        # b 1 160 608
        depth_s2_10 = kernel_conf3_s2*depth3_s2_10 + kernel_conf5_s2*depth5_s2_10 + kernel_conf7_s2*depth7_s2_10        # b 1 160 608
        depth_s2_11 = kernel_conf3_s2*depth3_s2_11 + kernel_conf5_s2*depth5_s2_11 + kernel_conf7_s2*depth7_s2_11        # b 1 160 608

        depth_s2[:, :, 0::2, 0::2] = depth_s2_00
        depth_s2[:, :, 0::2, 1::2] = depth_s2_01
        depth_s2[:, :, 1::2, 0::2] = depth_s2_10
        depth_s2[:, :, 1::2, 1::2] = depth_s2_11        # depth_s2: b 1 320 1216

        #feature_12 = torch.cat((feature_s1, self.upsample(self.dimhalf_s2(feature_s2))), 1)
        #att_map_12 = self.softmax(self.att_12(feature_12))
        refined_depth_s2 = depth * att_map_12[:, 0:1, :, :] + depth_s2 * att_map_12[:, 1:2, :, :]       # b 1 320 1216
        #refined_depth_s2 = depth

        depth3 = depth5 = depth7 = refined_depth_s2

        #prop
        for i in range(6):
            depth3 = self.CSPN3(guide3, depth3, depth)      # b 1 320 1216
            depth3 = mask*d + (1-mask)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth)
            depth5 = mask*d + (1-mask)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth)
            depth7 = mask*d + (1-mask)*depth7

        refined_depth = kernel_conf3 * depth3 + kernel_conf5 * depth5 + kernel_conf7 * depth7       # b 1 320 1216
        return refined_depth

if __name__ == "__main__":
    from main import args
    x = torch.rand(2, 3, 320, 1216)
    model = ENet(args)
    print(str(float(sum(p.numel() for p in model.parameters())/1000000)) + 'M')

    args.n = 'pe'
    model = PENet_C2_train(args)
    print(str(float(sum(p.numel() for p in model.parameters())/1000000)) + 'M')



