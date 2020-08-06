# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import warnings
import glob
import cv2
import xlwt
import xlrd
from xlutils.copy import copy
import os
from utils import (
    read_data,
    imsave,
    merge,
    gradient,
    lrelu,
    weights_spectral_norm,
    l2_norm
)

# reader = tf.train.NewCheckpointReader("./checkpoint/CGAN_120/CGAN.model-9")
from test_network import STMFusionNet

STMFusion_net = STMFusionNet()
warnings.filterwarnings('ignore')


class STMFusion:
    def imread(self, path, is_grayscale=True):
        """
        Read image using its path.
        Default value  is gray-scale, and image is read by YCbCr format as the paper said.
        """
        if is_grayscale:
            # flatten=True Read the image as a grayscale map.
            return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
        else:
            return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

    def imsave(self, image, path):
        return scipy.misc.imsave(path, image)

    def prepare_data(self, dataset):
        self.data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
        data = glob.glob(os.path.join(self.data_dir, "*.tif"))
        data.extend(glob.glob(os.path.join(self.data_dir, "*.bmp")))
        data.sort(key=lambda x: int(x[len(self.data_dir) + 1:-4]))
        return data

    def lrelu(x, leak=0.2):
        return tf.maximum(x, leak * x)

    def input_setup(self, index):
        padding = 0
        sub_ir_sequence = []
        sub_vi_sequence = []
        input_ir = self.imread(self.data_ir[index]) / 255 # (self.imread(self.data_ir[index]) - 127.5) / 127.5
        input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_ir.shape
        input_ir = input_ir.reshape([w, h, 1])
        input_vi = self.imread(self.data_vi[index]) / 255  #(self.imread(self.data_vi[index]) - 127.5) / 127.5#
        input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_vi.shape
        input_vi = input_vi.reshape([w, h, 1])
        sub_ir_sequence.append(input_ir)
        sub_vi_sequence.append(input_vi)
        train_data_ir = np.asarray(sub_ir_sequence)
        train_data_vi = np.asarray(sub_vi_sequence)
        return train_data_ir, train_data_vi


    def STMFusion(self):
        num_epochs = 30
        num_epoch = 28
        for idx_num in range(num_epoch, num_epochs):
            print("num_epoch:\t", num_epoch)
            while (num_epoch == idx_num):
                model_path = './checkpoint/STMFusion_32_Pixel_Grad/Fusion.model-' + str(num_epoch)
                fusion_reader = tf.compat.v1.train.NewCheckpointReader(model_path)
                with tf.name_scope('IR_input'):
                    # infrared image patch
                    ir_images = tf.placeholder(tf.float32, [1, None, None, 1], name='ir_images')
                with tf.name_scope('VI_input'):
                    # visible image patch
                    vi_images = tf.placeholder(tf.float32, [1, None, None, 1], name='vi_images')
                    # self.labels_vi_gradient=gradient(self.labels_vi)

                with tf.name_scope('fusion'):
                    self.fusion_image, self.feature = STMFusion_net.STMFusion_model(vi_images, ir_images, fusion_reader)
                with tf.Session() as sess:
                    init_op = tf.global_variables_initializer()
                    sess.run(init_op)

                    self.data_ir = self.prepare_data(r'/data/tlf/STMFusionNet/TNO/Test_ir')
                    self.data_vi = self.prepare_data(r'/data/tlf/STMFusionNet/TNO/Test_vi')
                    print(len(self.data_ir))
                    for i in range(len(self.data_ir)):
                        try:
                            train_data_ir, train_data_vi = self.input_setup(i)
                            start = time.time()
                            result, encoding_feature = sess.run([self.fusion_image, self.feature], feed_dict={
                                ir_images: train_data_ir, vi_images: train_data_vi})
                            result = result * 255
                            # fusion_feature_result = fusion_feature_result * 127.5 + 127.5
                            # fusion_feature_result = fusion_feature_result.squeeze()
                            result = result.squeeze()
                            encoding_feature = encoding_feature.squeeze()
                            # mask_result = mask_result.squeeze()
                            end = time.time()
                            image_path = os.path.join(os.getcwd(), 'TNO_Results',
                                                      'epoch' + str(num_epoch))
                            if not os.path.exists(image_path):
                                os.makedirs(image_path)
                            num = "%02d" % ( i + 1)
                            image_path = os.path.join(image_path, num + ".bmp")
                            # save_name = '测试时间统计.xls'
                            # self.writexls(save_name, i + 1, '%.6f'%(end - start))
                            # print(out.shape)
                            self.imsave(result, image_path)
                            print(image_path)
                            # if idx_num == num_epochs - 1:
                            #     save_Encode_name = os.path.join('.', 'Encoding_feature', 'epoch{}'.format(num_epoch))
                            #     if not os.path.exists(save_Encode_name):
                            #         os.makedirs(save_Encode_name)
                            #     for convi in range(np.size(encoding_feature, -1)):
                            #         encoder_convi = encoding_feature[:, :, convi]
                            #         encoder_convi = (encoder_convi - np.min(encoder_convi)) / (
                            #         np.max(encoder_convi) - np.min(encoder_convi))
                            #         encoder_convi = encoder_convi * 255
                            #         if i < 9:
                            #             image_save_name = os.path.join(save_Encode_name, '0' + str(i + 1) + '_' + str(convi + 1) + '.bmp')
                            #         else:
                            #             image_save_name = os.path.join(save_Encode_name, str(i + 1) + '_' + str(convi + 1) + '.bmp')
                            #
                            #         cv2.imwrite(image_save_name, encoder_convi)
                            #         print('Save {} Successful!'.format(image_save_name))
                            print("Testing [%d] successfully,Testing time is [%f]" % (i + 1, end - start))
                        except:
                            print("Testing [%d] unsuccess!" % (i + 1))
                            continue
                num_epoch = num_epoch + 1
            tf.reset_default_graph()


test_STMFusion = STMFusion()
test_STMFusion.STMFusion()
