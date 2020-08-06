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
            return scipy.misc.imread(path, flatten=True, mode='RGB').astype(np.float)
        else:
            return scipy.misc.imread(path, mode='RGB').astype(np.float)

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
        input_ir = self.imread(self.data_ir[index], is_grayscale=True) / 255 # (self.imread(self.data_ir[index]) - 127.5) / 127.5
        input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_ir.shape
        input_ir = input_ir.reshape([w, h, 1])
        input_vi = self.imread(self.data_vi[index], is_grayscale=False) / 255  #(self.imread(self.data_vi[index]) - 127.5) / 127.5#
        input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_vi.shape
        input_vi = input_vi.reshape([w, h, 1])
        sub_ir_sequence.append(input_ir)
        sub_vi_sequence.append(input_vi)
        train_data_ir = np.asarray(sub_ir_sequence)
        train_data_vi = np.asarray(sub_vi_sequence)
        return train_data_ir, train_data_vi



    def STMFusion(self):
        ir_path = r'/data/tlf/STMFusionNet/RoadScene/Test_ir'
        vi_path = r'/data/tlf/STMFusionNet/RoadScene/RGB_Test_vi'
        fused_path = r'RoadScene_RGB_Results'
        fused_path = os.path.join(os.getcwd(), fused_path)
        if not os.path.exists(fused_path):
            os.makedirs(fused_path)
        filelist = os.listdir(ir_path)

        num_epochs = 30
        num_epoch = 28
        for idx_num in range(num_epoch, num_epochs):
            print("num_epoch:\t", num_epoch)
            while (num_epoch == idx_num):

                model_path = './checkpoint/STMFusion_32_Pixel_Grad/Fusion.model-' + str(num_epoch)
                fusion_reader = tf.compat.v1.train.NewCheckpointReader(model_path)
                print('Read model!')
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
                    for item in filelist:
                        if item.endswith('.bmp') or item.endswith('.tif'):
                            num = int(item.split('.')[0])
                            ir_image_name = os.path.join(os.path.abspath(ir_path), item)
                            vi_image_name = os.path.join(os.path.abspath(vi_path), item)
                            fused_image_name = os.path.join(os.path.abspath(fused_path), item)
                            ir_image = cv2.imread(ir_image_name, 1) / 255
                            ir_b, ir_g, ir_r = np.array(cv2.split(ir_image))
                            [h, w] = ir_b.shape
                            ir_b = ir_b.reshape(1, h, w, 1)
                            ir_g = ir_g.reshape(1, h, w, 1)
                            ir_r = ir_r.reshape(1, h, w, 1)
                            vi_image = cv2.imread(vi_image_name) / 255
                            vi_b, vi_g, vi_r = np.array(cv2.split(vi_image))
                            vi_b = vi_b.reshape(1, h, w, 1)
                            vi_g = vi_g.reshape(1, h, w, 1)
                            vi_r = vi_r.reshape(1, h, w, 1)
                            start = time.time()
                            result_b, encoding_feature = sess.run([self.fusion_image, self.feature], feed_dict={
                                ir_images: ir_b, vi_images: vi_b})
                            result_g, encoding_feature = sess.run([self.fusion_image, self.feature], feed_dict={
                                ir_images: ir_g, vi_images: vi_g})
                            result_r, encoding_feature = sess.run([self.fusion_image, self.feature], feed_dict={
                                ir_images: ir_r, vi_images: vi_r})
                            result_b = result_b * 255
                            result_g = result_g * 255
                            result_r = result_r * 255
                            result_b = result_b.squeeze()
                            result_g = result_g.squeeze()
                            result_r = result_r.squeeze()
                            result = cv2.merge([result_r, result_g, result_b])
                            end = time.time()
                            image_path = os.path.join(fused_path, item)
                            print(image_path)
                            self.imsave(result, image_path)
                            print("Testing [%d] successfully,Testing time is [%f]" % (num, end - start))
                            try:
                                print('successfully!')
                            except:
                                print("Testing [0] unsuccess!" .format(num))
                                continue
                num_epoch = num_epoch + 1
            tf.reset_default_graph()


test_STMFusion = STMFusion()
test_STMFusion.STMFusion()
