import tensorflow as tf
from utils import weights_spectral_norm
class STMFusionNet():
    def vi_feature_extraction_network(self, vi_image, reader):
        with tf.compat.v1.variable_scope('vi_extraction_network'):
            with tf.compat.v1.variable_scope('conv1'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/vi_extraction_network/conv1/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                    'STMFusion_model/vi_extraction_network/conv1/b')))
                conv1 = tf.nn.conv2d(vi_image, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv1 = tf.contrib.layers.batch_norm(conv1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv1 = tf.nn.leaky_relu(conv1)
            # state size: 32
            with tf.compat.v1.variable_scope('conv2'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/vi_extraction_network/conv2/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/vi_extraction_network/conv2/b')))
                conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv2 = tf.contrib.layers.batch_norm(conv2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv2 = tf.nn.leaky_relu(conv2)
            # concat_conv2 = tf.concat([conv2, conv1], axis=-1)
            # state size: 32
            with tf.compat.v1.variable_scope('conv3'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/vi_extraction_network/conv3/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/vi_extraction_network/conv3/b')))
                conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv3 = tf.contrib.layers.batch_norm(conv3, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv3 = tf.nn.leaky_relu(conv3)
            concat_conv3 = tf.concat([conv3, conv2], axis=-1)
            # state size: 64
            with tf.compat.v1.variable_scope('conv4'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/vi_extraction_network/conv4/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/vi_extraction_network/conv4/b')))
                conv4 = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv4 = tf.contrib.layers.batch_norm(conv4, decay=0.9, updates_collections=None, epsilon=1e-5,
                #                                      scale=True)
                conv4 = tf.nn.sigmoid(conv4)
                concat_conv4 = tf.concat([conv4, concat_conv3], axis=-1)
                encoding_feature = concat_conv4
        return encoding_feature

    def ir_feature_extraction_network(self, ir_image, reader):
        with tf.compat.v1.variable_scope('ir_extraction_network'):
            with tf.compat.v1.variable_scope('conv1'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/ir_extraction_network/conv1/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/ir_extraction_network/conv1/b')))
                conv1 = tf.nn.conv2d(ir_image, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv1 = tf.contrib.layers.batch_norm(conv1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv1 = tf.nn.leaky_relu(conv1)
            # state size: 32
            with tf.compat.v1.variable_scope('conv2'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/ir_extraction_network/conv2/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/ir_extraction_network/conv2/b')))
                conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv2 = tf.contrib.layers.batch_norm(conv2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv2 = tf.nn.leaky_relu(conv2)
            # concat_conv2 = tf.concat([conv2, conv1], axis=-1)
            # state size: 32
            with tf.compat.v1.variable_scope('conv3'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/ir_extraction_network/conv3/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/ir_extraction_network/conv3/b')))
                conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv3 = tf.contrib.layers.batch_norm(conv3, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv3 = tf.nn.leaky_relu(conv3)
            concat_conv3 = tf.concat([conv3, conv2], axis=-1)
            # state size: 64
            with tf.compat.v1.variable_scope('conv4'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/ir_extraction_network/conv4/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/ir_extraction_network/conv4/b')))
                conv4 = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv4 = tf.contrib.layers.batch_norm(conv4, decay=0.9, updates_collections=None, epsilon=1e-5,
                #                                      scale=True)
                conv4 = tf.nn.sigmoid(conv4)
                concat_conv4 = tf.concat([conv4, concat_conv3], axis=-1)
                encoding_feature = concat_conv4
        return encoding_feature

    def feature_reconstruction_network(self, feature, reader):
        with tf.compat.v1.variable_scope('reconstruction_network'):
            with tf.compat.v1.variable_scope('conv1'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/reconstruction_network/conv1/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/reconstruction_network/conv1/b')))
                conv1 = tf.nn.conv2d(feature, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv1 = tf.contrib.layers.batch_norm(conv1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv1 = tf.nn.leaky_relu(conv1)
            # state size: 128
            with tf.compat.v1.variable_scope('conv2'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/reconstruction_network/conv2/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/reconstruction_network/conv2/b')))
                conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv2 = tf.contrib.layers.batch_norm(conv2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv2 = tf.nn.leaky_relu(conv2)
            # state size: 64
            with tf.compat.v1.variable_scope('conv3'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/reconstruction_network/conv3/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/reconstruction_network/conv3/b')))
                conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv3 = tf.contrib.layers.batch_norm(conv3, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv3 = tf.nn.leaky_relu(conv3)
            # state size: 32
            with tf.compat.v1.variable_scope('conv4'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/reconstruction_network/conv4/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/reconstruction_network/conv4/b')))
                conv4 = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv4 = tf.contrib.layers.batch_norm(conv4, decay=0.9, updates_collections=None, epsilon=1e-5,
                #                                      scale=True)
                conv4 = tf.nn.leaky_relu(conv4)
            with tf.compat.v1.variable_scope('conv5'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/reconstruction_network/conv5/w')))
                weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(
                    reader.get_tensor('STMFusion_model/reconstruction_network/conv5/b')))
                conv5 = tf.nn.conv2d(conv4, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv5 = tf.contrib.layers.batch_norm(conv5, decay=0.9, updates_collections=None, epsilon=1e-5,
                #                                      scale=True)
                conv5 = tf.nn.tanh(conv5)
                fusion_image = conv5
        return fusion_image

    def STMFusion_model(self, vi_image, ir_image, reader):
        with tf.variable_scope("STMFusion_model"):
            vi_encoding_feature = self.vi_feature_extraction_network(vi_image, reader)
            ir_encoding_feature = self.ir_feature_extraction_network(ir_image, reader)
            feature = tf.concat([vi_encoding_feature, ir_encoding_feature], axis=-1)
            f_image = self.feature_reconstruction_network(feature, reader)
        return f_image, feature
