import tensorflow as tf
from utils import weights_spectral_norm
class STMFusionNet():
    def vi_feature_extraction_network(self, vi_image):
        with tf.compat.v1.variable_scope('vi_extraction_network'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", [5, 5, 1, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(vi_image, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv1 = tf.contrib.layers.batch_norm(conv1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                    conv1 = tf.nn.leaky_relu(conv1)
                # state size: 32
                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", [3, 3, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv2 = tf.contrib.layers.batch_norm(conv2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                    conv2 = tf.nn.leaky_relu(conv2)
                # concat_conv2 = tf.concat([conv2, conv1], axis=-1)
                # state size: 32
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", [3, 3, 16, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv3 = tf.contrib.layers.batch_norm(conv3, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                    conv3 = tf.nn.leaky_relu(conv3)
                concat_conv3 = tf.concat([conv3, conv2], axis=-1)
                # state size: 64
                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", [3, 3, 32, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [64], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv4 = tf.contrib.layers.batch_norm(conv4, decay=0.9, updates_collections=None, epsilon=1e-5,
                    #                                      scale=True)
                    conv4 = tf.nn.sigmoid(conv4)
                    concat_conv4 = tf.concat([conv4, concat_conv3], axis=-1)
                    encoding_feature = concat_conv4
        return encoding_feature

    def ir_feature_extraction_network(self, ir_image):
        with tf.compat.v1.variable_scope('ir_extraction_network'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", [5, 5, 1, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(ir_image, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv1 = tf.contrib.layers.batch_norm(conv1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                    conv1 = tf.nn.leaky_relu(conv1)
                # state size: 32
                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", [3, 3, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv2 = tf.contrib.layers.batch_norm(conv2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                    conv2 = tf.nn.leaky_relu(conv2)
                # concat_conv2 = tf.concat([conv2, conv1], axis=-1)
                # state size: 32
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", [3, 3, 16, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv3 = tf.contrib.layers.batch_norm(conv3, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                    conv3 = tf.nn.leaky_relu(conv3)
                concat_conv3 = tf.concat([conv3, conv2], axis=-1)
                # state size: 64
                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", [3, 3, 32, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [64], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv4 = tf.contrib.layers.batch_norm(conv4, decay=0.9, updates_collections=None, epsilon=1e-5,
                    #                                      scale=True)
                    conv4 = tf.nn.sigmoid(conv4)
                    concat_conv4 = tf.concat([conv4, concat_conv3], axis=-1)
                    encoding_feature = concat_conv4
        return encoding_feature

    def feature_reconstruction_network(self, feature):
        with tf.compat.v1.variable_scope('reconstruction_network'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", [3, 3, 224, 128],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [128], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(feature, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv1 = tf.contrib.layers.batch_norm(conv1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                    conv1 = tf.nn.leaky_relu(conv1)
                # state size: 128
                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", [3, 3, 128, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [64], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv2 = tf.contrib.layers.batch_norm(conv2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                    conv2 = tf.nn.leaky_relu(conv2)
                # state size: 64
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", [3, 3, 64, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv3 = tf.contrib.layers.batch_norm(conv3, decay=0.9, updates_collections=None, epsilon=1e-5,
                    #  scale=True)
                    conv3 = tf.nn.leaky_relu(conv3)
                # state size: 32
                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", [3, 3, 32, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv4 = tf.contrib.layers.batch_norm(conv4, decay=0.9, updates_collections=None, epsilon=1e-5,
                    #                                      scale=True)
                    conv4 = tf.nn.leaky_relu(conv4)
                with tf.compat.v1.variable_scope('conv5'):
                    weights = tf.compat.v1.get_variable("w", [3, 3, 16, 1],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    weights = weights_spectral_norm(weights)
                    bias = tf.compat.v1.get_variable("b", [1], initializer=tf.constant_initializer(0.0))
                    conv5 = tf.nn.conv2d(conv4, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    # conv5 = tf.contrib.layers.batch_norm(conv5, decay=0.9, updates_collections=None, epsilon=1e-5,
                    #                                       scale=True)
                    conv5 = tf.nn.tanh(conv5)
                    fusion_image = conv5
        return fusion_image


    def STMFusion_model(self, vi_image, ir_image):
        with tf.variable_scope("STMFusion_model"):
            vi_feature = self.vi_feature_extraction_network(vi_image)
            ir_feature = self.ir_feature_extraction_network(ir_image)
            feature = tf.concat([vi_feature, ir_feature], axis=-1)
            f_image = self.feature_reconstruction_network(feature)
        return f_image