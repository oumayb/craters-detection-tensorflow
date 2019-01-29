import tensorflow as tf

import os

import numpy as np

import json


class CraterClassifier:
    def __init__(self, config):
        self.config = config
        self.n_classes = config["n_classes"]
        self.img_width = config["img_width"]
        self.img_height = config["img_height"]
        self.n_channels = config["n_channels"]
        self.base_filter_nums = config["base_filter_nums"]
        self.base_filter_sizes = config["base_filter_sizes"]
        self.pool_sizes = config["pool_sizes"]
        self.dense_layers = config["dense_layers"]
        self.dense_units = config["dense_units"]
        self.batch_size = config["batch_size"]
        self.n_epochs = config["n_epochs"]
        self.initial_lr = config["initial_lr"]

    def build_graph(self):

        tf.reset_default_graph()

        self.X_placeholder = tf.placeholder(tf.float32, shape=(None, self.img_width, self.img_height, self.n_channels),
                                            name='input')
        self.y_placeholder = tf.placeholder(tf.float32, shape=(None, 1), name='label')

        n_convs = len(self.base_filter_nums)

        # Conv block 1
        i = 0
        filters_num = self.base_filter_nums[i]
        filter_size = self.base_filter_sizes[i]
        pool_size = self.pool_sizes[i]
        # Conv2D
        net = tf.layers.conv2d(inputs=self.X_placeholder, filters=filters_num,
                               kernel_size=(filter_size, filter_size), name='conv_' + str(i))
        # Batch Norm
        net = tf.layers.batch_normalization(inputs=net, name='bn_' + str(i))
        # Max Pooling
        net = tf.layers.max_pooling2d(inputs=net, pool_size=(pool_size, pool_size), strides=(pool_size, pool_size),
                                      name="pool_" + str(i))

        # Conv blocks 2, 3, 4, 5
        for i in range(1, n_convs - 1):
            print("Building block {}".format(i))
            filters_num = self.base_filter_nums[i]
            filter_size = self.base_filter_sizes[i]
            pool_size = self.pool_sizes[i]
            # Conv2D
            net = tf.layers.conv2d(inputs=net, filters=filters_num,
                                   kernel_size=(filter_size, filter_size), name='conv_' + str(i))
            # Batch Norm
            net = tf.layers.batch_normalization(inputs=net, name='bn_' + str(i))
            # Max Pooling
            net = tf.layers.max_pooling2d(inputs=net, pool_size=(pool_size, pool_size), strides=(pool_size, pool_size),
                                          name="pool_" + str(i))

        # Conv block 6
        i = n_convs - 1
        filters_num = self.base_filter_nums[i]
        filter_size = self.base_filter_sizes[i]
        pool_size = self.pool_sizes[i]
        # Conv2D
        net = tf.layers.conv2d(inputs=net, filters=filters_num,
                               kernel_size=(filter_size, filter_size), name='conv_' + str(i + 1))
        # Batch Norm
        net = tf.layers.batch_normalization(inputs=net, name='bn_' + str(i))

        print("end conv shape: {}".format(net.shape))

        # Flatten
        net = tf.layers.flatten(inputs=net, name='flat')
        print("flat shape : {}".format(net.shape))

        # Dense layers
        dense_units = self.config["dense_units"]
        for i in range(self.dense_layers):
            net = tf.layers.dense(inputs=net, units=dense_units[i], name='dense_' + str(i + 1))
            print("Shape after dense layer n°{} : {}".format(i+1, net.shape))

        # logit layer
        logit = tf.layers.dense(inputs=net, units=1)
        print("Shape after last dense layer (logit layer): {}".format(logit.shape))

        self.prediction = tf.nn.sigmoid(logit)

        # Losses
        # Training loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_placeholder, logits=logit)
        self.loss = tf.reduce_mean(loss)

        # Validation loss
        valid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_placeholder, logits=logit)
        self.validation_loss = tf.reduce_mean(valid_loss)

        # Adam optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.initial_lr).minimize(self.loss)

        # Train metrics
        correct_pred = tf.equal(tf.cast(tf.round(self.prediction), tf.float32), self.y_placeholder)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Valid metrics
        valid_correct_pred = tf.equal(tf.cast(tf.round(self.prediction), tf.float32), self.y_placeholder)
        self.validation_acc = tf.reduce_mean(tf.cast(valid_correct_pred, tf.float32))

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.acc)
        tf.summary.scalar("valid_loss", self.validation_loss)
        tf.summary.scalar("valid_accuracy", self.validation_acc)

        self.merged_summary_op = tf.summary.merge_all()

        self.init_global = tf.global_variables_initializer()
        self.init_local = tf.local_variables_initializer()

    def fit(self, X_train, y_train, X_valid, y_valid, path_model, session, t):
        """
        Params
        ------
        :param X_train:
        :param y_train:
        :param X_valid:
        :param y_valid:
        :param path_model:
        :param session:
        :param t: time right now

        Returns
        -------
        :return:

        """

        n_train = X_train.shape[0]
        n_valid = X_valid.shape[0]

        n_iter_train = n_train // self.batch_size
        n_iter_valid = n_valid // self.batch_size

        # Save the model and the logs to keep track of the training
        logs_path = path_model + "/logs_{}".format(t)

        # Create paths for logs and model if they don't already exist
        if not os.path.exists(path_model):
            os.makedirs(path_model)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        # Save config file in the model folder to keep track of the parameters used
        with open(path_model + '/config.json', 'w') as fp:
            json.dump(self.config, fp)

        saver = tf.train.Saver()

        session.run(self.init_local)
        session.run(self.init_global)

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # Training
        for epoch in range(self.n_epochs):
            indices_train = np.arange(n_train)
            np.random.shuffle(indices_train)

            indices_valid = np.arange(n_valid)
            np.random.shuffle(indices_valid)

            train_losses = []
            valid_losses = []
            train_accuracies = []
            valid_accuracies = []

            # Train computations
            for i in range(n_iter_train):
                idx = indices_train[i * self.batch_size: (i + 1) * self.batch_size]
                X_batch = np.stack(X_train[idx])
                y_batch = np.stack(y_train[idx])
                train_loss, train_acc, _, = session.run(
                    [self.loss, self.acc, self.optimizer],
                    feed_dict={self.X_placeholder: X_batch, self.y_placeholder: y_batch})
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)

            train_loss = np.mean(train_losses)
            train_acc = np.mean(train_accuracies)

            # Valid computations
            for i in range(n_iter_valid):
                idx = indices_valid[i * self.batch_size: (i + 1) * self.batch_size]
                X_batch_valid = np.stack(X_valid[idx])
                y_batch_valid = np.stack(y_valid[idx])
                valid_loss, valid_acc = session.run(
                    [self.validation_loss, self.validation_acc],
                    feed_dict={self.X_placeholder: X_batch_valid, self.y_placeholder: y_batch_valid})
                valid_losses.append(valid_loss)
                valid_accuracies.append(valid_acc)

            valid_loss = np.mean(valid_losses)
            valid_acc = np.mean(valid_accuracies)

            """if (epoch + 1) % 10 == 0:
                print("Epoch {}: Train:  loss: {}, acc: {}".format(epoch, train_loss, train_acc))
                print("Epoch {}: Valid: loss: {}, acc: {}".format(epoch, valid_loss, valid_acc))"""

            print("Epoch {}: Train:  loss: {}, acc: {}".format(epoch, train_loss, train_acc))
            print("Epoch {}: Valid: loss: {}, acc: {}".format(epoch, valid_loss, valid_acc))

            summary = session.run(self.merged_summary_op, feed_dict={self.loss: train_loss, self.acc: train_acc,
                                                                     self.validation_loss: valid_loss,
                                                                     self.validation_acc: valid_acc})

            summary_writer.add_summary(summary, epoch)

            if epoch % 5 == 0:
                save_path = saver.save(session, path_model + "/model.ckpt")
                print("Model saved in path: %s" % save_path)

            if epoch + 1 == self.n_epochs:
                save_path = saver.save(session, path_model + "/last_model.ckpt")
                print("Last model saved in path: %s" % save_path)

    def predict(self, X, y, session):
        y_pred = session.run(self.prediction, feed_dict={self.X_placeholder: X, self.y_placeholder: y})

        return y_pred
