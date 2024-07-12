#!/usr/bin/env python3

"""
Assignment 6: Big Data Computing

hsreefman
"""

import sys
import model
import data


def train_model(input):
    """Main function"""

    x_s, y_s = data.mnist_mini(input, num=5000)

    trn_xs, trn_ys = x_s[0:2000], y_s[0:2000]
    val_xs, val_ys = x_s[2000:3500], y_s[2000:3500]
    tst_xs, tst_ys = x_s[3500:5000], y_s[3500:5000]

    my_model = model.InputLayer(144) + \
           model.DenseLayer(64) + model.ActivationLayer(64, activation=model.tanh) +    \
           model.DenseLayer(48) + model.ActivationLayer(48, activation=model.tanh) +    \
           model.DenseLayer(32) + model.ActivationLayer(32, activation=model.elish) +   \
           model.DenseLayer(24) + model.SoftmaxLayer(24) + \
           model.LossLayer(loss=model.categorical_crossentropy)

    my_history = my_model.fit(trn_xs, trn_ys, alpha=0.01, epochs=8, batch_size=32,
                              validation_data=(val_xs, val_ys))

    print(f'Loss: {my_model.evaluate(tst_xs, tst_ys)}')
    accuracy = data.confusion(tst_xs, tst_ys, model=my_model)
    print(f'Accuracy: {accuracy*100}')

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python assignment6.py <data.dat>")
        sys.exit(1)

    print(sys.argv[1])

    train_model(sys.argv[1])
