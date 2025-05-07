import pickle

with open("../data/mnist/pickled_mnist.pkl", "bw") as fh:
    data = (train_imgs, 
            test_imgs, 
            train_labels,
            test_labels)
    pickle.dump(data, fh)