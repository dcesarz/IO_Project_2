from time import perf_counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


def dtc(train_inputs, test_inputs, train_classes, test_classes):
    clf = DecisionTreeClassifier()
    start_time = perf_counter()
    clf.fit(train_inputs, train_classes)
    score = clf.score(test_inputs, test_classes)
    end_time = perf_counter()
    time = end_time - start_time
    disp = plot_confusion_matrix(clf, test_inputs, test_classes,
                                     normalize='true')
    disp.ax_.set_title("Normalized confusion matrix")
    plt.savefig('dtc_confusion_matrix.png')
    return score, time


def knn(n_neighbors, train_inputs, test_inputs, train_classes, test_classes):
    clf = KNeighborsClassifier(n_neighbors, weights='distance')
    start_time = perf_counter()
    clf.fit(train_inputs, train_classes)
    score = clf.score(test_inputs, test_classes)
    end_time = perf_counter()
    time = end_time - start_time
    disp = plot_confusion_matrix(clf, test_inputs, test_classes,
                                     normalize='true')
    disp.ax_.set_title("Normalized confusion matrix")
    plt.savefig('knn{}_confusion_matrix.png'.format(n_neighbors))
    return score, time


def nbc(train_inputs, test_inputs, train_classes, test_classes):
    clf = GaussianNB()
    start_time = perf_counter()
    clf.fit(train_inputs, train_classes)
    score = clf.score(test_inputs, test_classes)
    end_time = perf_counter()
    time = end_time - start_time
    disp = plot_confusion_matrix(clf, test_inputs, test_classes,
                                     normalize='true')
    disp.ax_.set_title("Normalized confusion matrix")
    plt.savefig('nbc_confusion_matrix.png')
    return score, time


def rfc(train_inputs, test_inputs, train_classes, test_classes):
    rfc = RandomForestClassifier()
    start_time = perf_counter()
    rfc.fit(train_inputs, train_classes)
    score = rfc.score(test_inputs, test_classes)
    end_time = perf_counter()
    time = end_time - start_time
    disp = plot_confusion_matrix(rfc, test_inputs, test_classes,
                                     normalize='true')
    disp.ax_.set_title("Normalized confusion matrix")
    plt.savefig('rfc_confusion_matrix.png')
    return score, time


def svc(train_inputs, test_inputs, train_classes, test_classes):
    svc = SVC()
    start_time = perf_counter()
    svc.fit(train_inputs, train_classes)
    score = svc.score(test_inputs, test_classes)
    end_time = perf_counter()
    time = end_time - start_time
    disp = plot_confusion_matrix(svc, test_inputs, test_classes,
                                     normalize='true')
    disp.ax_.set_title("Normalized confusion matrix")
    plt.savefig('svc_confusion_matrix.png')
    return score, time


def cnn(train_inputs, test_inputs, train_classes, test_classes):
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='sigmoid'))
    model.add(Dense(7, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    start_time = perf_counter()
    model.fit(train_inputs, train_classes, validation_data=(
        test_inputs, test_classes), epochs=10, batch_size=256, verbose=1)
    cnn_score = model.evaluate(test_inputs, test_classes, verbose=1)[1]
    end_time = perf_counter()
    time = end_time - start_time
    return cnn_score, time


def mlp(train_inputs, test_inputs, train_classes, test_classes):
    mlp = MLPClassifier(max_iter=150)
    start_time = perf_counter()
    mlp.fit(train_inputs, train_classes)
    score = mlp.score(test_inputs, test_classes)
    end_time = perf_counter()
    time = end_time - start_time
    disp = plot_confusion_matrix(mlp, test_inputs, test_classes,
                                     normalize='true')
    disp.ax_.set_title("Normalized confusion matrix")
    plt.savefig('mlp_confusion_matrix.png')
    return score, time
