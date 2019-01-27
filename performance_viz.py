####Importing necessary libraries####
import os
import numpy as np
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

####plotting roc curve####
def plot_roc_curve(model, X, y, weight_path, save_file, x_plotlim = None, y_plotlim = None, figSize = None):
    ####Loading weights of units in model####
    model.load_weights(weight_path)
    
    ####Getting prediction scores####
    y_pred_probs = model.predict(X)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= 3

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig = plt.figure(figsize=figSize)

    if not x_plotlim is None or not y_plotlim is None:
        plt.xlim(-.0125,x_plotlim)
        plt.ylim(y_plotlim,1.0125)
    
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='green', linestyle='--', linewidth=1)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='brown', linestyle='--', linewidth=1)

    colors = ['red', 'orange', 'blue']
    classes = ['BCC', 'SCC', 'Benign']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=colors[i], linewidth=1,
                 label='ROC curve of {0} class (area = {1:0.4f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0,1],[0,1],'k--', linewidth=1)
    plt.grid(True)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('ROC curves for our multi-class classification model',fontsize=18)
    plt.legend(loc="lower right", title = 'ROC curves', title_fontsize = 18, prop={'size':15})
    plt.show()
    if not os.path.exists('visualization'):
        os.makedirs('visualization')
    fig.savefig(os.path.join("visualization",save_file+".png"), transparent=True, dpi=5*fig.dpi)

####Plotting confusion matrix####
def plot_conf_mat(model, X, y, weight_path, save_file):
    model.load_weights(weight_path)
    y_true_labels = np.argmax(y, axis = 1)
    y_pred_labels = np.argmax(model.predict(X), axis = 1)

    confmat = confusion_matrix(y_true_labels, y_pred_labels)
    fig, _ = plot_confusion_matrix(conf_mat= confmat, figsize=(15,15))
    if not os.path.exists('visualization'):
        os.makedirs('visualization')
    fig.savefig(os.path.join("visualization",save_file+".png"), transparent=True, dpi=5*fig.dpi)
    plt.show()
