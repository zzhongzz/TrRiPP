from plot_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

RIPPS_CLASSES = ['NONRIPP', 'AUTO_INDUCING_PEPTIDE', 'BACTERIAL_HEAD_TO_TAIL_CYCLIZED',
                 'CYANOBACTIN', 'LANTHIPEPTIDE', 'LASSO_PEPTIDE', 'rSAM_MODIFIED_RiPP',
                 'THIOPEPTIDE', 'GRASPETIDE', 'OTHER']


def plot_cm(preds, oracles, is_binary: bool, file=None, loss=None):
    cm = confusion_matrix(oracles, preds)
    if is_binary:
        fig = plot_confusion_matrix(cm, ['NONRIPP', 'RIPP'], loss=loss, figsize=[8, 8], normalize=True)
    else:
        fig = plot_confusion_matrix(cm, RIPPS_CLASSES, loss=loss, figsize=[24, 20], normalize=True)
    if file is not None:
        plt.savefig(file)

    return fig


def plot_roc(labels, scores, file=None):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if file is not None:
        plt.savefig(file)

    return fig
