from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

def show_roc_curve(rf,X_test,y_test) :
    y_pred = [x[1] for x in rf.predict_proba(X_test)]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)

    roc_auc = auc(fpr, tpr)

    plt.figure(1, figsize = (15, 10))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()