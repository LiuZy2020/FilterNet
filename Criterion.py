import os
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

from DataReader_Radar import RadarDataset
from FilterNet import *

os.environ["CUDA_VISIBLE_DEVICES"] = ' '

def roc_auc_cureve(labels, preds, method, is_Tensor = True, is_OC = False):
    if is_Tensor:
        labels = labels.detach().numpy()
        preds = preds.detach().numpy()
    fpr, tpr, threshold = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    if is_OC:
        plt.plot(fpr, tpr, lw=2, label=method + '(AUC=%0.3f)' % roc_auc, linestyle='--')
    else:
        plt.plot(fpr, tpr, lw=2, label=method + '(AUC=%0.3f)' %roc_auc) #color='darkorange'
    plt.legend(loc='lower right')

def ConfusionMatrix(labels, preds, is_Tensor=True):
    if is_Tensor:
        labels = labels.detach().numpy()
        preds = preds.detach().numpy()
    CM = confusion_matrix(labels, preds)
    tn, fp, fn, tp = CM.ravel()
    acc = (tn+tp) / (tn+tp+fn+fp)
    error = 1 - acc
    precision = tp/(tp+fp)
    false_alarm = fp/(tn+fp)
    recall = tp/(tp+fn)
    F1score = 2*precision*recall/(precision+recall)
    TPR = tp/(tp+fn)
    FPR = fp/(fp+tn)
    TNR = tn/(fp+tn)
    print('acc:', acc)
    print('error:',error)
    print('precision:', precision)
    print('false alarm:', false_alarm)
    print('recall:', recall)
    print('F1-score:', F1score)
    print('TPR:', TPR)
    print('FPR:', FPR)
    print('TNR:', TNR)

if __name__ == '__main__':
    file_path_test = './Data'
    Dataset = RadarDataset(file_path_test, len(file_path_test))
    images = Dataset.images
    images_n = Dataset.noised
    labels = Dataset.labels
    print(len(labels))

    ### our method #####
    fig = plt.figure(dpi=300)
    fig.subplots_adjust(top=0.98)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    with torch.no_grad():
        index = '1'
        e_model = torch.load('./Model_saved/model_encoder.pth').cuda()
        g_model = torch.load('./Model_saved/model_generator.pth').cuda()
        c_model = torch.load('./Model_saved/model_classifier.pth').cuda()
        e_model.eval()
        g_model.eval()
        c_model.eval()

        images_R = g_model(e_model(images.cuda()))
        preds = c_model(images_R)
        preds = torchF.softmax(preds, dim=1)
        preds = preds.cpu()
        print("-----------Our Method-----------")
        roc_auc_cureve(labels, preds[:,1], 'Our Method')
        ConfusionMatrix(labels, preds.argmax(dim=1))
        print("+++++++++++Our Method+++++++++++")
        plt.show()
