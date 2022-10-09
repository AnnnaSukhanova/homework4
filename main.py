import numpy as np
import plotly
import plotly.express as px


class Config:
    max_height = 230
    N = 500
    height = 185


class BinaryClassificator:

    def __init__(self, height):
        self.height = height

    def __classify(self, X):
        return X < self.height

    def predict(self, X):
        return self.__classify(X)


def calcArea(X, Y):
    result = 0
    for i in range(len(X) - 1):
        result += (X[i + 1] - X[i]) * (Y[i] + Y[i + 1]) / 2
    return result


def accuracy(tp, tn):
    return (tp + tn) / (2 * Config.N)


def presicion(tp, fp):
    if fp == 0:
        return 1
    else:
        return (tp) / (tp + fp)


def recall(tp, fn):
    if fn == 0:
        return 1
    else:
        return (tp) / (tp + fn)


def f1_score(presicion, recall):
    return 2 * (recall * presicion) / (recall + presicion)


def error_1(tn, fp):
    return (fp) / (fp + tn)


def error_2(tp, fn):
    return (fn) / (fn + tp)


soccer_players = np.random.randn(Config.N) * 20 + 160
basket_players = np.random.randn(Config.N) * 10 + 190

presicions, recalls, accuracy_list, threshold = [], [], [], []
FPRs, TPRs = [], []

for t in range(Config.max_height):
    classificator = BinaryClassificator(t)
    TP = sum(classificator.predict(soccer_players))
    FP = sum(classificator.predict(basket_players))
    FN = Config.N - TP
    TN = Config.N - FP
    Precision = presicion(TP, FP)
    Recall = recall(TP, FN)
    acc = accuracy(TP, TN)
    Alpha = error_1(TN, FP)
    frp = Alpha
    tpr = Recall
    FPRs.append(frp)
    TPRs.append(tpr)
    accuracy_list.append(acc)
    threshold.append(t)
    presicions.append(Precision)
    recalls.append(Recall)

fig = px.line(x=recalls, y=presicions,
              title=f"График Presicion-Recall кривой, AUC={calcArea(recalls, presicions)}",
              labels=dict(x="recalls", y="presicions", hover_data_0="accuracy", hover_data_1="threshold"),
              hover_data=[accuracy_list, threshold])
plotly.offline.plot(fig, filename=f'C:/plotly/Presicion-Recall_Curve.html')


fig2 = px.line(x=FPRs, y=TPRs,
              title=f"График ROC кривой, AUC = {calcArea(FPRs, TPRs)}",
              labels=dict(x="FPRs", y="TPRs"))
plotly.offline.plot(fig2, filename=f'C:/plotly/ROC.html')

