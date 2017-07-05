# Build a CART model using the features extracted using the traditional method.
import numpy as np
import pandas as pd
from sklearn import model_selection, naive_bayes, metrics, svm, linear_model, tree


print('Loading data')
df = pd.read_csv('bb/supervised/expert_feats_data.csv')
features = [f for f in df if any(f.endswith(x) for x in ['_mean', '_min', '_max', '_stddev'])]
print(features)

X = df[features].replace(np.nan, 0).values
y = np.array([1 if l == 'BORED' else 0 for l in df.affect])
cm = []
kappas = []
aucs = []
for train_i, test_i in model_selection.GroupKFold(4).split(X, y, df.participant_id.values):
    model = tree.DecisionTreeClassifier()
    imbalance = sum(y[train_i]) / len(train_i)
    weights = np.array([imbalance if l == 0 else 1 - imbalance for l in y[train_i]])
    model.fit(X[train_i], y[train_i], sample_weight=weights)
    preds = model.predict(X[test_i])
    probs = model.predict_proba(X[test_i])
    aucs.append(metrics.roc_auc_score(y[test_i], probs[:, 1]))
    kappas.append(metrics.cohen_kappa_score(preds, y[test_i]))
    cm.append(metrics.confusion_matrix(y[test_i], preds))
cm = np.sum(cm, axis=0)
print(cm)
print(kappas)
print('mean kappa: %.3f' % (sum(kappas) / len(kappas)))
print('mean AUC:   %.3f' % (sum(aucs) / len(aucs)))
