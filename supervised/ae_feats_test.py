# Build a simple supervised model using one of the embeddings.
import re
import numpy as np
import pandas as pd
from sklearn import naive_bayes, model_selection, metrics, svm, linear_model, tree


RUN_ID = 'bb_lstm_20steps_2017-06-14_19.57.18'
MODEL_FILE = 'best.hdf5'

print('Loading data')
df = pd.read_csv('bb/models/' + RUN_ID + '/' + MODEL_FILE.replace('.hdf5', '-predict-aligned.csv'))
features = [f for f in df if re.fullmatch('^z[0-9]+$', f)]
print('Found ' + str(len(features)) + ' embedding features: ' + str(features))
print(df.behavior.value_counts())

X = df[features].values
y = np.array([1 if l == 'UDEF1' else 0 for l in df.behavior])
cm = []
aucs = []
kappas = []
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
