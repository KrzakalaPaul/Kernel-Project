from sklearn.model_selection import KFold 
from sklearn.metrics import roc_auc_score
import pickle as pkl
import numpy as np

with open('training_labels.pkl', 'rb') as file:
    train_labels = pkl.load(file)

kf = KFold(n_splits=3, shuffle=True)

def eval_model(model,C=1e-3,class_weight='balanced'):
    roc=[] 
    archetype_kernel=np.zeros((6000,6000))
    for i, (train_index, test_index) in enumerate(kf.split(archetype_kernel)):

        model.fit(train_indices=train_index,C=C,class_weight=class_weight)

        preds=model.pred(test_indices=test_index)
        y_test=train_labels[test_index]

        roc.append(roc_auc_score(y_test,preds))

    print(roc)
    #print(sum(roc)/len(roc))
    return sum(roc)/len(roc)