import pickle
import numpy as np

load_classifier = pickle.load(open('classifier.pickle','rb'))
load_scalar = pickle.load(open('sc.pickle','rb'))

pred = load_classifier.predict(load_scalar.transform(np.array([[24,45000]])))
print(pred)

pred_proba = load_classifier.predict_proba(load_scalar.transform(np.array([[24,45000]])))[:,1]
print(pred_proba)
