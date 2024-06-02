import pickle


def classify(data):
    with open('finalized_model.sav', 'rb') as file:
        clf = pickle.load(file)

    return clf.predict(data)
