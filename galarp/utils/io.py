import pickle


def pickle_obj(obj, name="obj.out"):
    with open(name, "wb") as f:
        pickle.dump(obj, f)
