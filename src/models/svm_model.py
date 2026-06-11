from sklearn.svm import LinearSVC

from config import RANDOM_STATE, SVM_C, SVM_MAX_ITER


def build_model(C=SVM_C):
    return LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=SVM_MAX_ITER,
        random_state=RANDOM_STATE,
    )
