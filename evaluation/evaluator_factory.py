from evaluation.ucr_evaluator import evaluate as ucr_evaluator


def get_evaluator(name):
    if name == "ucr_evaluator":
        return ucr_evaluator
