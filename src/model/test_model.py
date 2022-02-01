from .predictor import Predictor


class TestModel(Predictor):
    def __init__(self, funniness):
        """
        this model is for losers and nerds.
        """
        self.funniness = funniness

    def predict(self, data, targets, references, times):
        print(targets, references, times)

    def __str__(self):
        return f'i have {self.funniness} funny points'
