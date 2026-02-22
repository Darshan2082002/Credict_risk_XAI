import numpy as np

class ScoreScaler:
    def __init__(self, pdo=20, base_score=600, base_odds=50):
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds

        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def pd_to_score(self, pd):
        odds = (1 - pd) / pd
        score = self.offset - self.factor * np.log(odds)
        return score