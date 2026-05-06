class DecisionEngine:

    def __init__(self,
                 offer_cost=20,
                 success_rate=0.3,
                 retained_months=12):
        self.offer_cost = offer_cost
        self.success_rate = success_rate
        self.retained_months = retained_months

    def compute_clv(self, df):
        return df["MonthlyCharges"] * self.retained_months

    def expected_profit(self, prob, clv):
        return (prob * self.success_rate * clv) - self.offer_cost

    def decide(self, df, probs):
        clv = self.compute_clv(df)
        profit = self.expected_profit(probs, clv)

        return {
            "probability": probs,
            "clv": clv,
            "profit": profit,
            "target": profit > 0
        }
