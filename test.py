import unittest
import numpy as np
from src.model_class import leading_indicator_miner


class TestMinerDiscoversDeterministicIndicators(unittest.TestCase):
    def test_discover_deterministic_indicators(self):
        """
        Test whether the leading_indicator_miner can discover deterministic leading indicators hidden in the data
        ..and closely model their relationship to the response (y)
        """
        X_data = np.random.random((10_000, 20)) * 100
        X_names = [f"x_{i}" for i in range(X_data.shape[1])]
        perfect_indicator_idx = np.random.choice(
            range(len(X_names)), size=5, replace=False
        )
        perfect_indicator_varnames = [X_names[i] for i in perfect_indicator_idx]
        perfect_indicator_lags = np.random.choice(
            [6], size=len(perfect_indicator_varnames), replace=True
        )
        y_vec = np.random.uniform(low=0, high=100, size=len(X_data))
        X_data[:, perfect_indicator_idx[0]] = np.concatenate(
            [
                0.1 * y_vec[perfect_indicator_lags[0] :],
                X_data[
                    (len(X_data) - perfect_indicator_lags[0]) :,
                    perfect_indicator_idx[0],
                ],
            ]
        )
        X_data[:, perfect_indicator_idx[1]] = np.concatenate(
            [
                np.sqrt(y_vec[perfect_indicator_lags[1] :]),
                X_data[
                    (len(X_data) - perfect_indicator_lags[1]) :,
                    perfect_indicator_idx[1],
                ],
            ]
        )
        X_data[:, perfect_indicator_idx[2]] = np.concatenate(
            [
                np.log(y_vec[perfect_indicator_lags[2] :]),
                X_data[
                    (len(X_data) - perfect_indicator_lags[2]) :,
                    perfect_indicator_idx[2],
                ],
            ]
        )
        X_data[:, perfect_indicator_idx[3]] = np.concatenate(
            [
                (y_vec[perfect_indicator_lags[3] :]) ** 2 / 10,
                X_data[
                    (len(X_data) - perfect_indicator_lags[3]) :,
                    perfect_indicator_idx[3],
                ],
            ]
        )
        X_data[:, perfect_indicator_idx[4]] = np.concatenate(
            [
                y_vec[perfect_indicator_lags[4] :] ** 3,
                X_data[
                    (len(X_data) - perfect_indicator_lags[4]) :,
                    perfect_indicator_idx[4],
                ],
            ]
        )
        leading_indicator_miner_model = leading_indicator_miner(
            n_leading_indicators=5,
            n_knots={
                # optional: number of knots to use in the continuous piecewise linear spline fit (default is linear regression without splines)
                "min": 50,
                "max": 50,
            },
        )
        leading_indicator_miner_model.fit(
            X=X_data,  # X_data[:, perfect_indicator_idx],
            X_varnames=X_names,  # [X_names[i] for i in perfect_indicator_idx],
            y=y_vec,
            y_varname="outcome_y",
            n_iterations=1_000,
            n_lags_to_consider={
                "min": perfect_indicator_lags.min(),
                "max": perfect_indicator_lags.max(),
            },
        )
        best_indicators_identified = {
            x["leading_indicator_varname"]: np.round(x["mean_squared_error"], 2)
            for x in leading_indicator_miner_model.best_leading_indicators_vars_set
        }
        # check if all useful leading indicators have been identified:
        self.assertCountEqual(
            list(best_indicators_identified.keys()), perfect_indicator_varnames
        )
        # check that the relationship has been adequately modelled:


if __name__ == "__main__":
    unittest.main()
