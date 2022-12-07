import sys

sys.path.append("..")  # add the root project folder to the system path
import unittest
import numpy as np
from schuberts_leader.core_components import (
    simulate_leading_indicator_data,
    leading_indicator_miner,
)


class TestMeanSquaredErrorFunction(unittest.TestCase):
    """
    this test checks whether the .mean_squared_error() function in the schuberts_leader.leading_indicator_miner class works correctly for a simple example
    the "ignore_nan" functionality is tested in a separate test (within this test)
    """

    def test_meanSquaredErrorFunction(self):
        obs_y = np.array([6, -9, 4, -20, 999])
        model_y = np.array([-1, 2, -3, 4, np.nan])
        leading_indicator_miner_model = leading_indicator_miner(
            n_leading_indicators=1
        )  # we instantiate a model object because the MeanSquaredError function lives within the model class
        self.assertAlmostEqual(
            198.75,
            leading_indicator_miner_model.mean_squared_error(
                y_true=model_y[:4], y_pred=obs_y[:4]
            ),
            places=10,
        )
        self.assertAlmostEqual(
            198.75,
            leading_indicator_miner_model.mean_squared_error(
                y_true=model_y, y_pred=obs_y, ignore_nan=True
            ),
            places=10,
        )


class TestFullModelPipeline(unittest.TestCase):
    """
    this test checks whether {schuberts leader} can (near) perfectly identify and model (i.e. predict using) deterministic leading indicator signals hidden within a simulated dataset (it must find all of them in order to pass the test)
    this implicitly tests the functionality of the following component functions:
        1. schuberts_leader.core_components.simulate_leading_indicator_data()
        2. schuberts_leader.core_components.leading_indicator_miner.fit()
        3. schuberts_leader.core_components.leading_indicator_miner.create_linear_splines()
        4. schuberts_leader.core_components.leading_indicator_miner.estimate_OLS_linear_model_coefs()
        5. schuberts_leader.core_components.leading_indicator_miner.mean_squared_error()
        6. schuberts_leader.core_components.predict()
        7. schuberts_leader.core_components.leading_indicator_miner.generate_linear_model_preds()
    since random predictors are considered (at random lags) in the search for leading indicators, the leading indicator search is forced to continue running until every true leading indicator has been seen by the model
    """

    def test_full_model_pipeline(self):

        # simulate data with deterministic leading effect relationships #
        n_true_leading_indicators = 3
        sim_explain_dict, y_arr, X_arr = simulate_leading_indicator_data(
            n_time_points=1_000,
            n_predictors=10,
            n_leading_indicators=n_true_leading_indicators,
            lagged_effect_time_min_max=(7, 8),
            n_y_breakpoints=8,
            y_sim_method="independent_gaussian",
            x_sim_method="independent_gaussian",
            noise_std_dev=0.0,  # no noise = deterministic signal
        )
        X_varnames = [f"X_{i}" for i in range(X_arr.shape[1])]
        actual_leading_indicators_ref = set(
            (X_varnames[i], sim_explain_dict[i]["relationship_lag"])
            for i in sim_explain_dict.keys()
        )
        train_X_arr = X_arr[: len(X_arr) - 8, :]
        train_y_arr = y_arr[: len(y_arr) - 8]
        test_X_arr = X_arr[len(X_arr) - 8 :, :]
        test_y_arr = y_arr[len(y_arr) - 8 :]
        leading_indicator_miner_model = leading_indicator_miner(
            n_leading_indicators=n_true_leading_indicators
        )
        n_true_indicators_assessed = 0
        while (
            n_true_indicators_assessed < n_true_leading_indicators
        ):  # keep looping until the model has seen every true leading indicator
            leading_indicator_miner_model.fit(
                X=train_X_arr,
                X_varnames=X_varnames,
                y=train_y_arr,
                y_varname="y",
                n_iterations=100,
                lags_to_consider=[7, 8],
                n_knots=100,
                knot_strategy="quantiles",  # "evenly_spaced"
                keep_training_history=True,
            )
            leading_indicators_assessed = set(
                (x["leading_indicator_varname"], x["lag_n_time_periods"])
                for x in leading_indicator_miner_model.training_history
            )
            n_true_indicators_assessed = len(
                leading_indicators_assessed.intersection(actual_leading_indicators_ref)
            )

        best_leading_indicators_identified = {
            (x["leading_indicator_varname"], x["lag_n_time_periods"])
            for x in leading_indicator_miner_model.best_leading_indicators_vars_set
        }
        self.assertEqual(
            best_leading_indicators_identified, actual_leading_indicators_ref
        )
        best_leading_indicators_idx = [
            X_varnames.index(x[0]) for x in best_leading_indicators_identified
        ]
        test_data_preds = leading_indicator_miner_model.predict(
            X=train_X_arr[:, best_leading_indicators_idx],
            X_varnames=[X_varnames[i] for i in best_leading_indicators_idx],
        )
        test_data_mse_per_leading_indicator_list = []
        max_forecast_horizon = test_data_preds.shape[1]
        for i in range(test_data_preds.shape[1]):
            test_data_mse_per_leading_indicator_list.append(
                leading_indicator_miner_model.mean_squared_error(
                    y_true=test_y_arr[:max_forecast_horizon],
                    y_pred=test_data_preds[:max_forecast_horizon, i],
                    ignore_nan=True,
                )
            )
        self.assertAlmostEqual(
            max(test_data_mse_per_leading_indicator_list), 0.0, places=3
        )


class TestLinearSplinesModel(unittest.TestCase):
    def test_linear_splines_model(self):
        """
        this test checks whether the .create_linear_splines() function in the schuberts_leader.leading_indicator_miner class works as expected for a simple example
        this also implicitly tests the functionality of the following component functions:
            1. schuberts_leader.core_components.leading_indicator_miner.estimate_OLS_linear_model_coefs()
            2. schuberts_leader.core_components.leading_indicator_miner.generate_linear_model_preds()
            3. schuberts_leader.core_components.leading_indicator_miner.mean_squared_error()
        """
        x_vec = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
        y_vec = np.array([50, 60, 70, 60, 50, 60, 70, 60, 50])
        knot_points = [50, 100, 150]
        # we instantiate a model object because the following functions live within the model class:
        #   - create_linear_splines()
        #   - estimate_OLS_linear_model_coefs()
        #   - generate_linear_model_preds()
        #   - mean_squared_error()
        leading_indicator_miner_model = leading_indicator_miner(n_leading_indicators=1)
        x_matrix = np.column_stack(
            [
                np.ones(len(x_vec)),
                x_vec,
                leading_indicator_miner_model.create_linear_splines(
                    X_vec=x_vec, knot_points_list=knot_points
                ),
            ]
        )
        beta_est = leading_indicator_miner_model.estimate_OLS_linear_model_coefs(
            X_matrix=x_matrix, y_vec=y_vec
        )
        pred_y = leading_indicator_miner_model.generate_linear_model_preds(
            X_matrix=x_matrix, beta_coefs_vec=beta_est
        )
        model_fit_mse = leading_indicator_miner_model.mean_squared_error(
            y_true=y_vec, y_pred=pred_y
        )
        self.assertAlmostEqual(model_fit_mse, 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
