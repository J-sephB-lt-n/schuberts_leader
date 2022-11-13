import numpy as np


class leading_indicator_miner:
    """
    object for mining data in search of non-linear leading time series indicators (possibly multivariate)
        - currently only supports numeric indicator and numeric outcome variable

    goal of this model:
        * identifies leading indicators (univariate), allowing non-linear relationship between the leading indicator (x) and the outcome (y)
        * uses only numpy (no extra package dependencies)
        * doesn't try to do too much - does 1 thing well
        * can be easily controlled in terms of calculation time
        * is very opinionated
        * doesn't accept many different kinds of input (only numpy array). Only integer or float.
        * has a sklearn-like interface (e.g. .fit(), .predict() etc.)
        * proper class documentation (i.e. go fix this Joe)

    -- EXAMPLE USAGE --
    help(leading_indicator_miner)
    import numpy as np
    raw_input_data = np.random.random( (10_000,50) )

    leading_indicator_miner_model = leading_indicator_miner(
        n_knots={"min":0,"max":10},
        n_leading_indicators=5,
    )
    leading_indicator_miner_model.fit(
        X = raw_input_data[:,0:49],
        X_varnames = [f"x{i}" for i in range(1,50)],
        y = raw_input_data[:,49],
        y_varname = "y",
        n_lags_to_consider={"min":5,"max":10},
        n_iterations = 1_000
    )
    leading_indicator_miner_model.best_leading_indicators_vars_set
    """

    def __init__(
        self,
        n_knots={
            # optional: number of knots to use in the continuous piecewise linear spline fit (default is linear regression without splines)
            "min": 0,
            "max": 0,
        },
        n_leading_indicators=2,  # number of leading indicators to identify
    ):
        assert (
            1 == 1
        ), "joe you need to add an assertion here checking the input data types"

        self.n_knots = n_knots
        self.n_leading_indicators = n_leading_indicators
        self.best_leading_indicators_vars_set = (
            # list of top leading indicators found so far, ranked best to worst
            []
        )
        self.total_iterations_counter = 0
        self.best_mse_seen_in_training = None

    def mean_squared_error(self, y_true, y_pred):
        return np.sum((y_pred - y_true) ** 2)

    def fit(
        self,
        X,
        X_varnames,
        y,
        y_varname,
        n_iterations,
        n_lags_to_consider,  # e.g. n_lags_to_consider={"min":5,"max":10}
    ):
        assert len(X[0]) == len(
            X_varnames
        ), "must have number of columns in X == len(X_varnames)"
        assert len(X) == len(
            y
        ), "X must contain the same number of time steps as y (same number of rows)"
        for i in range(n_iterations):
            self.total_iterations_counter += 1

            # choose random model hyperparameters for this iteration #
            predictor_varname_this_iter = np.random.choice(X_varnames)
            predictor_idx_this_iter = X_varnames.index(predictor_varname_this_iter)
            if n_lags_to_consider["min"] == n_lags_to_consider["max"]:
                lag_this_iter = n_lags_to_consider["min"]
            else:
                lag_this_iter = np.random.randint(
                    low=n_lags_to_consider["min"], high=n_lags_to_consider["max"]
                )
            if self.n_knots["min"] == self.n_knots["max"]:
                n_knots_this_iter = self.n_knots["min"]
            else:
                n_knots_this_iter = np.random.randint(
                    low=self.n_knots["min"], high=self.n_knots["max"]
                )

            # illustration of how features are lagged:
            """
            time:       0 1 2 3 4 5 6 7 8 9 10
            lag size 3:
                        X	y
                        0	3
                        1	4
                        .	.
                        .	.
                        7	10
            """
            # prepare X features for model:
            # (features are stored in a dictionary, with the variable name as key)
            X_dict = {}
            X_dict[predictor_varname_this_iter] = X[
                : (len(X) - lag_this_iter), predictor_idx_this_iter
            ]
            X_dict["intercept"] = np.array(
                [1] * len(X_dict[predictor_varname_this_iter])
            )
            # create spline features (trendline change features):
            knot_locations_this_iter = np.unique(
                # only unique quantiles are kept as knot locations
                np.quantile(
                    a=X_dict[predictor_varname_this_iter],
                    q=[
                        1.0 / (n_knots_this_iter + 1) * i
                        for i in range(1, n_knots_this_iter + 1)
                    ],
                )
            )
            quantile_counter = 0
            for quantile in knot_locations_this_iter:
                quantile_counter += 1
                X_dict[
                    f"{predictor_varname_this_iter}_slopechange_{quantile_counter}"
                ] = (X_dict[predictor_varname_this_iter] > quantile) * (
                    X_dict[predictor_varname_this_iter] - quantile
                )

            y_this_iter = y[lag_this_iter:]

            # fit linear regression model:
            X_mat = np.column_stack(list(X_dict.values()))
            xT_x = np.matmul(X_mat.transpose(), X_mat)
            xT_x_inv = np.linalg.inv(xT_x)
            beta_estimates = np.matmul(
                np.matmul(xT_x_inv, X_mat.transpose()), y_this_iter
            )
            fit_y = np.matmul(X_mat, beta_estimates)

            # assess model fit to training data:
            train_mse = self.mean_squared_error(y_true=y_this_iter, y_pred=fit_y)

            if (
                self.best_mse_seen_in_training is None
                or self.best_mse_seen_in_training > train_mse
            ):
                self.best_mse_seen_in_training = train_mse

            leading_indicator_varnames_currently_in_best_set = [
                x["leading_indicator_varname"]
                for x in self.best_leading_indicators_vars_set
            ]

            # if we haven't got [n_leading_indicators] yet, then just add this one in
            if (
                len(self.best_leading_indicators_vars_set) < self.n_leading_indicators
                and predictor_varname_this_iter
                not in leading_indicator_varnames_currently_in_best_set
            ):
                self.best_leading_indicators_vars_set.append(
                    {
                        "leading_indicator_varname": predictor_varname_this_iter,
                        "mean_squared_error": train_mse,
                        "lag_n_time_periods": lag_this_iter,
                        "n_knots": n_knots_this_iter,
                        "knot_positions": knot_locations_this_iter,
                        "model_coefficients": beta_estimates,
                    }
                )
            elif (
                # if we already have [n_leading_indicators], and current leading indicator is better than the worst one in [best_leading_indicators_vars_set]
                len(self.best_leading_indicators_vars_set) == self.n_leading_indicators
                and train_mse
                < self.best_leading_indicators_vars_set[0]["mean_squared_error"]
            ):
                # if the current leading indicator is already in the [best_leading_indicators_vars_set]
                if (
                    predictor_varname_this_iter
                    in leading_indicator_varnames_currently_in_best_set
                ):
                    idx_in_best_set = [
                        i
                        for i in range(len(self.best_leading_indicators_vars_set))
                        if self.best_leading_indicators_vars_set[i][
                            "leading_indicator_varname"
                        ]
                        == predictor_varname_this_iter
                    ][0]
                    # replace the existing current entry with the current (better) entry:
                    self.best_leading_indicators_vars_set[idx_in_best_set] = {
                        "leading_indicator_varname": predictor_varname_this_iter,
                        "mean_squared_error": train_mse,
                        "lag_n_time_periods": lag_this_iter,
                        "n_knots": n_knots_this_iter,
                        "knot_positions": knot_locations_this_iter,
                        "model_coefficients": beta_estimates,
                    }

                else:
                    # if the current leading indicator is not yet in the [best_leading_indicators_vars_set], then insert it (replacing the worst one already in the set)
                    self.best_leading_indicators_vars_set[0] = {
                        "leading_indicator_varname": predictor_varname_this_iter,
                        "mean_squared_error": train_mse,
                        "lag_n_time_periods": lag_this_iter,
                        "n_knots": n_knots_this_iter,
                        "knot_positions": knot_locations_this_iter,
                        "model_coefficients": beta_estimates,
                    }

            # sort best leading indicator list by Mean Squared Error (worst to best):
            self.best_leading_indicators_vars_set = sorted(
                self.best_leading_indicators_vars_set,
                key=lambda d: -d["mean_squared_error"],
            )

            print(
                f"\riteration {i+1} of {n_iterations}. best MSE: {self.best_mse_seen_in_training:.3f}",
                end="",
                flush=True,
            )

    def predict(self, X, X_varnames):
        print(X)
        print(X_varnames)
