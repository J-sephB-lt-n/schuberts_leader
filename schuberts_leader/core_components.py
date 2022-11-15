import numpy as np


def create_linear_splines(X_vec, knot_points_list):
    """TODO: needs some documentation (see https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)"""
    splines_list = []
    for knot in knot_points_list:
        splines_list.append((X_vec > knot) * (X_vec - knot))
    return np.column_stack(splines_list)


def estimate_OLS_linear_model_coefs(X_matrix, y_vec):
    """TODO: needs some documentation (see https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)"""
    xT_x = np.matmul(X_matrix.transpose(), X_matrix)
    xT_x_inv = np.linalg.inv(xT_x)
    return np.matmul(np.matmul(xT_x_inv, X_matrix.transpose()), y_vec)


def generate_linear_model_preds(X_matrix, beta_coefs_vec):
    """TODO: needs some documentation (see https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)"""
    return np.matmul(X_matrix, beta_coefs_vec)


def mean_squared_error(y_true, y_pred):
    """TODO: needs some documentation (see https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)"""
    return np.sum((y_pred - y_true) ** 2)


def simulate_leading_indicator_data(
    n_time_points,
    n_predictors,
    n_leading_indicator_effects,
    lagged_effect_time_min_max,
    polynomial_coefs_min_max,
):
    """
    this function simulates data containing leading indicators with a noisy time-lagged cubic polynomial effect on the response variable (y)
    note that the same leading indicator can have multiple effects on the outcome variable Y (at different time lags)
    note that all X variables are populated with random independent draws from a uniform distribution on [-100,100]
    y is initialized as random independent draws from a uniform distribution on [-10,10]

    Parameters
    ----------
    n_time_points : int
        number of time points to simulate
    n_predictors : int
        number of predictor variables to simulate
    n_leading_indicator_effects : int
        number of leading indicator effects to include
    lagged_effect_time_min_max : tuple containing 2 integers
        e.g. lagged_effect_time_min_max=(1,19)
        the smallest and largest time between a leading indicator value and it's effect on the outcome
    polynomial_coefs_min_max : tuple containing 3 tuples containing 2 floats
        e.g. polynomial_coefs_min_max=( (-1,1), (-0.1,0.1), (-0.01,0.01) )
        the simulated leading effect of leading indicator X on (lagged) outcome Y is given by Y += aX + bX^2 + cX^3
        the coefficients a, b and c are randomly drawn for each simulated leading indicator from the range defined by polynomial_coefs_min_max=( (MIN(a),MAX(a)), (MIN(b),MAX(b)), (MIN(c),MAX(c)) )

    Returns
    ----------
    dict, numpy.array(), numpy.array()
        the first element is a dictionary describing the simulated relationships between the (leading) predictor variables (X) and the outcome (y)
        the second element is a 1-D numpy array containing the simulated outcome (y), of shape (n_time_points,)
        the third element is a 2-D numpy array, of shape (n_time_points, n_predictors)
    """
    y_vec = np.random.uniform(low=-10, high=10, size=n_time_points)
    X_matrix = np.random.uniform(low=-100, high=100, size=(n_time_points, n_predictors))
    simulated_effects_history_dict = {}
    for i in range(n_leading_indicator_effects):
        leading_indicator_idx = np.random.choice(range(n_predictors))
        if leading_indicator_idx not in simulated_effects_history_dict:
            simulated_effects_history_dict[leading_indicator_idx] = []
        effect_lag = np.random.randint(
            low=lagged_effect_time_min_max[0], high=lagged_effect_time_min_max[1]
        )
        a = np.random.uniform(
            low=polynomial_coefs_min_max[0][0], high=polynomial_coefs_min_max[0][1]
        )
        b = np.random.uniform(
            low=polynomial_coefs_min_max[1][0], high=polynomial_coefs_min_max[1][1]
        )
        c = np.random.uniform(
            low=polynomial_coefs_min_max[2][0], high=polynomial_coefs_min_max[2][1]
        )
        relevant_x = X_matrix[: n_time_points - effect_lag, leading_indicator_idx]
        y_vec += np.concatenate(
            [
                np.zeros(effect_lag),
                a * relevant_x + b * relevant_x**2 + c * relevant_x**3,
            ]
        )
        simulated_effects_history_dict[leading_indicator_idx].append(
            {
                "lag": effect_lag,
                "cubic_polynomial_coefs": [a, b, c],
            }
        )

    return simulated_effects_history_dict, y_vec, X_matrix


class leading_indicator_miner:
    """
    A class containing a model which searches data for leading indicator variables
    ...

    Attributes
    ----------
    n_leading_indicators : int
        number of leading indicators to keep track of
    best_leading_indicators_vars_set : list
        list storing information about the best [n_leading_indicators] found so far (sorted from worst to best)
    total_iterations_counter : int
        count of the total number of search iterations run ever i.e. summed over all of the time that the .fit() function was called
    best_mse_seen_in_training : int
        a record of the best Mean Squared Error on the training data seen on any iteration (over all training runs using the .fit() function)
    training_history : list
        a list (optional) containing full history of every leading indicator evaluated, and how it performed

    Methods
    -------
    list
        a list of strings representing the header columns

    Joe old Rubbish to clean up
    -------
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
    """

    def __init__(
        self,
        n_leading_indicators=2,  # number of leading indicators to identify
    ):
        """
        Parameters
        ----------
        n_leading_indicators : int
            the miner will maintain a set of the top [n_leading_indicators] found so far 
        """
        assert 1 == 1, "TODO: add an assertion here checking the input data types"
        self.n_leading_indicators = n_leading_indicators
        self.best_leading_indicators_vars_set = []
        self.total_iterations_counter = 0
        self.best_mse_seen_in_training = None
        self.training_history = []  # (optionally) keeps

    def fit(
        self,
        X,
        X_varnames,
        y,
        y_varname,
        n_iterations,
        n_lags_to_consider,  # e.g. n_lags_to_consider={"min":5,"max":10}
        n_knots={
            # optional: number of knots to use in the continuous piecewise linear spline fit (default is linear regression without splines)
            "min": 0,
            "max": 0,
        },
        keep_training_history=False,
    ):
        """
        TODO: nice documentation here
        (refer to https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)
        """
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
            lead size 3:
                        X	y
                        0	3
                        1	4
                        2   5
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

            if keep_training_history:
                self.training_history.append(
                    {
                        "leading_indicator_varname": predictor_varname_this_iter,
                        "mean_squared_error": train_mse,
                        "lag_n_time_periods": lag_this_iter,
                        "n_knots": n_knots_this_iter,
                    }
                )

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


if __name__ == "main":
    ## EXAMPLE USAGE ##
    help(leading_indicator_miner)
    import numpy as np

    raw_input_data = np.random.random((10_000, 50))

    leading_indicator_miner_model = leading_indicator_miner(
        n_knots={"min": 0, "max": 10},
        n_leading_indicators=5,
    )
    leading_indicator_miner_model.fit(
        X=raw_input_data[:, 0:49],
        X_varnames=[f"x{i}" for i in range(1, 50)],
        y=raw_input_data[:, 49],
        y_varname="y",
        n_lags_to_consider={"min": 5, "max": 10},
        n_iterations=1_000,
    )
    leading_indicator_miner_model.best_leading_indicators_vars_set
