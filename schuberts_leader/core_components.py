import numpy as np


def simulate_leading_indicator_data(
    n_time_points,
    n_predictors,
    n_leading_indicators,
    lagged_effect_time_min_max,
    n_y_breakpoints,
):
    """
    this function simulates multivariate data containing leading indicators with a noisy time-lagged monotonic relationship with the response variable (y)
    y is modelled as a random gaussian walk starting at y=0, with standard deviation 100
    x variables are modelled directly from y if they are leading indicators, otherwise as random gaussian walks (the same as y)

    Parameters
    ----------
    n_time_points : int
        number of time points to simulate
    n_predictors : int
        number of predictor variables to simulate
    n_leading_indicators : int
        number of predictors which are leading indicators of the outcome y
    lagged_effect_time_min_max : tuple containing 2 integers
        e.g. lagged_effect_time_min_max=(1,19)
        the smallest and largest time between a leading indicator (x) value and it's relationship with the outcome (y)
    n_y_breakpoints : int
        the monotonic relationship between x (leading indicator) and y (outcome) is a linear interpolation
        between randomly chosen monotone increasing (or decreasing) breakpoints
        this parameter defines the number of breakpoints

    Returns
    ----------
    dict, numpy.array(), numpy.array()
        the first element is a dictionary describing the simulated relationships between the (leading) predictor variables (X) and the outcome (y)
        the second element is a 1-D numpy array containing the simulated outcome (y), of shape (n_time_points,)
        the third element is a 2-D numpy array, of shape (n_time_points, n_predictors)
    """
    leading_effect_lags = np.random.choice(
        range(lagged_effect_time_min_max[0], lagged_effect_time_min_max[1] + 1),
        size=n_leading_indicators,
        replace=True,
    )
    y_vec_extended = np.random.normal(
        # y_vec is padded on the end (in order to be able to generate the leading x variables)
        # these extra y values are removed before returning the y vector
        loc=0,
        scale=100,
        size=n_time_points + leading_effect_lags.max(),
    )
    leading_indicator_ind = np.random.choice(
        [0] * (n_predictors - n_leading_indicators) + [1] * n_leading_indicators,
        size=n_predictors,
        replace=False,
    )
    X_vectors_list = []
    for i in range(n_predictors):
        if leading_indicator_ind[i] == 0:
            X_vectors_list.append(
                np.random.normal(
                    loc=0,
                    scale=100,
                    size=n_time_points,
                )
            )
        else:
            y_breakpoints = np.random.uniform(low=,high=,size=n_y_breakpoints)
            relationship_asc_desc = np.random.choice(
                ["asc", "desc"]
            )  # decide whether monotonic relationship is ascending or descending
            if relationship_asc_desc == "asc":
                y_breakpoints.sort()
            elif relationship_asc_desc=="desc":
                pass

def OLD_simulate_leading_indicator_data(
    n_time_points,
    n_predictors,
    n_leading_indicator_effects,
    lagged_effect_time_min_max,
    polynomial_coefs_min_max,
):
    """
    this function simulates data containing leading indicators with a noisy time-lagged cubic polynomial effect on the response variable (y)
    note that the same leading indicator can have multiple effects on the outcome variable Y (at different time lags)
    note that all X variables are populated as a cumulative sum of random independent draws from a uniform distribution on [-100,100]
    y is initialized as a cumulative sum of random independent draws from a uniform distribution on [-10,10]

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
    y_vec = np.random.uniform(low=-10, high=10, size=n_time_points).cumsum()
    X_matrix = np.random.uniform(
        low=-100, high=100, size=(n_time_points, n_predictors)
    ).cumsum(axis=0)
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
    mse_history : list
        TODO explanation here

    Methods
    -------
    create_linear_splines
        TODO explanation here
    estimate_OLS_linear_model_coefs
        TODO explanation here
    generate_linear_model_preds
        TODO explanation here
    mean_squared_error
        TODO explanation here
    fit
        TODO explanation here
    predict
        TODO explanation here
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
        self.training_history = []
        self.mse_history = []

    def create_linear_splines(self, X_vec, knot_points_list):
        """TODO: needs some documentation (see https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)"""
        splines_list = []
        for knot in knot_points_list:
            splines_list.append((X_vec > knot) * (X_vec - knot))
        return np.column_stack(splines_list)

    def estimate_OLS_linear_model_coefs(self, X_matrix, y_vec):
        """TODO: needs some documentation (see https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)"""
        xT_x = np.matmul(X_matrix.transpose(), X_matrix)
        xT_x_inv = np.linalg.inv(xT_x)
        return np.matmul(np.matmul(xT_x_inv, X_matrix.transpose()), y_vec)

    def generate_linear_model_preds(self, X_matrix, beta_coefs_vec):
        """TODO: needs some documentation (see https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)"""
        return np.matmul(X_matrix, beta_coefs_vec)

    def mean_squared_error(self, y_true, y_pred):
        """TODO: needs some documentation (see https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings)"""
        return np.mean((y_pred - y_true) ** 2)

    def fit(
        self,
        X,
        X_varnames,
        y,
        y_varname,
        n_iterations,
        n_lags_to_consider,  # e.g. n_lags_to_consider={"min":5,"max":10}
        n_knots_to_consider={
            # optional: number of knots to use in the continuous piecewise linear spline fit (default is linear regression without splines)
            "min": 0,
            "max": 0,
        },
        keep_training_history=False,
        verbose=1,
    ):
        """
        Parameters
        ----------
        X : numpy array of shape (?,?) (float or int)
            TODO explanation here
        X_varnames : list of str
            TODO explanation here
        y : numpy array of shape (?,?) (float or int)
            TODO explanation here
        y_varname : str
            TODO explanation here
        n_iterations : int
            TODO explanation here
        n_lags_to_consider : dict
            TODO explanation here
        n_knots_to_consider : dict
            TODO explanation here
        keep_training_history : bool
            TODO explanation here
        verbose : int
            TODO explanation here

        Returns
        ----------
        None
            .fit() updates attributes of the parent class (leading_indicator_miner) and (optionally) prints logging information while it runs, but does not explicitly return anything
        """
        assert X.shape[1] == len(X_varnames), "must have X.shape[1] == len(X_varnames)"
        assert X.shape[0] == len(
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
            if n_knots_to_consider["min"] == n_knots_to_consider["max"]:
                n_knots_this_iter = n_knots_to_consider["min"]
            else:
                n_knots_this_iter = np.random.randint(
                    low=n_knots_to_consider["min"],
                    high=n_knots_to_consider["max"],
                )

            # illustration of how features are lagged:
            """
            time:       0 1 2 3 4 5 6 7 8 9 10
            lag size 3:
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
            x_vec_this_iter = X[: (len(X) - lag_this_iter), predictor_idx_this_iter]
            y_vec_this_iter = y[lag_this_iter:]
            intercept_var = np.ones(len(x_vec_this_iter))
            # create spline features (trendline change features):

            knot_locations_this_iter = np.unique(
                # only unique quantiles are kept as knot locations
                np.quantile(
                    a=x_vec_this_iter,
                    q=[
                        1.0 / (n_knots_this_iter + 1) * i
                        for i in range(1, n_knots_this_iter + 1)
                    ],
                )
            )
            if n_knots_this_iter > 0:
                spline_features_this_iter = self.create_linear_splines(
                    X_vec=x_vec_this_iter, knot_points_list=knot_locations_this_iter
                )
                x_matrix_this_iter = np.column_stack(
                    [intercept_var, x_vec_this_iter, spline_features_this_iter]
                )
            else:
                x_matrix_this_iter = np.column_stack([intercept_var, x_vec_this_iter])

            beta_coef_this_iter = self.estimate_OLS_linear_model_coefs(
                X_matrix=x_matrix_this_iter,
                y_vec=y_vec_this_iter,
            )
            fit_y = self.generate_linear_model_preds(
                X_matrix=x_matrix_this_iter,
                beta_coefs_vec=beta_coef_this_iter,
            )

            # assess model fit to training data:
            train_mse_this_iter = self.mean_squared_error(
                y_true=y_vec_this_iter, y_pred=fit_y
            )

            if keep_training_history:
                self.training_history.append(
                    {
                        "leading_indicator_varname": predictor_varname_this_iter,
                        "mean_squared_error": train_mse_this_iter,
                        "lag_n_time_periods": lag_this_iter,
                        "n_knots": n_knots_this_iter,
                    }
                )

            if (
                self.best_mse_seen_in_training is None
                or self.best_mse_seen_in_training > train_mse_this_iter
            ):
                self.best_mse_seen_in_training = train_mse_this_iter
                self.mse_history.append(
                    (self.total_iterations_counter, self.best_mse_seen_in_training)
                )

            leading_indicator_varnames_currently_in_best_set = [
                x["leading_indicator_varname"]
                for x in self.best_leading_indicators_vars_set
            ]

            # if we haven't got [n_leading_indicators] yet..
            # ..and this variable is not already in [best_leading_indicators_vars_set]..
            # ..then add this one in:
            if (
                len(self.best_leading_indicators_vars_set) < self.n_leading_indicators
                and predictor_varname_this_iter
                not in leading_indicator_varnames_currently_in_best_set
            ):
                self.best_leading_indicators_vars_set.append(
                    {
                        "leading_indicator_varname": predictor_varname_this_iter,
                        "mean_squared_error": train_mse_this_iter,
                        "lag_n_time_periods": lag_this_iter,
                        "n_knots": n_knots_this_iter,
                        "knot_locations": knot_locations_this_iter,
                        "beta_coefs": beta_coef_this_iter,
                    }
                )
            elif (
                # if we already have [n_leading_indicators]..
                # ..and the current leading indicator is better than the worst one in [best_leading_indicators_vars_set]..
                # ..then replace the worst one in [best_leading_indicators_vars_set] with the current leading indicator:
                len(self.best_leading_indicators_vars_set) == self.n_leading_indicators
                and train_mse_this_iter
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
                        "mean_squared_error": train_mse_this_iter,
                        "lag_n_time_periods": lag_this_iter,
                        "n_knots": n_knots_this_iter,
                        "knot_locations": knot_locations_this_iter,
                        "beta_coefs": beta_coef_this_iter,
                    }

                else:
                    # if the current leading indicator is not yet in the [best_leading_indicators_vars_set]..
                    # ..then insert it (replacing the worst one already in the set):
                    self.best_leading_indicators_vars_set[0] = {
                        "leading_indicator_varname": predictor_varname_this_iter,
                        "mean_squared_error": train_mse_this_iter,
                        "lag_n_time_periods": lag_this_iter,
                        "n_knots": n_knots_this_iter,
                        "knot_locations": knot_locations_this_iter,
                        "beta_coefs": beta_coef_this_iter,
                    }

            # sort best leading indicator list by Mean Squared Error (worst to best):
            self.best_leading_indicators_vars_set = sorted(
                self.best_leading_indicators_vars_set,
                key=lambda d: -d["mean_squared_error"],
            )

            if verbose > 0:
                print(
                    f"\riteration {(i+1):,} of {n_iterations:,}. best MSE: {self.best_mse_seen_in_training:,.3f}",
                    end="",
                    flush=True,
                )

    def predict(self, X, X_varnames):
        """
        returns future predictions using the best leading indicators set discovered during .fit()
        a separate set of predictions is returned for each leading

        Parameters
        ----------
        X : np.array(), float
            Numpy array with each row a consecutive time point and each column a predictor
        X_varnames : list, str
            List (of strings) of length X.shape[1] containing the names of the columns (variables) in X

        Returns
        ----------
        np.array(), float
            returns a numpy array of predictions, where each column of the array is the series of predictions from on of the leading indicators
            since the number of predictions differs per leading indicator (due to the length of the leading effect), the array has shape (largest_lead_effect_time_amongst_leading_indicators, n_leading_indicators)
        """
        max_lead_time = max(
            [x["lag_n_time_periods"] for x in self.best_leading_indicators_vars_set]
        )
        assert (
            len(X) >= max_lead_time
        ), f"predictor matrix X must contain at least {max_lead_time} time points (the largest leading effect length in the set of best leading indicators) prior to the prediction period"

        preds_list = []
        for x in self.best_leading_indicators_vars_set:
            x_varname = x["leading_indicator_varname"]
            x_lead_time = x["lag_n_time_periods"]
            x_idx = X_varnames.index(x_varname)
            x_values = X[(len(X) - x_lead_time) :, x_idx]

            # generate predictions #
            intercept_var = np.ones(len(x_values))
            # create spline features (trendline change features):
            n_knots = x["n_knots"]
            knot_locations = x["knot_locations"]
            if n_knots > 0:
                spline_features = self.create_linear_splines(
                    X_vec=x_values, knot_points_list=knot_locations
                )
                x_matrix = np.column_stack([intercept_var, x_values, spline_features])
            else:
                x_matrix = np.column_stack([intercept_var, x_values])

            pred_y = self.generate_linear_model_preds(
                X_matrix=x_matrix,
                beta_coefs_vec=x["beta_coefs"],
            )
            preds_list.append(pred_y)

        # pad all of the prediction vectors to be the same length #
        longest_preds_series_length = max([len(y) for y in preds_list])
        for i in range(len(preds_list)):
            this_preds_series_length = len(preds_list[i])
            if this_preds_series_length < longest_preds_series_length:
                preds_list[i] = np.concatenate(
                    [
                        preds_list[i],
                        (
                            np.empty(
                                longest_preds_series_length - this_preds_series_length
                            )
                            * np.nan
                        ),
                    ]
                )

        return np.column_stack(preds_list)
