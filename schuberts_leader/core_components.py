import numpy as np


def simulate_leading_indicator_data(
    n_time_points,
    n_predictors,
    n_leading_indicators,
    lagged_effect_time_min_max,
    n_y_breakpoints,
    y_sim_method,
    x_sim_method,
    noise_std_dev,
):
    """
    this function simulates multivariate feature data (X), in which some of the variables have a with a noisy time-lagged monotonic relationship with a simulated response/outcome variable (y)
    y is modelled either as a random gaussian walk starting at y=0 with each step Normal(0,1) or as independent draws from Normal(0,1)
    x variables are modelled directly from y if they are leading indicators, otherwise either as a random gaussian walk starting at x=0 with each step Normal(0,1) or as independent draws from Normal(0,1)
    random gaussian noise - randomly drawn from a Normal(0,[noise_std_dev]) distribution - is (optionally) added to the x variables to obscure the deterministic relationship with y

    Parameters
    ----------
    n_time_points : int
        number of time points to simulate
    n_predictors : int
        number of predictor variables to simulate
    n_leading_indicators : int
        number of predictors which are leading indicators of (i.e. predictive of) the response/outcome variable y
    lagged_effect_time_min_max : tuple containing 2 integers
        the shortest and longest allowable length of (simulated) lagged relationship between any leading indicator (x) and the outcome (y)
        example: lagged_effect_time_min_max=(1,19)
    n_y_breakpoints : int
        the monotonic relationship between x (leading indicator) and y (outcome) is a linear interpolation between randomly chosen monotone increasing (or decreasing) breakpoints
        this parameter [n_y_breakpoints] defines the number of breakpoints used
    y_sim_method : str
        method used to simulate the response/outcome variable (y)
        one of:
            "gaussian_random_walk": random gaussian walk starting at y=0 with each step a random draw from Normal(0,1)
            "independent_gaussian": each value an independent random draw from Normal(0,1)
    x_sim_method : str
        method used to simulate each feature (independent) variables (X)
        one of:
            "gaussian_random_walk": random gaussian walk starting at x=0 with each step a random draw from Normal(0,1)
            "independent_gaussian": each value is an independent random draw from Normal(0,1)
    noise_std_dev : float
        the standard deviation of the random gaussian noise added to leading indicators to obscure their deterministic relationship with outcome y

    Returns
    ----------
    dict, numpy.array(), numpy.array()
        the first returned element is a dictionary describing the simulated relationships between the (leading) predictor variables (X) and the outcome (y). The key in the dict is the index of the leading indicator in the matrix of predictors (X)
        the second returned element is a 1-D numpy array containing the simulated outcome (y), of shape (n_time_points,)
        the third returned element is a 2-D numpy array of shape (n_time_points, n_predictors)
    """

    assert y_sim_method in (
        ["independent_gaussian", "gaussian_random_walk"]
    ), "y_sim_method must be one of {'independent_gaussian','gaussian_random_walk'}"
    assert x_sim_method in (
        ["independent_gaussian", "gaussian_random_walk"]
    ), "x_sim_method must be one of {'independent_gaussian','gaussian_random_walk'}"

    leading_indicator_info_dict = {}
    leading_indicator_ind = np.random.choice(
        [0] * (n_predictors - n_leading_indicators) + [1] * n_leading_indicators,
        size=n_predictors,
        replace=False,
    )
    leading_effect_lags = (
        np.random.choice(
            range(lagged_effect_time_min_max[0], lagged_effect_time_min_max[1] + 1),
            size=n_predictors,
            replace=True,
        )
        * leading_indicator_ind
    )
    y_vec_extended = np.random.normal(
        # y_vec is padded on the end (in order to be able to generate the leading x variables)
        # these extra y values are discarded when returning the y vector
        loc=0,
        scale=1,
        size=n_time_points + leading_effect_lags.max(),
    )
    if y_sim_method == "gaussian_random_walk":
        y_vec_extended = y_vec_extended.cumsum()

    X_vectors_list = []
    for i in range(n_predictors):
        if leading_indicator_ind[i] == 0:  # variable has no leading relationship with y
            x = np.random.normal(
                loc=0,
                scale=2,
                size=n_time_points,
            )
            if x_sim_method == "gaussian_random_walk":
                x = x.cumsum()
            X_vectors_list.append(x)
            del x
        else:
            lead_lag_i = leading_effect_lags[i]
            y_lagged = y_vec_extended[lead_lag_i:]
            y_breakpoints = np.quantile(
                a=y_vec_extended,
                q=np.linspace(start=0, stop=1, num=n_y_breakpoints + 2),
            )
            x_breakpoints = np.concatenate(
                [[-20], np.random.uniform(low=-20, high=20, size=n_y_breakpoints), [20]]
            )
            relationship_asc_desc = np.random.choice(
                ["asc", "desc"]
            )  # decide whether monotonic relationship is ascending or descending
            if relationship_asc_desc == "asc":
                y_breakpoints.sort()
                x_breakpoints.sort()
                x_vec_val_list = []
                for y in y_lagged:
                    for j in range(len(y_breakpoints) - 1):
                        if y >= y_breakpoints[j] and y <= y_breakpoints[j + 1]:
                            pnt1_yx = (y_breakpoints[j], x_breakpoints[j])
                            pnt2_yx = (y_breakpoints[j + 1], x_breakpoints[j + 1])
                            x_vec_val_list.append(
                                # linear interpolation between point 1 and point 2
                                (
                                    pnt1_yx[1] * (pnt2_yx[0] - y)
                                    + pnt2_yx[1] * (y - pnt1_yx[0])
                                )
                                / (pnt2_yx[0] - pnt1_yx[0])
                            )
                X_vectors_list.append(
                    np.array(x_vec_val_list)[:n_time_points]
                    + np.random.normal(
                        loc=0,
                        scale=noise_std_dev,
                        size=n_time_points,  # add random gaussian noise
                    )
                )
                leading_indicator_info_dict[i] = {
                    "relationship_lag": lead_lag_i,
                }
            elif relationship_asc_desc == "desc":
                y_breakpoints.sort()
                x_breakpoints[::-1].sort()  # sort descending
                x_vec_val_list = []
                for y in y_lagged:
                    for j in range(len(y_breakpoints) - 1):
                        if y >= y_breakpoints[j] and y <= y_breakpoints[j + 1]:
                            pnt2_yx = (y_breakpoints[j], x_breakpoints[j])
                            pnt1_yx = (y_breakpoints[j + 1], x_breakpoints[j + 1])
                            x_vec_val_list.append(
                                # linear interpolation between point 1 and point 2
                                (
                                    pnt1_yx[1] * (pnt2_yx[0] - y)
                                    + pnt2_yx[1] * (y - pnt1_yx[0])
                                )
                                / (pnt2_yx[0] - pnt1_yx[0])
                            )
                X_vectors_list.append(
                    np.array(x_vec_val_list)[:n_time_points]
                    + np.random.normal(
                        loc=0,
                        scale=noise_std_dev,
                        size=n_time_points,  # add random gaussian noise
                    )
                )
                leading_indicator_info_dict[i] = {
                    "relationship_lag": lead_lag_i,
                }

    return (
        leading_indicator_info_dict,
        y_vec_extended[:n_time_points],
        np.column_stack(X_vectors_list),
    )


class leading_indicator_miner:
    """
    leading_indicator_miner is a model which searches data for leading indicators (variables with a leading correlation with outcome variable y)

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
        creates truncated power basis splines for a given single predictor variable (using user-provided knot points)
        these splines facilitate a piecewise continuous linear spline fit (i.e. degree=1 splines)
    estimate_OLS_linear_model_coefs
        calculates the Ordinary Least Squares (OLS) linear model coefficients for a given set of predictors (X) and response variable (y)
    generate_linear_model_preds
        calculates predictions using a trained OLS linear model, requiring only the trained model coefficients and the matrix of predictors
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
        """
        this method creates truncated power basis splines for a given single predictor variable (using user-provided knot points)
        these splines facilitate a piecewise continuous linear spline fit (i.e. degree=1 splines)

        Parameters
        ----------
        X_vec : np.array() (int or float)
            a 1-dimensional input variable numpy array (i.e. shape (n,) for some value n)
        knot_points_list : list (same dtype as X_vec)
            a list of knot points (slope change points) for the spline fit

        Returns
        ----------
        np.array()
            returns a numpy array containing the truncated power basis variables
            the shape of the returned array is (len(X_vec), len(knot_points_list))
        """
        assert (
            len(X_vec.shape) == 1
        ), "input [X_vec] must have shape (n,) i.e. is 1-dimensional"
        splines_list = []
        for knot in knot_points_list:
            splines_list.append((X_vec > knot) * (X_vec - knot))
        return np.column_stack(splines_list)

    def estimate_OLS_linear_model_coefs(self, X_matrix, y_vec):
        """
        this method calculates the Ordinary Least Squares (OLS) linear model coefficients for a given set of predictors (X) and response variable (y)
        note that perfect multicollinearity in the input [X_matrix] will lead to a singular matrix error

        Parameters
        ----------
        X_matrix : np.array() (int or float)
            numpy array of shape (n_samples, n_predictors)
        y_vec : np.array() (int or float)
            numpy array of shape (n_samples,)

        Returns
        ----------
        np.array()
            numpy array of shape (n_predictors,) containing the Ordinary Least Squares (OLS) linear model coefficients
        """
        xT_x = np.matmul(X_matrix.transpose(), X_matrix)
        xT_x_inv = np.linalg.inv(xT_x)
        return np.matmul(np.matmul(xT_x_inv, X_matrix.transpose()), y_vec)

    def generate_linear_model_preds(self, X_matrix, beta_coefs_vec):
        """
        this method calculates predictions using a trained OLS linear model, requiring only the trained model coefficients and the matrix of predictors

        Parameters
        ----------
        X_matrix : np.array() (int or float)
            matrix of predictors, of shape (n_prediction_samples, n_predictors)
        beta_coefs_vec : np.array() (int)
            numpy array of trained linear model coefficients, of shape (n_predictors)

        Returns
        ----------
        np.array() (float)
            returns numpy array of model predictions, of shape (n_prediction_samples,)
        """
        return np.matmul(X_matrix, beta_coefs_vec)

    def mean_squared_error(self, y_true, y_pred, ignore_nan=False):
        """
        this function ... TODO

        Parameters
        ----------
        X_matrix : np.array()
            TODO of shape ()
        y_vec : np.array()
            TODO of shape ()

        Returns
        ----------
        np.array()
            returns TODO of shape ()
        """
        if ignore_nan:
            return np.nanmean((y_pred - y_true) ** 2)
        else:
            return np.mean((y_pred - y_true) ** 2)

    def fit(
        self,
        X,
        X_varnames,
        y,
        y_varname,
        n_iterations,
        lags_to_consider,
        n_knots,
        knot_strategy,
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
        lags_to_consider : list
            TODO explanation here
            e.g. lags_to_consider=[1,3,6,12]
        n_knots : int
            TODO explanation here
        knot_strategy : str
            controls how the knot positions are decided
            one of {"quantiles","evenly_spaced"}
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
            lag_this_iter = np.random.choice(lags_to_consider)

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

            if knot_strategy == "quantiles":
                knot_locations_this_iter = np.unique(
                    # only unique quantiles are kept as knot locations
                    np.quantile(
                        a=x_vec_this_iter,
                        q=[1.0 / (n_knots + 1) * i for i in range(1, n_knots + 1)],
                    )
                )
            elif knot_strategy == "evenly_spaced":
                knot_locations_this_iter = np.linspace(
                    start=x_vec_this_iter.min(),
                    stop=x_vec_this_iter.max(),
                    num=n_knots + 2,
                )[1:-1]

            if n_knots > 0:
                spline_features_this_iter = self.create_linear_splines(
                    X_vec=x_vec_this_iter, knot_points_list=knot_locations_this_iter
                )
                x_matrix_this_iter = np.column_stack(
                    [intercept_var, x_vec_this_iter, spline_features_this_iter]
                )
            else:
                x_matrix_this_iter = np.column_stack([intercept_var, x_vec_this_iter])

            # this assertion will improve the user experience but is too computationally expensive (so I commented it out)
            # assert np.linalg.det(
            #    np.matmul(x_matrix_this_iter.transpose(), x_matrix_this_iter)
            # ), "X matrix contains perfect multicollinearity (reduce number of knots)"

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
                        "n_knots": n_knots,
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
            # ..then add it to [best_leading_indicators_vars_set]:
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
                        "n_knots": n_knots,
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
                        "n_knots": n_knots,
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
                        "n_knots": n_knots,
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
        a separate set of predictions is returned for each leading indicator in the best leading indicators set (self.best_leading_indicators_vars_set)
        note:
            * input [X] only needs to contain enough time points (rows) prior to the forecast period in order to generate a prediction (doesn't need to contain the whole training data)
            * input [X] only needs to contain the variables in the best leading indicators set (self.best_leading_indicators_vars_set)

        Parameters
        ----------
        X : np.array() of floats
            Numpy array with each row a consecutive time point and each column a predictor
        X_varnames : list of str
            List (of strings) of length X.shape[1] containing the names of the columns (variables) in input [X] - these are used

        Returns
        ----------
        np.array() of floats
            returns a numpy array of predictions, where each column of the array is the series of predictions from one of the leading indicators in the best leading indicators set
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
