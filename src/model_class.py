class leading_indicator_miner:
    """
    object for mining data in search of non-linear leading time series indicators (possibly multivariate)
        - currently only supports numeric indicator and numeric outcome variable

    goal of this code:
        * uses only numpy (no extra package dependencies)
        * is very opinionated
        * doesn't accept many different kinds of input (only numpy array). Only integer or float
        * doesn't try to do too much - does 1 thing well
        * has a sklearn-like interface (e.g. .fit(), .predict() etc.)
        * proper class documentation (i.e. go fix this Joe)

    -- EXAMPLE USAGE --
    help(leading_indicator_miner)
    import numpy as np
    raw_input_data = np.random.random( (20,4) )

    leading_indicator_miner_model = leading_indicator_miner(
        n_knots={"min":0,"max":10},
        n_predictors_to_combine={"min":1,"max":3},
    )
    leading_indicator_miner_model.fit(
        X = raw_input_data[:,0:3],
        X_varnames = ("x1","x2","x3"),
        y = raw_input_data[:,3],
        y_varname = "y",
        n_lags_to_consider={"min":5,"max":10},
        n_iterations = 4
    )
    leading_indicator_miner_model.best_model_so_far
    """

    def __init__(
        self,
        metric="MSE",
        n_knots={
            "min": 0,
            "max": 0,
        },  # optional: number of knots to use in the continuous piecewise linear spline fit (default is linear regression without splines)
        n_predictors_to_combine={
            "min": 1,
            "max": 1,
        },  # optional: largest number of predictor variables to use simultaneously to generate a prediction
    ):
        assert (
            1 == 1
        ), "joe you need to add an assertion here checking the input data types"

        self.metric = metric
        self.n_knots = n_knots
        self.n_predictors_to_combine = n_predictors_to_combine
        self.best_model_so_far = {
            "metric": self.metric,
            "metric_value_train_data": None,
            "lag_n_time_periods": None,
            "n_knots": None,
            "X_predictor_varnames": None,
            "knot_positions": None,
            "model_coefficients": None,
        }
        self.total_iterations_counter = 0

    def mean_squared_error(self, y_true, y_pred):
        return np.sum((y_pred - y_true) ** 2)

    def fit(
        self,
        X,
        X_varnames,
        y,
        y_varname,
        n_iterations,
        n_lags_to_consider,
        verbose=1,
    ):
        assert len(X[0]) == len(
            X_varnames
        ), "must have number of columns in X == len(X_varnames)"
        assert len(X) == len(
            y
        ), "X must contain the same number of time steps as y (same number of rows)"
        best_iter_so_far = None
        for i in range(n_iterations):
            self.total_iterations_counter += 1

            # choose random model hyperparameters for this iteration #
            lag_this_iter = np.random.randint(
                low=n_lags_to_consider["min"], high=n_lags_to_consider["max"]
            )
            n_predictors_this_iter = np.random.randint(
                low=self.n_predictors_to_combine["min"],
                high=self.n_predictors_to_combine["max"],
            )
            predictors_idx_this_iter = np.random.choice(
                range(0, len(X_varnames)),
                size=n_predictors_this_iter,
                replace=False,
            )
            predictors_idx_this_iter.sort()
            predictor_varnames_this_iter = [
                X_varnames[i] for i in predictors_idx_this_iter
            ]
            n_knots_this_iter = np.random.randint(
                low=self.n_knots["min"], high=self.n_knots["max"]
            )

            if verbose > 0:
                print(
                    f"""-- started iteration {i+1} (of {n_iterations}) --
                    lag size:   {lag_this_iter}
                    predictors: {predictor_varnames_this_iter}
                    n knots:    {n_knots_this_iter}"""
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
            # prepare X features for model
            # features are stored in a dictionary, with the variable name as key
            X_dict = {
                predictor_varnames_this_iter[i]: X[
                    0 : (len(X) - lag_this_iter), predictors_idx_this_iter[i]
                ]
                for i in range(len(predictors_idx_this_iter))
            }
            # add an intercept column to the predictors (X):
            X_dict["intercept"] = np.array([1] * (len(X) - lag_this_iter))

            # create spline features (trendline change features):
            X_splines_knot_ref_dict = {}
            for x_varname in predictor_varnames_this_iter:
                X_splines_knot_ref_dict[x_varname] = np.unique(
                    # only unique quantiles are kept as knot locations
                    np.quantile(
                        a=X_dict[x_varname],
                        q=[
                            1.0 / (n_knots_this_iter + 1) * i
                            for i in range(1, n_knots_this_iter + 1)
                        ],
                    )
                )
                quantile_counter = 0
                for quantile in X_splines_knot_ref_dict[x_varname]:
                    quantile_counter += 1
                    X_dict[f"{x_varname}_slopechange_{quantile_counter}"] = (
                        X_dict[x_varname] > quantile
                    ) * (X_dict[x_varname] - quantile)

            # create lead reponse vector:
            model_y = y[lag_this_iter:]

            # fit linear regression model:
            X_incl_splines = np.column_stack(list(X_dict.values()))
            xT_x = np.matmul(X_incl_splines.transpose(), X_incl_splines)
            xT_x_inv = np.linalg.inv(xT_x)
            beta_estimates = np.matmul(
                np.matmul(xT_x_inv, X_incl_splines.transpose()), model_y
            )
            fit_y = np.matmul(X_incl_splines, beta_estimates)

            # assess model fit to training data:
            train_mse = self.mean_squared_error(y_true=model_y, y_pred=fit_y)
            if (
                self.best_model_so_far["metric_value_train_data"] is None
                or train_mse < self.best_model_so_far["metric_value_train_data"]
            ):
                self.best_model_so_far["metric_value_train_data"] = train_mse
                self.best_model_so_far["lag_n_time_periods"] = lag_this_iter
                self.best_model_so_far["n_knots"] = n_knots_this_iter
                self.best_model_so_far[
                    "X_predictor_varnames"
                ] = predictor_varnames_this_iter
                self.best_model_so_far["knot_positions"] = X_splines_knot_ref_dict
                self.best_model_so_far["model_coefficients"] = {
                    list(X_dict.keys())[i]: beta_estimates[i]
                    for i in range(len(X_dict))
                }

            def predict(self, X, X_varnames):
                print(X)
                print(X_varnames)

            if verbose > 0:
                print(
                    f"""
                    {self.best_model_so_far['metric']}:   {train_mse:.5f} (best so far is {self.best_model_so_far["metric_value_train_data"]:.5f})
-- COMPLETED iteration {i+1} (of {n_iterations}) --
                """
                )
