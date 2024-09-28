"""
Functions for the models recalibration
"""
# Authors: Sara Cupini, Davide Pagani, Francesco Panichi
# License: Apache-2.0 license
# Notice: these functions were completely developed by us, but rely on the structure built by Alessandro Brusaferri

import os
import pandas as pd
import numpy as np
import json

os.environ["TF_USE_LEGACY_KERAS"] = "1"
from tools.PrTSF_Recalib_tools import PrTsfRecalibEngine, load_data_model_configs
from tools.prediction_quantiles_tools import plot_quantiles
from tools.models.SARIMAX import SARIMAXRegressor
from datetime import datetime
from sklearn.metrics import root_mean_squared_error as rmse, mean_squared_error as mse, mean_absolute_error as mae
from scipy.optimize import minimize
from tools.conformal_prediction import compute_cp


def load_json(file_path):
    """ Load json file """
    with open(file_path, 'r') as file:
        return json.load(file)


def save_json(data, file_path):
    """ Save json file """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def modify_json(path, file_name, start_train_set, start_test_set, end_test_set):
    """ Modify json file """
    config = load_json(path)

    # From string to datetime
    date_start_train_set = datetime.strptime(start_train_set, '%Y-%m-%d')
    # From datetime to a numerical array
    start_train_set = [date_start_train_set.year, date_start_train_set.month, date_start_train_set.day]

    # From string to datetime
    date_start_test_set = datetime.strptime(start_test_set, '%Y-%m-%d')
    # From datetime to a numerical array
    start_test_set = [date_start_test_set.year, date_start_test_set.month, date_start_test_set.day]

    # From string to datetime
    date_end_test_set = datetime.strptime(end_test_set, '%Y-%m-%d')
    # From datetime to a numerical array
    end_test_set = [date_end_test_set.year, date_end_test_set.month, date_end_test_set.day]

    if 'data_config' in config:
        config['data_config']['pred_horiz'] = 24
        config['data_config']['num_vali_samples'] = 100
        config['data_config']['idx_start_train']['y'] = start_train_set[0]
        config['data_config']['idx_start_train']['m'] = start_train_set[1]
        config['data_config']['idx_start_train']['d'] = start_train_set[2]
        config['data_config']['idx_start_oos_preds']['y'] = start_test_set[0]
        config['data_config']['idx_start_oos_preds']['m'] = start_test_set[1]
        config['data_config']['idx_start_oos_preds']['d'] = start_test_set[2]
        config['data_config']['idx_end_oos_preds']['y'] = end_test_set[0]
        config['data_config']['idx_end_oos_preds']['m'] = end_test_set[1]
        config['data_config']['idx_end_oos_preds']['d'] = end_test_set[2]
        config['data_config']['dataset_name'] = file_name

    # Save the modified JSON back to the file
    save_json(config, path)


def recalibration(PF_task_name: str, exper_setup: str, hyper_mode: str, file_name: str,
                  dates: dict):
    """"
    Function to run the recalibration experiments.
    The available models are point-ARX, point-TARX and point-DNN.

    INPUT:
    PF_task_name: name of the task
    exper_setup: name of the exper
    hyper_mode: name of the hyper-mode (load_tuned or optuna_tuner)
    file_name: name of the dataset file
    dates: dictionary of dates composed by:
          start_train: date of the start of training set (YYYY-MM-DD)
          start_test: date of the start of test set (YYYY-MM-DD)
          end_test: date of the end of test set (YYYY-MM-DD)

    OUTPUT:
    test_prediction: Dataframe with predicted and true values

    """
    print('-' * 70)
    print('Starting recalibration of config: ' + exper_setup)
    print('-' * 70)

    # Extract data from dictionaries
    start_train = dates['start_train']
    start_test = dates['start_test']
    end_test = dates['end_test']

    # Set run configs
    run_id = 'recalib_opt_grid_1_1'

    # Plot train history flag
    plot_train_history = False
    plot_weights = False

    # Load the current JSON configuration
    path = os.path.join(os.getcwd(), 'experiments', 'tasks', PF_task_name, exper_setup, run_id, 'exper_configs.json')

    # Modify JSON with our params
    modify_json(path, file_name, start_train, start_test, end_test)

    # Load experiments configuration from json file
    configs = load_data_model_configs(task_name=PF_task_name, exper_setup=exper_setup, run_id=run_id)

    # Load dataset
    dir_path = os.getcwd()
    ds = pd.read_csv(os.path.join(dir_path, 'data', 'datasets', configs['data_config'].dataset_name))
    ds.set_index(ds.columns[0], inplace=True)

    # Instantiate recalibratione engine
    PrTSF_eng = PrTsfRecalibEngine(dataset=ds,
                                   data_configs=configs['data_config'],
                                   model_configs=configs['model_config'])

    # Get model hyperparameters (previously saved or by tuning)
    model_hyperparams = PrTSF_eng.get_model_hyperparams(method=hyper_mode, optuna_m=configs['model_config']['optuna_m'])

    # Exec recalib loop over the test_set samples, using the tuned hyperparams
    test_predictions = PrTSF_eng.run_recalibration(model_hyperparams=model_hyperparams,
                                                   plot_history=plot_train_history,
                                                   plot_weights=plot_weights)

    # Plot test predictions
    plot_quantiles(test_predictions, target=PF_task_name)

    return test_predictions


def recalibration_sarimax(exper_setup: str, file_path: str, dates: dict, spike_preproc_flag: bool, order: tuple,
                          seasonal_order=(0, 0, 0, 0)):
    """
    Import settings from the main script, create the sarimax regressor object according to the orders specified in
    input and run the recalibration for each day in dates, then return the test_predictions.
    Has also a flag to specify if the spike preprocessing on the target variable needs to be performed.
    The available models are point-ARIMAX, point-P-ARIMAX, point-SARIMAX and point-P-SARIMAX.

    INPUT:
    exper_setup: name of the exper
    file_path: path of the dataset file
    dates: dictionary of dates composed by:
          start_train: date of the start of training set (YYYY-MM-DD)
          start_test: date of the start of test set (YYYY-MM-DD)
          end_test: date of the end of test set (YYYY-MM-DD)
    spike_preproc_flag: flag to specify if the spike preprocessing on the target variable needs to be performed.
    order: tuple of the orders specified
    seasonal_order: tuple of the seasonal orders specified

    OUTPUT:
    test_prediction: Dataframe with predicted and true values
    """
    print('-' * 70)
    if spike_preproc_flag is True:
        print('Starting recalibration of config: ' + exper_setup[0:6]+'P-'+exper_setup[6:])
    else:
        print('Starting recalibration of config: ' + exper_setup)
    print('-' * 70)

    # Create a dictionary of settings for the SARIMAX object
    settings = {'file_path': file_path,
                'PF_method': exper_setup[:5],
                'model_class': exper_setup[6:],
                'pred_horiz': 24,
                'order': order,
                'seasonal_order': seasonal_order,
                'spike_preproc': spike_preproc_flag}

    # Create model object
    regressor = SARIMAXRegressor(file_path=file_path, settings=settings, dates=dates)

    # Obtain the train and test datasets (saved in a "sets block" which is a list, containing:
    # x_train, x_test, y_train, y_test in this order --> this a structural decision, it's a fixed order)
    # and also the list of scalers used for the target variable, hour per hour
    sets_block = regressor.get_train_test_set_from_dataset()

    # Obtain test predictions from the model
    test_predictions = regressor.get_test_predictions(sets_block=sets_block, order=order, seasonal_order=seasonal_order)

    # Return the test predictions
    return test_predictions


class MixedEnsemble:
    """
    Class to be used to create mixed ensemble point predictions
    """

    def __init__(self, true_prices: np.array, preds_list: list, ensemble_method='simple_AVG', num_cali_samples=28):

        # Initialize some needed attributes of the class
        self.ensemble_size = len(preds_list)  # number of point prediction models used for our ensemble
        self.predictions_length = len(preds_list[0])  # number of hourly predictions of each model
        self.pred_horiz = 24  # prediction horizon = 24 hours in a day
        self.num_days = int(self.predictions_length / self.pred_horiz)  # number of days predicted by each model
        self.num_cali_samples = num_cali_samples  # size of ensemble calibratrion bag

        # Use the input prices and predictions to create hourly matrices
        self.true_hourly, self.pred_hourly = (
            self.assemble_hourly_matrices(true_prices=true_prices, preds_list=preds_list))

        # Store the type of ensemble method to be used
        self.ensemble_method = ensemble_method

    def assemble_hourly_matrices(self, true_prices: np.array, preds_list: list):
        """
        Assemble hourly matrices from input variables, to be used for ensemble method

        INPUT
        true_prices: array of true prices
        preds_list: array of predicted prices

        OUTPUT
        true_hourly: assembled hourly matrices for true value
        pred_hourly: assembled hourly matrices for predicted value

        """
        # Initialize matrix of true prices as: hour x day
        true_hourly = np.zeros((self.pred_horiz, self.num_days))
        # Initialize matrix of predictions as: ensemble_number x hour x day
        pred_hourly = np.zeros((self.ensemble_size, self.pred_horiz, self.num_days))

        # Arrange the true values in an hourly fashion
        for i in range(self.pred_horiz):  # loop over the hours
            for j in range(self.num_days):  # loop over the days
                true_hourly[i, j] = true_prices[self.pred_horiz * j + i]

        # Arrange the predicted values in an hourly fashion
        for ensemble in range(self.ensemble_size):  # loop over the ensemble models

            # Transform to numpy the current model's vector of predictions
            vector_pred = preds_list[ensemble].to_numpy()

            for i in range(self.pred_horiz):  # loop over the hours
                for j in range(self.num_days):  # loop over the days
                    pred_hourly[ensemble, i, j] = vector_pred[self.pred_horiz * j + i]

        return true_hourly, pred_hourly

    def combine_predictions(self, true_prices: np.array):
        """
        Start the actual ensemble technique, combining the predictions according to the method chosen

        INPUT
        true_prices: array of true prices

        OUTPUT
        test_prediction: Dataframe with predicted and true values
        """

        print('-' * 70)
        print('Combining predictions with ensemble method ' + self.ensemble_method)
        print('-' * 70)

        # Evaluate which method was chosen and call the appropriate function
        if self.ensemble_method == 'simple_AVG':
            preds_ensemble = self.combine_preds_simple_averaging()

        elif self.ensemble_method == 'IRMSE':
            preds_ensemble = self.combine_preds_irmse()

        elif self.ensemble_method == 'const_LAD':
            preds_ensemble = self.combine_preds_const_lad_regression()

        else:
            print('Specified invalid ensemble method: please select a valid method')

        # Reshape the predictions in a flattened way, as original
        preds_ensemble = preds_ensemble.flatten('F')

        # Combine the predictions in the final dataframe (removing first unneeded values of true prices)
        test_predictions = pd.DataFrame({0.5: preds_ensemble,
                                         'EM_price': true_prices[self.pred_horiz * self.num_cali_samples:]})

        return test_predictions

    def combine_preds_simple_averaging(self):
        """
        Combine the predictions using simple average

        OUTPUT
        new_preds: predictions using simple average ensemble method
        """

        # Weights are constant and equally divided
        new_preds = np.sum(self.pred_hourly / self.ensemble_size, axis=0)[:, self.num_cali_samples:]

        return new_preds

    def combine_preds_irmse(self):
        """
        Combine the predictions using inverse root mean squared error

        OUTPUT:
        new_preds: predictions using inverse root mean squared error ensemble method
        """

        # Start organizing the prediction, we will have weights in a sliding window manner
        weights = np.zeros((self.ensemble_size, self.pred_horiz, self.num_days - self.num_cali_samples))
        new_preds = np.zeros((self.pred_horiz, self.num_days - self.num_cali_samples))

        for i in range(self.num_days - self.num_cali_samples):  # loop over days

            # Select the regressors and the target variable
            preds = self.pred_hourly[:, :, i:self.num_cali_samples + i]
            trues = self.true_hourly[:, i:self.num_cali_samples + i]

            for hour in range(self.pred_horiz):  # loop over the hours

                # Set current median pred
                pred_curr = preds[:, hour].reshape(-1, self.ensemble_size)
                true_curr = trues[hour]

                rmse_models = np.zeros((self.ensemble_size,))

                for model in range(self.ensemble_size):  # loop over ensemble models
                    rmse_models[model] = 1 / rmse(pred_curr[:, model], true_curr)

                # Compute weights with the formula (Nowotarski et al., 2014)
                weights[:, hour, i] = rmse_models / np.sum(rmse_models)

                # Save prediction
                x_futu = self.pred_hourly[:, hour, self.num_cali_samples + i].reshape(-1, self.ensemble_size)
                new_preds[hour, i] = np.dot(x_futu, weights[:, hour, i])[0]

            print('Combining predictions of day: ' + str(i + 1) + '/' + str(self.num_days - self.num_cali_samples))

        return new_preds

    def combine_preds_const_lad_regression(self):
        """
        Combine the predictions using constrained least absolute deviation regression

        OUTPUT
        new_preds: predictions using constrained least absolute deviation regression ensemble method
        """

        # Start organizing the prediction, we will have weights in a sliding window manner
        weights = np.zeros((self.ensemble_size, self.pred_horiz, self.num_days - self.num_cali_samples))
        new_preds = np.zeros((self.pred_horiz, self.num_days - self.num_cali_samples))

        # Define the MAE loss function
        def mae_loss(theta, X, y):
            return np.sum(np.abs(X.dot(theta) - y))

        for i in range(self.num_days - self.num_cali_samples):

            # Select the regressors and the target variable
            X = self.pred_hourly[:, :, i:self.num_cali_samples + i]
            Y = self.true_hourly[:, i:self.num_cali_samples + i]

            for hour in range(self.pred_horiz):  # loop over the hours

                # Take current regressor and target variable
                x_curr = X[:, hour].reshape(-1, self.ensemble_size)
                y_curr = Y[hour]

                # Initial guess for the weights
                rand_theta = np.random.randn(self.ensemble_size)
                rand_theta = rand_theta / np.sum(rand_theta)  # Normalize to ensure they sum to 1

                # Initial guess for the weights
                initial_theta = weights[:, hour, i - 1] if i > 0 else rand_theta

                # Constraints:
                constraints = [
                    {'type': 'ineq', 'fun': lambda theta: theta},  # All weights should be >= 0
                    {'type': 'eq', 'fun': lambda theta: np.sum(theta) - 1}  # Weights should sum to 1
                ]

                # Minimize the MAE loss function with constraints
                result = minimize(mae_loss, initial_theta, args=(x_curr, y_curr), constraints=constraints,
                                  method='SLSQP')

                # Save calibrated weights
                weights[:, hour, i] = result.x

                # Predict new time stamp
                x_futu = self.pred_hourly[:, hour, self.num_cali_samples + i].reshape(-1, self.ensemble_size)
                # Save prediction
                new_preds[hour, i] = np.dot(x_futu, weights[:, hour, i])[0]

            print('Combining predictions of day: ' + str(i + 1) + '/' + str(self.num_days - self.num_cali_samples))

        return new_preds


def create_intervals_cp(test_predictions: pd.DataFrame, final_test_length: int):
    """
        Creates prediction intervals from point predictions, using the conformal prediction algorithm

        INPUT
        test_predictions: predictions using conformal prediction algorithm
        final_test_length: length of prediction intervals

        OUTPUT
        interval_test_preds: prediction intervals
    """

    # Initialize needed variables
    pred_horiz = 24
    task_name = 'EM_price'

    # Conformal Prediction settings: confidence levels and number of calibration samples
    cp_settings = {'target_alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]}
    num_cali_samples = int(len(test_predictions) / pred_horiz) - final_test_length

    # Build the settings to build intervals from point using CP
    cp_settings['pred_horiz'] = pred_horiz
    cp_settings['task_name'] = task_name
    cp_settings['num_cali_samples'] = num_cali_samples

    # Execute conformal prediction
    interval_test_preds = compute_cp(test_predictions, cp_settings)

    return interval_test_preds


def compute_test_metrics(test_predictions: pd.DataFrame):
    """
    Compute final evaluation of model. Inside, the point predictions become interval prediction thanks to
    conformal prediction algorithm

    INPUT
    test_predictions: predictions using conformal prediction algorithm

    """
    # Initialize needed variables
    pred_horiz = 24
    task_name = 'EM_price'

    # Extract the true and predicted values (after the CP, so in case of point predictions, we have removed
    # from the test set the values that were present only to be used in the calibration bag --> now good size
    # True values
    y = test_predictions.loc[:, task_name]
    # Predicted median values
    y_hat = test_predictions.loc[:, 0.5]

    # Print point prediction metrics: MSE, MAE, sMAPE
    print('------------------------------------------------------')
    print('Evaluation of final model')
    print('------------------------------------------------------')
    print('--------------- Point Prediction Metrics -------------')
    print('MSE:', mse(y, y_hat))
    print('MAE:', mae(y, y_hat))
    print('RMSE:', rmse(y, y_hat))
    print('sMAPE:', sMAPE(y, y_hat))

    # Set variable needed for the interval forecasting metrics
    quantiles_levels = test_predictions.columns[1:].tolist()

    # Compute Delta Coverage
    alpha_min = 0.9
    alpha_max = 0.99
    delta_scores = compute_delta_coverage(y_true=y,
                                          pred_quantiles=test_predictions.loc[:,
                                                         test_predictions.columns != task_name],
                                          quantiles_levels=quantiles_levels, alpha_min=alpha_min, alpha_max=alpha_max)

    # Compute Winkler scores
    winkler_scores = compute_winkler_scores(y_true=y.to_numpy().reshape(-1, pred_horiz),
                                            pred_quantiles=test_predictions.loc[:,
                                                           test_predictions.columns != task_name].
                                            to_numpy().reshape(-1, pred_horiz, len(quantiles_levels)),
                                            quantiles_levels=quantiles_levels)
    # Compute pinball scores
    pinball_scores = compute_pinball_scores(y_true=y.to_numpy().reshape(-1, pred_horiz),
                                            pred_quantiles=test_predictions.loc[:,
                                                           test_predictions.columns != task_name].
                                            to_numpy().reshape(-1, pred_horiz, len(quantiles_levels)),
                                            quantiles_levels=quantiles_levels)

    print('--------------- Interval Prediction Metrics ----------')
    print('Delta Coverage:', delta_scores)
    print('Winkler Score:', winkler_scores.mean().mean())
    print('Pinball Loss:', pinball_scores.mean().mean())


def compute_pinball_scores(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the pinball score on the test results
    return: pinball scores computed for each quantile level and each step in the pred horizon

    INPUT
    y_true: test results
    pred_quantiles: prediction intervals
    quantiles_levels: quantile levels

    OUTPUT
    score: pinball score
    """

    score = []
    for i, q in enumerate(quantiles_levels):
        error = np.subtract(y_true, pred_quantiles[:, :, i])
        loss_q = np.maximum(q * error, (q - 1) * error)  #
        score.append(np.expand_dims(loss_q, -1))
    score = np.mean(np.concatenate(score, axis=-1), axis=0)
    return score


def compute_winkler_scores(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the Winkler's score on the test results
    return: Winkler's scores computed for each quantile level and each step in the pred horizon

    INPUT
    y_true: test results
    pred_quantiles: prediction intervals
    quantiles_levels: quantile levels

    OUTPUT
    score: Winkler score
    """

    score = []

    # Variable with how many quantiles I have (expect the median)
    amount_quantiles = len(quantiles_levels) - 1
    # for example if I have 21 quantiles, it's 20, because we do not care about the median

    for i, q in enumerate(quantiles_levels[:int(amount_quantiles / 2)]):
        # The confidence level is now q; the corresponding 1-q confidence interval
        # is constituted by: lower bound is the i-th element of the pred_quantiles,
        # upper bound is the (amount_quantiles - i)-th element

        # Take all the lower and upper bounds of the prediction interval of level q
        lower_bound = pred_quantiles[:, :, i]
        upper_bound = pred_quantiles[:, :, -1 - i]  # -1-i to take elements from the back

        # Computed delta_N for each of the empirical prediction intervals
        deltaN = np.subtract(upper_bound, lower_bound)

        # Compute the two possible values I could add to score(instead of IF clause)
        added_value_down = np.subtract(lower_bound, y_true)
        added_value_up = np.subtract(y_true, upper_bound)

        # Create the needed matrix of zeros (third possible added value)
        added_value_zero = np.zeros((pred_quantiles.shape[0], pred_quantiles.shape[1]))

        # This is in place of the IF clause (I cannot do both max at the same time)
        added_value = np.maximum(added_value_down, added_value_up)
        added_value = np.maximum(added_value, added_value_zero)

        # Compute Winkler Score for a single prediction interval, in a bidimensional matrix
        single_score = deltaN + 2 / (1 - q) * added_value

        # Append the above value
        score.append(np.expand_dims(single_score, -1))

    score = np.mean(np.concatenate(score, axis=-1), axis=0)
    return score


def compute_delta_coverage(y_true, pred_quantiles, quantiles_levels, alpha_min, alpha_max):
    """
    Utility function to compute the Delta Coverage metric, to evaluate the goodness of
    an interval forecasting method. Works for alpha_min = 90% and alpha_max = 99%,
    given all the percentiles between alpha_min and alpha_max (included)


    INPUT
    y_true: test results
    pred_quantiles: prediction intervals
    quantiles_levels: quantile levels
    alpha_min: lower bound
    alpha_max: upper bound

    OUTPUT
    score: delta coverage
    """

    # Initialize score: we will append single "scores" and then compute the final Delta coverage
    score = []

    # Variable with how many quantiles I have
    amount_quantiles = len(quantiles_levels) - 1
    # for example if I have 21 quantiles, it's 20, because we do not care about the median

    for i in range(int(amount_quantiles / 2)):
        # Evaluate lower and upper bounds
        lower_bound = pred_quantiles.iloc[:, i]  # all the lower bounds for the considered time range
        upper_bound = pred_quantiles.iloc[:, -1 - i]  # all the upper bounds for the considered time range

        # Compute indicator function of the coverage for the current alpha level
        indicator = (y_true >= lower_bound) & (y_true <= upper_bound)
        indicator = indicator.astype(int)

        # Compute empirical coverage for current alpha level EC_alpha
        emp_coverage = np.mean(indicator)
        # Evaluate absolute difference with nominal alpha level (wanted confidence level of the PI)
        nominal_alpha = 1 - quantiles_levels[i] * 2
        single_score = 100 * abs(emp_coverage - nominal_alpha)

        # Append the above value
        score.append(np.expand_dims(single_score, -1))

    # Sum all the single scores of the different alpha levels
    score = np.sum(score)

    # Evaluate denominator of the delta coverage formula
    denominator = 100 * (alpha_max - alpha_min)
    # Evaluate the formula
    score = score / denominator

    return score


def sMAPE(y, y_hat):
    """
    Computes the SMAPE metric

    INPUT
    y: test results
    y_hat: prediction results

    OUTPUT
    sMAPE_: SMAPE metric
    """

    sMAPE_ = np.sum(abs(y - y_hat) / 0.5 * (abs(y) + abs(y_hat))) / len(y)
    return sMAPE_
