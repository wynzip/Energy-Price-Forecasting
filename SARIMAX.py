"""
SARIMAX model class
"""

# Authors: Sara Cupini, Davide Pagani, Francesco Panichi
# License: Apache-2.0 license
# Notice: these functions were completely developed by us, but rely on the structure built by Alessandro Brusaferri


import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler


class SARIMAXRegressor:

    def __init__(self, file_path: str, settings: dict, dates: dict):
        '''
        Constructor of the SARIMAX regressor class
        '''

        # Import settings
        self.settings = settings

        # Extract the whole dataset from the csv file
        self.dataset = pd.read_csv(file_path)

        # Add additional columns to the dataset
        self.expand_dataset()

        # Build dictionary of dates: start train, start test, end test
        self.dates_dict = {'idx_start_train': self.get_global_idx_from_date(dates['start_train']),
                           'idx_start_oos_preds': self.get_global_idx_from_date(dates['start_test']),
                           'idx_end_oos_preds': self.get_global_idx_from_date(dates['end_test'], mode='end')}

        # Add two other values to the settings dictionary: number of train and test days
        self.settings['num_train_days'] = int(self.dates_dict['idx_start_oos_preds']/self.settings['pred_horiz'])
        self.settings['num_test_days'] = int((self.dates_dict['idx_end_oos_preds']+1 -
                                              self.dates_dict['idx_start_oos_preds'])/self.settings['pred_horiz'])

    def get_test_predictions(self, sets_block: list, order: tuple, seasonal_order=(0, 0, 0, 0)):
        '''
        Run the daily recalibration routine and obtain the test predictions for all the needed days
        '''

        # Extract the datasets from the structure sets_block
        x_train = sets_block[0]
        x_test = sets_block[1]
        y_train = sets_block[2]
        y_test = sets_block[3]

        # Initialize the matrix of the predictions y_hat we compute in the recalibration
        predictions = np.zeros((self.settings['pred_horiz'], self.settings['num_test_days'], 1))

        # The calibration occurs hour per hour, so we first recalibrate all the days for a single hour,
        # and then we move on to the next hour
        for hour in range(self.settings['pred_horiz']):  # loop over the hours

            # Select current hour's initial training set
            x_train_curr = x_train[hour, :, :]
            y_train_curr = y_train[hour, :, :]

            # Here, if wanted, perform preprocessing of the current y train test, to smoothen the price outliers
            if self.settings['spike_preproc']:

                # Save a copy of the current y train, that will be used to restore it back to original values at the end
                # We do this just in the spike preprocessing case, because in this case we're losing track of the real
                # maximum levels of the time series
                y_train_curr_copy = y_train_curr.copy()

                # Perform spike preprocessing and return the updated y train
                y_train_curr = self.perform_spike_preprocessing(y_train_curr)


            # Rescale the values of the training set and save the x and y scalers
            scaler_x = MinMaxScaler()
            x_train_curr = scaler_x.fit_transform(x_train_curr)

            scaler_y = MinMaxScaler()
            y_train_curr = scaler_y.fit_transform(y_train_curr)

            # Set some starting parameters for this hour --> we use a model fit with more iterations,
            # to have a "warm start" for the actual parameters of the first day of the current hour
            if self.settings['model_class'] == 'ARIMAX':
                model = SARIMAX(endog=y_train_curr, exog=x_train_curr, order=order)
            elif self.settings['model_class'] == 'SARIMAX':
                model = SARIMAX(endog=y_train_curr, exog=x_train_curr, order=order, seasonal_order=seasonal_order)
            else:
                print('Check exper_setup in input: point-' + str(self.settings['model_class']) + ' is not implemented!')

            # Fit the model with a high number of max iterations
            model_fit = model.fit(maxiter=500, disp=False)

            # Save currently calibrated params
            old_model_params = model_fit.params.tolist()

            for t in range(self.settings['num_test_days']):  # loop over the test days (sliding window)

                # Create SARIMAX model object, according to the current model class
                if self.settings['model_class'] == 'ARIMAX':
                    model = SARIMAX(endog=y_train_curr, exog=x_train_curr, order=order)
                elif self.settings['model_class'] == 'SARIMAX':
                    model = SARIMAX(endog=y_train_curr, exog=x_train_curr, order=order, seasonal_order=seasonal_order)
                else:
                    print('Check exper_setup in input: point-' + str(
                        self.settings['model_class']) + 'is not implemented!')

                # Fit the model using past calibrated parameters
                model_fit = model.fit(start_params=old_model_params, disp=False)  # fit the model

                # Save currently calibrated params
                old_model_params = model_fit.params.tolist()

                # Obtain current test day's exogenous variables
                x_test_curr = x_test[hour, t, :].reshape(1, -1)
                # reshaping needed to append the vector afterward (transpose and virtually add a dimension)

                # Scale also the current x test values, with the same scaler previoulsy fitted
                x_test_curr = scaler_x.transform(x_test_curr)

                # Forecast the y value on the current test day, using this day's exogenous variables
                output = model_fit.forecast(steps=1, exog=x_test_curr)

                # Extract the predicted value from the model's output
                y_hat_curr = output[0]
                # Scale it back to normal values
                y_hat_curr = scaler_y.inverse_transform(y_hat_curr.reshape(-1, 1))[0]
                # Save it in the predictions matrix
                predictions[hour, t, :] = y_hat_curr

                # Extract the true value for this test day
                y_test_curr = y_test[hour, t, :]

                # Perform the inverse transformation on the exogenous variables
                x_train_curr = scaler_x.inverse_transform(x_train_curr)
                x_test_curr = scaler_x.inverse_transform(x_test_curr)

                # Restore the target variable to the original scale (different procedures whether we performed
                # the spike preprocessing or not)
                y_train_curr = y_train_curr_copy if self.settings['spike_preproc'] else (
                    scaler_y.inverse_transform(y_train_curr))

                # Update the train set, shifting one day in the future (today's test day gets integrated in the training
                # set, and the first day of the training set gets removed --> SLIDING WINDOW)

                # Update the train set for the target variable
                y_train_curr = np.delete(y_train_curr, 0, axis=0)
                # from y_train, delete the first row (since axis=0) (it's a column vector anyway)
                y_train_curr = np.append(y_train_curr, y_test_curr)

                # Update the train set for the exogenous variables
                x_train_curr = np.delete(x_train_curr, 0, axis=0)
                # from x_train, delete first row (since axis=0) (x is a matrix)
                x_train_curr = np.append(x_train_curr, x_test_curr, axis=0)
                # append the vector of current ex. var over the rows (as a new final row)

                # Here, if wanted, perform preprocessing of the current y train test, to smoothen the outliers
                if self.settings['spike_preproc']:

                    # Save a copy of the current y train, that will be used to restore it back to original values
                    # at the end. We do this just in the spike preprocessing case, because in this case we're losing
                    # track of the real maximum levels of the time series
                    y_train_curr_copy = y_train_curr.copy()

                    # Perform spike preprocessing and return the updated y train
                    y_train_curr = self.perform_spike_preprocessing(y_train_curr)

                # Transform the new training data, saving the new scalers of x and y fitted on the new train sets
                scaler_x = MinMaxScaler()
                x_train_curr = scaler_x.fit_transform(x_train_curr)

                scaler_y = MinMaxScaler()
                y_train_curr = scaler_y.fit_transform(y_train_curr.reshape(-1, 1))

                # Print info message
                print('Evaluating test day ' + str(t + 1) + ' of ' + str(self.settings['num_test_days']) + ' for hour '
                      + str(hour))

        # Let's put back in the right shape the predictions and the y_true
        median_pred = predictions.flatten('F')
        y_true = y_test.flatten('F')

        # Create the final dataframe to output from the function
        test_predictions = pd.DataFrame({0.5: median_pred, 'EM_price': y_true})

        return test_predictions

    def get_train_test_set_from_dataset(self):
        '''
        Obtain the training and test set from the whole dataset, looking at the dates, organized in hourly manner.
        Since the function we use to fit the SARIMAX models uses in-sample validation during fitting,
        we will omit to create a validation set.
        Returns:
            sets_block: contains x_train, y_train, x_test, y_test,
            scaler_y: list of scalers for the target variable, hour per hour
        '''

        # Divide in training and test set
        train_set = self.dataset.iloc[self.dates_dict['idx_start_train']: self.dates_dict['idx_start_oos_preds']]
        test_set = self.dataset.iloc[self.dates_dict['idx_start_oos_preds']: self.dates_dict['idx_end_oos_preds'] + 1]

        # We now drop the columns which will not be used for the calibration, except for the h
        train_set = train_set.loc[:, ['TARG__EM_price', 'Hour', 'FUTU__EM_load_f', 'FUTU__EM_wind_f', 'FUTU__weekend',
                                      'FUTU__spiky_regime']]
        test_set = test_set.loc[:, ['TARG__EM_price', 'Hour', 'FUTU__EM_load_f', 'FUTU__EM_wind_f', 'FUTU__weekend',
                                    'FUTU__spiky_regime']]

        # Obtain group-by objects to separate the train_set and test_set in hourly manner
        train_groupby_hour = train_set.groupby('Hour')
        test_groupby_hour = test_set.groupby('Hour')

        # Create a dictionary having as keys the hours and as value the hourly dataframes
        train_datasets = {hour: group.drop('Hour', axis=1) for hour, group in
                          train_groupby_hour}  # in the form of a dictionary hour - group
        test_datasets = {hour: group.drop('Hour', axis=1) for hour, group in test_groupby_hour}
        # how do we access them? e.g. train_datasets[11] gives the training dataset for the 11th hour

        # Drop also the Hour column from the train and test sets for consistency reasons, otherwise the columns used
        # below are wrong
        train_set = train_set.drop('Hour', axis=1)
        test_set = test_set.drop('Hour', axis=1)

        # Initialize 3D-vectors of train and test set, features and target variable
        num_ex_var = train_set.shape[1] - 1  # how many exogenous variables I am using (do not count target variable)
        x_train = np.zeros((self.settings['pred_horiz'], self.settings['num_train_days'], num_ex_var))
        y_train = np.zeros((self.settings['pred_horiz'], self.settings['num_train_days'], 1))
        x_test = np.zeros((self.settings['pred_horiz'], self.settings['num_test_days'], num_ex_var))
        y_test = np.zeros((self.settings['pred_horiz'], self.settings['num_test_days'], 1))

        # Loop over the hours to correctly insert the values in the train and test sets, performing scaling
        for hour in range(self.settings['pred_horiz']):  # looping over the hours

            # Select exogenous variables of training set
            x_train_tmp = train_datasets[hour].loc[:, train_set.columns != 'TARG__EM_price'].to_numpy()
            # Select target variable of training set
            y_train_tmp = train_datasets[hour].loc[:, 'TARG__EM_price'].to_numpy().reshape(-1, 1)
            # numpy and reshape needed for the scaler

            # Select exogenous variables of test set
            x_test_tmp = test_datasets[hour].loc[:, test_set.columns != 'TARG__EM_price'].to_numpy()
            # Select target variable of test set
            y_test_tmp = test_datasets[hour].loc[:, 'TARG__EM_price'].to_numpy().reshape(-1, 1)
            # numpy and reshape needed for the scaler

            # Finally save the scaled target variable in the corresponding matrix, both for training and test set
            x_train[hour, :, :] = x_train_tmp
            x_test[hour, :, :] = x_test_tmp

            # Finally save the scaled target variable in the corresponding matrix, both for training and test set
            y_train[hour, :, :] = y_train_tmp.reshape(-1, 1)
            y_test[hour, :, :] = y_test_tmp.reshape(-1, 1)

        # Everything is now well organized in the 3D matrices; to have fewer outputs, append them to a list,
        # in this fixed order: x_train, x_test, y_train, y_test
        sets_block = [x_train, x_test, y_train, y_test]

        return sets_block

    def expand_dataset(self):
        '''
        Assemble the dataset as needed for the calibration routine, looking at the
        dates and removing unneeded columns
        '''

        # Add column: dummy variable for the weekends
        self.add_dummy_weekend()

        # Add column: dummy variable for the spiky regimes
        self.add_dummy_spiky_regime()

    def add_dummy_weekend(self):
        '''
        Adds a columns to the dataset which is a dummy variable that signals
        if the current day is weekend day (1) or a weekday (0)
        '''

        # Find sin corresponding to each day of the week
        num_week_days = 7
        wd_sin = np.array([self.dataset.loc[:, 'CONST__wd_sin'].iloc[i * self.settings['pred_horiz']]
                           for i in range(num_week_days)])

        # Add the 'weekend' column to the dataset
        wd_sin_weekend = [wd_sin[0], wd_sin[1]]  # knowing the original dataset starts from a saturday (2015-01-03)
        self.dataset['FUTU__weekend'] = self.dataset['CONST__wd_sin'].apply(lambda x: 1 if x in [wd_sin[0], wd_sin[1]] else 0)

    def add_dummy_spiky_regime(self):
        """
        This function evaluates if the day we want to predict is to be considered a possible spiky day; that is,
        the energy price is in an upward dynamic
        """

        # Group the dataset by the index step, which is the daily index
        df_groupby_day = self.dataset.groupby(['IDX__step'])

        # Compute the daily means, thanks to the groupby object; they're well-ordered
        daily_means = df_groupby_day['TARG__EM_price'].mean()

        # Now we compute the 8-days means
        num_past_days_mean = 8
        eight_day_means = daily_means.rolling(window=num_past_days_mean).mean()

        # We evaluate the difference of yesterday's daily mean with the mean of the 8 prior days,
        # and then if the difference is higher than a selected threshold

        # Compute shifted difference between eight_day_means and daily_means
        diff = np.subtract(daily_means[num_past_days_mean:],
                           eight_day_means[num_past_days_mean - 1:-1]).dropna().to_numpy()
        diff = np.concatenate((np.zeros(num_past_days_mean + 1), diff))

        # Evaluate threshold; let's create the vector that should signal the regime we're in
        threshold = 2  # explanation of the threshold on the report
        spiky_dummy = (diff > threshold).astype(int)
        spiky_dummy.mean()

        # Before adding the column to the dataset, we just have to make it the right length (using sth like repelem)
        spiky_dummy = np.repeat(spiky_dummy, self.settings['pred_horiz'])

        # Add the column to the dataset
        self.dataset['FUTU__spiky_regime'] = spiky_dummy

    def get_global_idx_from_date(self, date_id, mode='start'):
        """
        Get the global idx related to the input date.
        Mode: 'start': return the idx of first sub_step; 'end': return the idx of first sub_step
        """
        date_idxs = self.dataset[self.dataset['Date'] == date_id].index.tolist()
        if mode == 'start':
            global_idx = date_idxs[0]
        elif mode == 'end':
            global_idx = date_idxs[-1]

        return global_idx

    def perform_spike_preprocessing(self, y_target: np.array):
        '''
        This function performs the preprocessing of the target variable time series, using the damping scheme suggested
        in Nowotarski et al. (2014), An empirical comparison of alternative schemes for combining electricity spot price
        forecasts. The function returns the modified time series.
        '''

        # But we go on to modify the actual y_train_curr
        # Find mean and standard deviation of the target price in the training set
        mean_y_train = y_target.mean()
        std_y_train = y_target.std()

        # Find the threshold for prices to be considered spikes, according to Nowotarski et al. (2014)
        t_star = mean_y_train + 3 * std_y_train

        # Find the indexes of the observations that go over the threshold
        condition = y_target > t_star
        # Change the spike values --> spike preprocessing
        y_target[condition] = t_star + t_star * np.log10(y_target[condition] / t_star)

        # Return spike preprocessed time series
        return y_target

