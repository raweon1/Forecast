import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error


class TimeSeriesPlot:
    def __init__(self, df: pd.DataFrame, target_col: str, prediction_col: str, time_col: str = "time"):
        self.df = df.copy(deep=True)
        self.target = target_col
        self.prediction = prediction_col
        self.time = time_col
        self.df['year'] = self.df[self.time].dt.year
        self.df['month'] = self.df[self.time].dt.month
        self.df['day'] = self.df[self.time].dt.day
        self.df['hours'] = self.df[self.time].dt.hour
        self.df["dayofweek"] = self.df[self.time].dt.weekday
        self.df["se"] = self.squared_error()
        self.df["mae"] = self.absolute_error()
        self.df["mape"] = self.absolute_percentage_error()

    def mape(self):
        return self.df["mape"].mean()

    def mae(self):
        return self.df["mae"].mean()

    def rmse(self):
        return np.sqrt(mean_squared_error(self.df[self.target], self.df[self.prediction]))

    def squared_error(self):
        return np.power(np.subtract(self.df[self.target], self.df[self.prediction]), 2)

    def absolute_error(self):
        return np.abs(np.subtract(self.df[self.target], self.df[self.prediction]))

    def absolute_percentage_error(self):
        return np.abs(np.divide(np.subtract(self.df[self.target], self.df[self.prediction]), self.df[self.target])) * 100

    def plot_rmse_by_time(self, time: str):
        fig = plt.figure()
        col_map = {"y": "year", "m": "month", "d": "day", "h": "hours",  "wd": "dayofweek"}
        group_by_cols = [col_map[param] for param in time.split("%") if param in col_map.keys()]
        grouped = self.df.groupby(by=group_by_cols)
        mean = grouped["se"].mean()
        group_index = [str(index) for index in mean.index]
        plt.plot(group_index, np.sqrt(mean), label="rmse")
        return fig

    def plot_multiple(self, *cols: str):
        fig = plt.figure()
        for col in cols:
            plt.plot(self.df[self.time].values, self.df[col].values, alpha=0.5, label=col)
        plt.legend()
        plt.show()
        return fig

    def plot_target_by_time(self, time: str, *targets: str):
        col_map = {"y": "year", "m": "month", "d": "day", "h": "hours",  "wd": "dayofweek"}
        group_by_cols = [col_map[param] for param in time.split("%") if param in col_map.keys()]
        self.df.boxplot(column=[*targets], by=group_by_cols,
                        layout=(targets.__len__(), 1), rot=45, figsize=(50,40), grid=False)
        return plt.gcf()

    def plot_target_by_time_band(self, time: str, *targets: str, band: bool=False):
        fig = plt.figure()
        col_map = {"y": "year", "m": "month", "d": "day", "h": "hours",  "wd": "dayofweek"}
        group_by_cols = [col_map[param] for param in time.split("%") if param in col_map.keys()]
        grouped = self.df.groupby(by=group_by_cols)
        for tar in targets:
            mean = grouped[tar].mean()
            std = grouped[tar].std()
            group_index = [str(index) for index in mean.index]
            plt.plot(group_index, mean, label=tar)
            if band:
                plt.fill_between(group_index, mean - 2*std, mean + 2*std, alpha=.5)
        plt.legend()
        return fig


def encode_cyclic(values):
    """
    :param values: np array of integer values to encode
    :return: two np arrays which contain the encoded sin and cos coordinates
    """
    min_value = min(values)
    max_value_scaled = max(values) - min_value + 1
    sin_coord = np.sin(2 * np.pi * (values - min_value) / max_value_scaled)
    cos_coord = np.cos(2 * np.pi * (values - min_value) / max_value_scaled)
    return sin_coord, cos_coord


def create_input_output_sequence(input_data, seq_len, pred_len=1, multivar=False):
    """
    Takes a TimeSeries and creates new data points, where each data point is a sequence of timesteps from the TimeSeries
    :param input_data: TimeSeries-Data from which so create sequences,
    where each row is one Timestep
    and each column is one Feature
    :param seq_len: Length of the sequence, i.e. the number of timesteps from the input_data for each input
    :param pred_len: Length of the prediction, i.e. the number of timesteps from the input_data which are prediction by the input
    :param multivar: bool; if false only takes the last column as features for the prediction
    :return: Returns two numpy arrays X, y
    X contains sequences of shape [seq_len, input_data.number_of_columns]
    y contains sequences of shape [pred_len, 1] if multivar=False, [pred_len, input_data.number_of_columns] otherwise
    X is the input, where y is the corresponding label
    """
    # if multivar: target column has to be the last column
    X = []
    y = []
    for i in range(len(input_data) - seq_len - pred_len + 1):
        train_seq = input_data[i: i + seq_len]
        train_label = input_data[i + seq_len: i + seq_len + pred_len]
        if not multivar:
            train_label = train_label[:, -1]
        X.append(train_seq)
        y.append(train_label)
    X = np.array(X)
    y = np.array(y)
    return X, y


def create_input_sequence(input_data, seq_len):
    """
    Takes a TimeSeries and creates new data points, where each data point is a sequence of timesteps from the TimeSeries
    :param input_data: TimeSeries-Data from which so create sequences,
    where each row is one Timestep
    and each column is one Feature
    :param seq_len: Length of the sequence, i.e. the number of timesteps from the input_data for each input
    :return: returns a single numpy array containing all possible sequences with the length of seq_len out of input_data
    """
    X = []
    for i in range(len(input_data) - seq_len + 1):
        train_seq = input_data[i: i + seq_len]
        X.append(train_seq)
    X = np.array(X)
    return X


def split_validation(X, y, num_val):
    """
    Splits X, y randomly for a validation set. The validation set contains num_val values
    :param X: Feature-Vector
    :param y: Label-Vector
    :param num_val: Number of samples in the validation set
    :return: X_train, y_train, X_val, y_val
    """
    idx = np.random.choice(X.shape[0], num_val, replace=False)
    mask = np.ones(X.shape[0], bool)
    mask[idx] = False
    return X[mask], y[mask], X[idx], y[idx]


def batch_generator(stop, desired_batch_size=100):
    div = int(stop / desired_batch_size)
    rest = stop % desired_batch_size
    for i in range(div):
        yield desired_batch_size
    if rest != 0:
        yield rest


def batch_iterator(stop, desired_batch_size=100):
    return [(batch * desired_batch_size, batch_size) for batch, batch_size
            in enumerate(batch_generator(stop, desired_batch_size))]


def predict_seq(model, input, desired_batch_size=100):
    """
    Uses the input to create predictions with a model
    :param model: Some PyTorch.nn Model
    :param input: Input must be sequences, i.e. not the raw TimeSeries-data
    :param desired_batch_size:
    :return: Returns the predictions of this model with this input
    """
    prediction = []
    with torch.no_grad():
        for batch_start, batch_size in batch_iterator(len(input), desired_batch_size):
            output = model(input[batch_start: batch_start + batch_size])
            prediction.append(output.cpu().numpy())
    prediction = np.vstack(prediction)
    return prediction.reshape(len(input), -1)


def predict(model, input, seq_len,  desired_batch_size=100):
    """
    Uses the input to create predictions with a model. The input is transformed to sequences of seq_len length first
    :param model: Some PyTorch.nn Model
    :param input: Input must be the raw TimeSeries-data
    :param seq_len: Length of the sequences used as input for the model
    :param desired_batch_size:
    :return: Returns the predictions of this model with this input
    """
    X = create_input_sequence(input, seq_len)
    X = torch.FloatTensor(X).to(next(model.parameters()).device)
    return predict_seq(model, X, desired_batch_size)


def sum_abs_diff_batch(a, b):
    return np.sum(np.abs(a-b), 1)


def saliancy_per_sample_seq(model, input, desired_batch_size=100):
    """
    Computes the absolute difference between a prediction and a prediction where for each timestep the features where set
    to zero for each sample
    :param model: Some Pytorch.nn Model
    :param input: Input must be of shape [samples, seq_len, features]
    :param desired_batch_size:
    :return: Returns a np.array of shape [samples, seq_len] where the second dimension is the absolut difference
    between the original prediction and the prediction by setting that specific timestep to zero
    """
    prediction = []
    with torch.no_grad():
        mask = torch.zeros(input.shape[2]).to(next(model.parameters()).device)
        for batch, (batch_start, batch_size) in enumerate(batch_iterator(len(input), desired_batch_size)):
            seq = input[batch_start: batch_start + batch_size]
            pred = model(seq).cpu().numpy()
            pred_ = np.empty((input.shape[1], batch_size, pred.shape[-1]))
            for feature in range(input.shape[1]):
                feature_vector_batch = seq[:, feature].clone()
                seq[:, feature] = mask
                pred_[feature] = model(seq).cpu().numpy()
                seq[:, feature] = feature_vector_batch
            prediction.append(np.array([sum_abs_diff_batch(pred, p) for p in pred_]).swapaxes(0, 1))
    prediction = np.vstack(prediction)
    return prediction.reshape(-1, input.shape[1])


def saliancy_per_sample(model, input, seq_len, desired_batch_size=100):
    """
    Computes the absolute difference between a prediction and a prediction where for each timestep the features where set
    to zero for each sample. The input is transformed to sequences of seq_len length first
    :param model: Some Pytorch.nn Model
    :param input: Input must be the raw TimeSeries-data
    :param seq_len: Length of the sequences used as input for the model
    :param desired_batch_size:
    :return: Returns a np.array of shape [samples, seq_len] where the second dimension is the absolut difference
    between the original prediction and the prediction by setting that specific timestep to zero
    """
    X = create_input_sequence(input, seq_len)
    X = torch.FloatTensor(X).to(next(model.parameters()).device)
    return saliancy_per_sample_seq(model, X, desired_batch_size)


def saliancy_per_feature_seq(model, input, desired_batch_size=100):
    """
    Computes the absolute difference between a prediction and a prediction where for each timestep each features is set
    to zero for each sample
    :param model: Some Pytorch.nn Model
    :param input: Input must be of shape [samples, seq_len, features]
    :param desired_batch_size:
    :return: Returns a np.array of shape [samples, seq_len, num_features] where each value corresponds to setting
    that feature of that sample to zero
    """
    prediction = []
    with torch.no_grad():
        mask_mat = (torch.eye(input.shape[2]) - torch.ones(input.shape[2])).abs().to(next(model.parameters()).device)
        for batch, (batch_start, batch_size) in enumerate(batch_iterator(len(input), desired_batch_size)):
            seq = input[batch_start: batch_start + batch_size]
            pred = model(seq).cpu().numpy()
            pred_ = np.empty((input.shape[1], input.shape[2], batch_size, pred.shape[-1]))
            for timestep in range(input.shape[1]):
                feature_vector_batch = seq[:, timestep].clone()
                for feature in range(input.shape[2]):
                    seq[:, timestep] = feature_vector_batch * mask_mat[feature]
                    pred_[timestep, feature] = model(seq).cpu().numpy()
                    seq[:, timestep] = feature_vector_batch
            prediction.append(np.array([[sum_abs_diff_batch(pred, f) for f in s] for s in pred_]).transpose((2, 0, 1)))
    prediction = np.vstack(prediction)
    return prediction.reshape(-1, input.shape[1], input.shape[2])


def saliancy_per_feature(model, input, seq_len, desired_batch_size=100):
    X = create_input_sequence(input, seq_len)
    X = torch.FloatTensor(X).to(next(model.parameters()).device)
    return saliancy_per_feature_seq(model, X, desired_batch_size)

