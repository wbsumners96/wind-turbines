import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import TransformedTargetRegressor
from tqdm import tqdm

from model.kernel_ridge_regressors import KernelRidgeRegressor


def wind_direction_location(data_positions, time, targets, references,
        filename=None):
    """
    Display scatterplot of turbine locations with wind speed and direction.

    Takes the output of load_data_positions() from load_data.py, and
    displays points as turbine locations with wind speed and direction as
    vectors.

    Parameters
    ----------
    data_positions : pd.DataFrame
        Wind turbine position data. 
    time : str
        Timestamp with the datetime format 'DD-MMM-YYYY hh:mm:ss'.
    filename : str
        The name of the file containing the resulting plot.
    """
    if data_positions.data_type == 'np.ndarray':
        data = data_positions.data[time]
        xs = data[:, 5]
        ys = data[:, 6]
        vxs = data[:, 1]*np.sin(data[:, 4]*np.pi/180)
        vys = data[:, 1]*np.cos(data[:, 4]*np.pi/180)

        plt.quiver(xs, ys, vxs, vys)
        plt.scatter(xs, ys)
        plt.scatter(xs[targets], ys[targets], label='Target turbine')
        plt.scatter(xs[references], ys[references], label='Reference turbine')
        plt.legend()

        if filename != None:
            plt.savefig(filename, format='png')

        plt.show()
    elif data_positions.data_type == 'pd.DataFrame':
        data_positions = data_positions.data
        data_time0 = data_positions[data_positions.ts 
            == data_positions.ts[time]]
        xs = data_time0['Easting'].to_numpy()
        ys = data_time0['Northing'].to_numpy()
        vxs = (data_time0['Wind_speed'].to_numpy()
               * np.sin(data_time0['Wind_direction_calibrated']
               * np.pi/180)).to_numpy()
        vys = (data_time0['Wind_speed'].to_numpy()
               * np.cos(data_time0['Wind_direction_calibrated']
               * np.pi/180)).to_numpy()

        plt.quiver(xs, ys, vxs, vys)
        plt.scatter(xs, ys)
        plt.scatter(xs[targets], ys[targets], label='Target turbine')
        plt.scatter(xs[references], ys[references], label='Reference turbine')
        plt.legend()

        if filename != None:
            plt.savefig(filename, format='png')

        plt.show()


def direction_power_histogram(data):
    """
    Plot histogram of wind directions and power outputs.

    For a given dataframe, bin all wind directions and corresponding power
    outputs and plot a 2D histogram.

    Parameters
    ----------
    data : pd.DataFrame
    """
    if data.data_type == 'np.ndarray':
        _dir = data.data[:, :, 4].reshape(-1)
        _pow = data.data[:, :, 2].reshape(-1)
        data_np = np.stack((_dir, _pow), axis=-1)
    elif data.data_type == 'pd.DataFrame':
        data_np = data.data[['Wind_direction_calibrated', 'Power']].to_numpy()

    plt.hist2d(data_np[:, 0], data_np[:, 1], bins=100, cmap='inferno')
    plt.xlabel('Turbine angle (degrees)')
    plt.ylabel('Turbine power (kW)')
    plt.title(data.farm)
    plt.colorbar()
    plt.show()


def prediction_measured_histogram(predictions, measurements):
    """
    Plot a 2D Heatmap of predicted vs measured powers
    """
    plt.hist2d(measurements, predictions, bins=100, norm=mpl.colors.LogNorm(), 
            cmap='binary')
    plt.xlabel('Measured Power (kW)')
    plt.ylabel('Predicted Power (kW)')
    plt.show()


def visualize_cor_func_behaviour(X, Y, ys):
    """
    Visualises the behaviour of the power-angle pairwise regression
    X and Y are measured reference and target data respectively,
    ys is predicted data.
    """
    cmap='binary'
    x = X[:, 0]  # Power
    xa = X[:, 1] # Angle
    y = Y[:, 0]  # Power
    ya = Y[:, 1] # Angle

    plt.scatter(x[::10], y[::10], label='Measured', alpha=0.05)
    plt.scatter(x[1::10], ys[1::10, 0], label='Prediction (on test data)', alpha=0.05)
    plt.xlabel('Reference power')
    plt.legend()
    plt.ylabel('Target power')
    plt.show()

    plt.scatter(xa[::10], ya[::10], label='Measured', alpha=0.05)
    plt.scatter(xa[1::10], ys[1::10, 1], label='Prediction (on test data)', alpha=0.05)

    plt.xlabel('Reference angle')
    plt.legend()
    plt.ylabel('Target angle')
    plt.show()

    _, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    ax[0].hist2d(xa, xa-ya, bins=100,
            range=[[np.min(xa), np.max(xa)], [-20, 20]],
            norm=mpl.colors.LogNorm(), cmap=cmap)

    ax[0].set_ylabel(r'$\theta_j-\theta_i$')
    ax[0].set_title('Measured')

    ax[1].hist2d(xa, xa-ys[:, 1], bins=100,
            range=[[np.min(xa), np.max(xa)], [-20, 20]],
            norm=mpl.colors.LogNorm(), cmap=cmap)
    ax[1].set_xlabel(r'$\theta_j$')
    ax[1].set_ylabel(r'$\theta_j - \hat{\theta}_i$')
    ax[1].set_title('Predicted')

    plt.show()

    _, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    ax[0].hist2d(xa, (x-y), bins=100, norm=mpl.colors.LogNorm(), cmap=cmap)
    ax[0].set_ylabel(r'$P_i-P_j$')
    ax[0].set_title('Measured')

    ax[1].hist2d(xa, (x-ys[:, 0]), norm=mpl.colors.LogNorm(), bins=100, cmap=cmap)
    ax[1].set_xlabel(r'$\theta_j$')
    ax[1].set_ylabel(r'$P_i-\hat{P}_j$')
    ax[1].set_title('Predicted')
    plt.show()

    plt.hist2d(ya, y-ys[:, 0], bins=100, norm=mpl.colors.LogNorm(), cmap=cmap)
    plt.xlabel(r'$\theta_j$')
    plt.ylabel(r'$P_j - \hat{P}_j$')
    plt.show()

    plt.scatter((xa[::10] + ya[::10])/2, (x[::10] - y[::10]), label='Measured', 
            alpha=0.05)
    plt.scatter((xa[::10] + ys[::10, 1])/2, (x[::10] - ys[::10,0]), 
            label='Prediction', alpha=0.05)
    plt.xlabel(r'$\frac{\theta_i+\theta_j}{2}$')
    plt.ylabel(r'$P_i-P_j$')
    plt.title('Power difference vs mean angle correlation')
    plt.legend()
    plt.show()

    plt.hist2d(y, ys[:, 0], bins=100, norm=mpl.colors.LogNorm(), cmap=cmap)
    plt.xlabel(r'$P_j$')
    plt.ylabel(r'$\hat{P}_j$')
    plt.show()

    plt.scatter(y, (y - ys[:, 0])/y, alpha=0.1)
    plt.xlabel(r'$P_j$')
    plt.ylabel(r'$\frac{P_j - \hat{P}_j}{P_j}$')
    plt.show()

    plt.hist2d(y, (y - ys[:, 0])/(1 + y), bins=100, norm=mpl.colors.LogNorm(), 
            cmap=cmap)
    plt.xlabel(r'$P_j$')
    plt.ylabel(r'$\frac{P_j - \hat{P}_j}{P_j}$')
    plt.show()

    h, xedges, yedges = np.histogram2d(y, y - ys[:, 0], bins=50)
    plt.imshow(h.T, origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
            interpolation='gaussian', cmap='Greens')

    h_av = np.zeros(xedges.shape[0] - 1)
    h_var = np.zeros(xedges.shape[0] - 1)
    
    for i in range(h_av.shape[0]):
        h_av[i] = np.average(yedges[1:], weights=h[i])
        h_var[i]= np.sqrt(np.average((yedges[1:] - h_av[i])**2, weights=h[i]))

    plt.plot(xedges[:-1], h_av, color='red')
    plt.plot(xedges[:-1], h_av + h_var, color='red', alpha=0.2)
    plt.plot(xedges[:-1], h_av - h_var, color='red', alpha=0.2)
    plt.show()

    plt.hist(y, bins=100, alpha=0.4, label='Measured', density=True)
    plt.hist(ys[:, 0], bins=100, alpha=0.4, label='Predicted', density=True)
    plt.xlabel('Target turbine power (kW)')
    plt.legend()
    plt.ylabel('Count')
    plt.show()


def average_power_gain_curve_dataframes(data, predictor):
    def all_predictions(data, regressor):
        N = data.n_turbines
        targets = list(range(1, 2))
        references = list(range(2, 16))

        data.select_normal_operation_times()
        # data_copy.select_unsaturated_times()
        # data_copy.select_power_min()
        
        predictions_fr = regressor.predict(data, targets, references, None)
        
        predictions = predictions_fr['predicted_power'].to_numpy()
        measurements = predictions_fr['target_power'].to_numpy()
        errors = predictions - measurements

        print(predictions)
        print(measurements)
        print(errors)

        return predictions, measurements, errors

    _, measured, errors = all_predictions(data, predictor)

    # Plots power errors as functions of measured powers, averaged over
    # turbines, like Oli showed in meeting
    M = np.array(measured, dtype=object).flatten()
    E = np.array(errors, dtype=object).flatten()

    M = np.array(M, dtype=float)
    E = np.array(E, dtype=float)

    h, xedges, yedges = np.histogram2d(M, E, bins=100, 
            range=[[M.min(), M.max()], [E.min(), E.max()]])

    plt.imshow(h.T, origin='lower', 
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
            interpolation='gaussian', cmap='Greens')

    h_av = np.zeros(xedges.shape[0] - 1)
    h_var = np.zeros(xedges.shape[0] - 1)
    
    for i in range(h_av.shape[0]):
        h_av[i] = np.average(yedges[1:], weights=h[i])
        h_var[i] = np.sqrt(np.average((yedges[1:] - h_av[i])**2, weights=h[i]))

    plt.plot(xedges[:-1], h_av, color='red', label='Weighted average')
    plt.plot(xedges[:-1], h_av + h_var, color='red', alpha=0.2, 
            label='Standard deviation')
    plt.plot(xedges[:-1], h_av - h_var, color='red', alpha=0.2)
    plt.xlabel(r'$P$')
    plt.ylabel(r'$\delta P$')
    plt.colorbar()
    plt.legend()
    plt.show()




def average_power_gain_curve(data, k_mat):
    """
    Produces a power gain curve averaged over every pair of
    turbines in the farm. Expects data as np.array
    """
    def all_predictions(data, k_mat):
        N = data.n_turbines
        its = list(range(N))
        predictions = [[] for _ in range(N)]
        measurements = [[] for _ in range(N)]
        errors = [[] for _ in range(N)]
        for i in tqdm(its):
            for j in tqdm(its[:i] + its[i + 1:]):
                data_copy = copy.deepcopy(data)
                data_copy.select_turbine([i, j])
                data_copy.select_normal_operation_times()
                data_copy.select_unsaturated_times()
                data_copy.select_power_min()
                x = data_copy.data[:, 1, 2] # reference power
                xa = data_copy.data[:, 1, 4] # reference angle
                y = data_copy.data[:, 0, 2] # target power
                X = np.stack((x, xa), axis=1)

                p = k_mat[i, j].predict(X)
                predictions[i].append(p[:, 0])
                measurements[i].append(y)
                errors[i].append(y - p[:, 0])

        return predictions, measurements, errors

    _, measured, errors = all_predictions(data, k_mat)

    # Plots power errors as functions of measured powers, averaged over
    # turbines, like Oli showed in meeting
    M = np.array(measured, dtype=object).flatten()
    E = np.array(errors, dtype=object).flatten()

    M = np.array(M,dtype=float)
    E = np.array(E,dtype=float)

    h, xedges, yedges = np.histogram2d(M, E, bins=100, 
            range=[[M.min(), M.max()], [E.min(), E.max()]])

    plt.imshow(h.T, origin='lower', 
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
            interpolation='gaussian', cmap='Greens')

    h_av = np.zeros(xedges.shape[0] - 1)
    h_var = np.zeros(xedges.shape[0] - 1)
    
    for i in range(h_av.shape[0]):
        h_av[i] = np.average(yedges[1:], weights=h[i])
        h_var[i] = np.sqrt(np.average((yedges[1:] - h_av[i])**2, weights=h[i]))

    plt.plot(xedges[:-1], h_av, color='red', label='Weighted average')
    plt.plot(xedges[:-1], h_av + h_var, color='red', alpha=0.2, 
            label='Standard deviation')
    plt.plot(xedges[:-1], h_av - h_var, color='red', alpha=0.2)
    plt.xlabel(r'$P$')
    plt.ylabel(r'$\delta P$')
    plt.colorbar()
    plt.legend()
    plt.show()

