import copy
import math
from typing import Dict
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, \
                            mean_absolute_percentage_error, \
                            median_absolute_error
from tqdm import tqdm
from model.kernel_ridge_regressors import KernelRidgeRegressor


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'axes.unicode_minus': False
})


def heatmap(predictions, filename=None):
    target = predictions['target_power'].to_numpy()
    predicted = predictions['predicted_power'].to_numpy()
    errors = target - predicted

    plt.hist2d(target, errors, cmap='YlOrBr', norm=colors.LogNorm())

    plt.show()


def summary_metrics(predictions, row_name, filename=None):
    """
    Construct a LaTeX row with columns mean absolute error, mean absolute
    percentage error, and median error.
    """
    target = predictions['target_power'].to_numpy()
    predicted = predictions['predicted_power'].to_numpy()

    mae = mean_absolute_error(target, predicted)
    mape = mean_absolute_percentage_error(target, predicted)
    medae = median_absolute_error(target, predicted)

    row = f'{row_name} & {mae} & {mape} & {medae}'

    if filename is not None:
        with open(filename, 'w') as fs:
            fs.write(row)

    return row


def r2_matrices(r2_scores, filename=None):
    def f(entry):
        if entry is None:
            return 1.0
        else:
            return entry

    base_text_width = 6.13888888889
    fig, axes = plt.subplots(2, 4, figsize=(0.9*base_text_width,
        base_text_width/1.8))

    farms = ['ARD', 'CAU']
    kernels = { 'lpp': 'Periodic Laplacian',
                'l': 'Power Laplacian',
                'lll': 'Laplacian',
                'rb': 'Radial Basis' }
    for i, farm in enumerate(farms):
        for j, kernel in enumerate(kernels.keys()):
            fk_r2_scores = r2_scores[farm][kernel]
            r2 = [fk_r2_scores[key] for key in sorted(fk_r2_scores.keys())]
            r2 = [[f(entry[key]) for key in sorted(entry.keys())] for entry in r2]
            r2 = np.array(r2)

            ax = axes[i, j]

            img = ax.matshow(r2, vmin=0.0, vmax=1.0)

            # ax.set_xlabel('Reference')
            # ax.set_ylabel('Target')

            # axis = range(len(fk_r2_scores.keys()))[::6]

            # ax.set_xticks(axis)
            # ax.set_yticks(axis)
            # ax.set_xticklabels(sorted(fk_r2_scores.keys())[::6])
            # ax.set_yticklabels(sorted(fk_r2_scores.keys())[::6])

            ax.tick_params(top=False, 
                           bottom=False,
                           left=False,
                           labeltop=False,
                           labelleft=False)

    fig.tight_layout()

    for i, farm in enumerate(farms):
        axes[i, -1].set_ylabel(farm)
        axes[i, -1].yaxis.set_label_coords(1.15, 0.5)
        axes[i, -1].legend()

    for j, key in enumerate(kernels.keys()):
        axes[-1, j].set_xlabel(kernels[key])

    axes[0, 0].set_xlabel('Reference')
    axes[0, 0].set_ylabel('Target')
    axes[0, 0].xaxis.set_label_coords(0.5, 1.15)

    fig.subplots_adjust(top=0.76,
                        bottom=0.08,
                        left=0.05,
                        right=0.85,
                        hspace=0.0,
                        wspace=0.16)

    cbar_ax = fig.add_axes([0.9, 0.1, 0.03, 0.65])
    fig.colorbar(img, cax=cbar_ax)
    fig.patch.set_visible(False)

    fig.savefig('r2_matrix.pgf')


def r2_matrix(r2_scores, filename=None):
    """
    Display the dictionary of R^2 scores as an array.
    """
    def f(entry):
        if entry is None:
            return 1.0
        else:
            return entry

    r2 = [r2_scores[key] for key in sorted(r2_scores.keys())]
    r2 = [[f(entry[key]) for key in sorted(entry.keys())] for entry in r2]
    r2 = np.array(r2)

    fig = plt.figure()
    ax = fig.gca()

    img = ax.matshow(r2, vmin=0.0, vmax=1.0)
    
    ax.set_xlabel('Reference')
    ax.set_ylabel('Target')

    axis = range(len(r2_scores.keys()))[::6]

    ax.set_xticks(axis)
    ax.set_yticks(axis)
    ax.set_xticklabels(sorted(r2_scores.keys())[::6])
    ax.set_yticklabels(sorted(r2_scores.keys())[::6])

    fig.colorbar(img)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    

def power_gain_curve(predictions, filename=None):
    cmap = cm.get_cmap('gist_rainbow')
    target_ids = predictions['target_id'].drop_duplicates()
    for j, target_id in enumerate(target_ids):
        target_predictions = predictions.query('target_id == @target_id')

        target = target_predictions['target_power'].to_numpy()
        predicted = target_predictions['predicted_power'].to_numpy()
        errors = target - predicted

        h, xedges, yedges = np.histogram2d(target, errors, bins=100, 
                range=[[target.min(), target.max()],
                    [errors.min(), errors.max()]])

        h_av = np.zeros(xedges.shape[0] - 1)
        h_var = np.zeros(xedges.shape[0] - 1)
        
        for i in range(h_av.shape[0]):
            if math.isclose(sum(h[i]), 0):
                h_av[i] = h_av[i-1]
                h_var[i] = h_var[i-1]
            else:
                h_av[i] = np.average(yedges[1:], weights=h[i])
                h_var[i] = np.sqrt(np.average((yedges[1:] - h_av[i])**2,
                        weights=h[i]))
        
        color = cmap(j/len(target_ids))

        plt.plot(xedges[:-1], h_av, color=color, label=f'{target_id}')
        plt.fill_between(xedges[:-1], h_av - h_var, h_av + h_var, color=color,
                alpha=0.2)

    plt.xlim(0, 2000)
    plt.ylim(-500, 500)

    plt.xlabel(r'$P$')
    plt.ylabel(r'$\Delta P$')
    
    plt.grid()
    plt.legend()

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def power_gain_curves(predictionss, filename=None):
    base_text_width = 6.13888888889
    fig, axes = plt.subplots(2, 4, figsize=(1.0*base_text_width,
        base_text_width/1.5))

    farms = ['ARD', 'CAU']
    kernels = { 'lpp': 'Periodic Laplacian',
                'l': 'Power Laplacian',
                'lll': 'Laplacian',
                'rb': 'Radial Basis' }
    all_turbines = { 'ARD': [f'ARD_WTG{x:02}' for x in range(1, 16)],
                     'CAU': [f'CAU_WTG{x:02}' for x in list(range(1, 11)) +
                         list(range(14, 24))] }
    turbines = { 'ARD': np.random.choice(all_turbines['ARD'], size=3),
                 'CAU': np.random.choice(all_turbines['CAU'], size=3) }
    for i, farm in enumerate(farms):
        for j, kernel in enumerate(kernels.keys()):
            predictions = predictionss[farm][kernel]
            targets = turbines[farm].tolist()
            predictions.query('target_id == @targets', inplace=True)

            ax = axes[i, j]

            ax.tick_params(top=False, 
                           bottom=False,
                           left=False,
                           labeltop=False,
                           labelleft=False,
                           labelbottom=False)

            cmap = cm.get_cmap('gist_rainbow')
            target_ids = predictions['target_id'].drop_duplicates()
            for j1, target_id in enumerate(target_ids):
                target_predictions = predictions.query('target_id == @target_id')

                target = target_predictions['target_power'].to_numpy()
                predicted = target_predictions['predicted_power'].to_numpy()
                errors = target - predicted

                h, xedges, yedges = np.histogram2d(target, errors, bins=100, 
                        range=[[target.min(), target.max()],
                            [errors.min(), errors.max()]])

                h_av = np.zeros(xedges.shape[0] - 1)
                h_var = np.zeros(xedges.shape[0] - 1)
                
                for i1 in range(h_av.shape[0]):
                    if math.isclose(sum(h[i1]), 0):
                        h_av[i1] = h_av[i1-1]
                        h_var[i1] = h_var[i1-1]
                    else:
                        h_av[i1] = np.average(yedges[1:], weights=h[i1])
                        h_var[i1] = np.sqrt(np.average((yedges[1:] - h_av[i1])**2,
                                weights=h[i1]))
                
                color = cmap(j1/len(target_ids))

                ax.plot(xedges[:-1], h_av, color=color, label=f'{target_id}')
                ax.fill_between(xedges[:-1],
                        h_av - h_var,
                        h_av + h_var,
                        color=color,
                        alpha=0.2)

            ax.set_xlim(0, 2000)
            ax.set_ylim(-500, 500)
            
            ax.set_xmargin(0)
            ax.set_ymargin(0)
            
            ax.grid()

    axes[0, 0].tick_params(top=True, 
                           left=True,
                           labeltop=True,
                           labelleft=True)

    axes[0, 0].set_xlabel(r'$P$')
    axes[0, 0].set_ylabel(r'$\Delta P$')
    axes[0, 0].xaxis.set_label_coords(0.5, 1.2)

    fig.tight_layout()

    for i, farm in enumerate(farms):
        axes[i, -1].set_ylabel(farm)
        axes[i, -1].yaxis.set_label_coords(1.1, 0.5)
        axes[i, -1].legend()

    for j, key in enumerate(kernels.keys()):
        axes[-1, j].set_xlabel(kernels[key])

    plt.subplots_adjust(top=0.8,
                        bottom=0.05,
                        left=0.1,
                        right=0.95,
                        hspace=0.034,
                        wspace=0.033)

    fig.savefig('power_gain_curves.pgf')


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
        turbines = data.data['instanceID'].drop_duplicates()
        turbine_numbers = [int(turbine[-2:]) for turbine in turbines]

        data.select_normal_operation_times()
        # data_copy.select_unsaturated_times()
        # data_copy.select_power_min()
        
        predictions_fr = regressor.predict(data,
                                           turbine_numbers,
                                           turbine_numbers,
                                           None)
        
        predictions = predictions_fr['predicted_power'].to_numpy()
        measurements = predictions_fr['target_power'].to_numpy()
        errors = predictions - measurements

        return predictions, measurements, errors

    _, measured, errors = all_predictions(data, predictor)
    measured = measured.flatten()
    errors = errors.flatten()

    h, xedges, yedges = np.histogram2d(measured, errors, bins=100, 
            range=[[measured.min(), measured.max()],
                [errors.min(), errors.max()]])

    # plt.imshow(h.T, origin='lower', 
    #         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
    #         interpolation='gaussian', cmap='Greens')

    h_av = np.zeros(xedges.shape[0] - 1)
    h_var = np.zeros(xedges.shape[0] - 1)
    
    for i in range(h_av.shape[0]):
        h_av[i] = np.average(yedges[1:], weights=h[i])
        h_var[i] = np.sqrt(np.average((yedges[1:] - h_av[i])**2, weights=h[i]))

    plt.plot(xedges[:-1], h_av, color='red', label='Weighted average')
    plt.fill_between(xedges[:-1], h_av - h_var, h_av + h_var, color='red',
            alpha=0.2, label='Standard deviation')

    plt.xlim(0, 2000)
    plt.ylim(-500, 500)

    plt.xlabel(r'$P$')
    plt.ylabel(r'$\Delta P$')
    
    plt.grid()
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
