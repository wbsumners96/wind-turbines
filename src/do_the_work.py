import os
from pathlib import Path

from tqdm.std import tqdm
import load_data
import correlation
from model.weighted_average import GaussianWeightedAverage
from model.kernel_ridge_regressors import LaplacianKRR, PeriodicLaplacianKRR, PowerLaplacianKRR, RadialBasisKRR
import visualize
from joblib import dump, load


path = Path('~/.turbines/predictions/').expanduser()

farms = ['ARD', 'CAU']
kernels = ['lpp', 'l', 'rb', 'lll']
kernel_names = { 'lpp': 'Periodic Laplacian',
                 'l': 'Power Laplacian',
                 'rb': 'Radial Basis',
                 'lll': 'Laplacian' }
turbines = { 'ARD': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             'CAU': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20,
                     21, 22, 23, 24] }

if not path.is_dir():
    os.mkdir(path)

for farm in tqdm(['ARD', 'CAU'], leave=False):
    data = load_data.TurbineData('../../Data/', farm)
    data.clear_wake_affected_turbines()
    data.select_baseline(inplace=True)
    data.select_normal_operation_times()
    data.sample(0.5, inplace=True)

    for key, predictor in tqdm(kernels.items(), leave=False):
        predictor.fit(data)
        dump(predictor, path/f'{farm}_{key}_predictor.joblib')
        dump(predictor.scores(data), path/f'{farm}_{key}_scores.joblib')

        predictor.aggregation = 'none'
        predictions = predictor.predict(data, turbines[farm], turbines[farm])
        dump(predictions, path/f'{farm}_{key}_predictions_none.joblib')

        predictor.aggregation = 'r2'
        predictions = predictor.predict(data, turbines[farm], turbines[farm])
        dump(predictions, path/f'{farm}_{key}_predictions_r2.joblib')



# r2_scores = {}
# for farm in farms:
#     farm_r2_scores = {}
#     for kernel in kernels:
#         farm_r2_scores[kernel] = load(path/f'{farm}_{kernel}_scores.joblib')
# 
#     r2_scores[farm] = farm_r2_scores
# 
# visualize.r2_matrices(r2_scores)

# predictionss = {}
# for farm in farms:
#     farm_predictionss = {}
#     for kernel in kernels:
#         farm_predictionss[kernel] = \
#                 load(path/f'{farm}_{kernel}_predictions_r2.joblib')
# 
#     predictionss[farm] = farm_predictionss
# 
# visualize.power_gain_curves(predictionss)

# data = load_data.TurbineData('../../Data/', 'CAU')
# data.clear_wake_affected_turbines()
# data.select_baseline(inplace=True)
# data.select_normal_operation_times()

# kernels = { 'lpp': PeriodicLaplacianKRR(1),
#             'l': PowerLaplacianKRR(),
#             'rb': RadialBasisKRR(),
#             'lll': LaplacianKRR() }


# data = load_data.TurbineData('../../Data/', 'ARD')
# data.clear_wake_affected_turbines()
# data.select_baseline(inplace=True)
# data.select_normal_operation_times()
# data.sample(frac=0.5, inplace=True)
# for key, predictor in tqdm(kernels.items(), leave=False):
#     predictor.fit(data)
#     predictor.aggregation = 'r2'
#     predictions = predictor.predict(data, turbines['ARD'], turbines['ARD'])
#     dump(predictions, path/f'ARD_{key}_predictions_r2.joblib')
# 
# data = load_data.TurbineData('../../Data/', 'CAU')
# data.clear_wake_affected_turbines()
# data.select_baseline(inplace=True)
# data.select_normal_operation_times()
# data.sample(frac=0.5, inplace=True)
# for key, predictor in tqdm(kernels.items(), leave=False):
#     predictor.fit(data)
# 
#     if key == 'lpp':
#         predictor.aggregation = 'none'
#         predictions = predictor.predict(data, turbines['CAU'], turbines['CAU'])
#         dump(predictions, path/f'CAU_{key}_predictions_none.joblib')
#     else:
#         predictor.aggregation = 'r2'
#         predictions = predictor.predict(data, turbines['CAU'], turbines['CAU'])
#         dump(predictions, path/f'CAU_{key}_predictions_r2.joblib')

# data.sample(frac=0.5, inplace=True)
# 
# predictor = PeriodicLaplacianKRR(1)
# predictor.fit(data)

# data.sample(frac=0.1, inplace=True)
# 
# 
# predictor = load(path/'cau_predictor.joblib')
# predictions = predictor.predict(data, [1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 14, 15,
#     16, 17, 18, 19, 20, 21, 22, 23, 24])
# 
# print(predictions)
# 
# visualize.power_gain_curve(predictions)

# dump(predictor, path/'cau_predictor.joblib')
# dump(predictions, path/'03181030_cau_pl_1234_151617.joblib')

# data = load_data.TurbineData('../../Data/', 'CAU')
# data.clear_wake_affected_turbines()
# data.select_baseline(inplace=True)
# data.select_normal_operation_times()
# 
# data.sample(frac=0.3, inplace=True)
# 
# predictor = GaussianWeightedAverage(10e-9)
# predictor.fit(data)

# data = load_data.TurbineData('../../Data/', 'ARD')
# data.select_normal_operation_times()
# turbines = data.data['instanceID'].drop_duplicates()
# data.sample(frac=0.1, inplace=True)

# predictions = predictor.predict(data, [1, 2, 3, 4], [5, 6, 7, 8])
# print(predictions)

#visualize.average_power_gain_curve_dataframes(data, predictor)

# r2_scores = {}
# for turbine in turbines:
#     target_path = \
#     Path(f'~/.turbines/scores/{turbine}_kernel_ridge_scores.joblib').expanduser()
#     target_r2_score = load(target_path)
# 
#     r2_scores[turbine] = target_r2_score
# 
# visualize.r2_matrix(r2_scores)

