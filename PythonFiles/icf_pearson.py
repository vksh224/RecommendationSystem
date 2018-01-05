from surprise import KNNBasic
from surprise import Dataset
from surprise import evaluate, print_perf
import os
from surprise import Reader
#load data from a file
file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=3)
algo = KNNBasic(sim_options = {'name':'pearson','user_based': False })
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)