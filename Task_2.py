#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from surprise import KNNBasic, SVD, Reader, accuracy, Dataset
from surprise.model_selection import cross_validate, train_test_split
%matplotlib inline

#Read Dataset - CSV file
movie_ratings = pd.read_csv('ratings_small.csv')
movie_ratings

#PERFORMANCE EVALUATION OF VARIOUS MODELS

#Probablity Mass Function

reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

SVD_pmf = SVD(biased = False)
crossVal_pmf = cross_validate(SVD_pmf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);

crossVal_pmf

print('Mean MAE for PMF Collaborative Filtering: ', crossVal_pmf['test_mae'].mean())
print('Mean RMSE for PMF Collaborative Filtering: ', crossVal_pmf['test_rmse'].mean())

#User Based Collaborative Filtering

sim_options = {'user_based': True}
cf_user_based = KNNBasic(sim_options=sim_options)
cv_ub = cross_validate(cf_user_based, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);

print('Mean MAE for User-based Collaborative Filtering: ', cv_ub['test_mae'].mean())
print('Mean RMSE for User-based Collaborative Filtering: ', cv_ub['test_rmse'].mean())

#Item Based Collaborative Filtering

sim_options = {'user_based': False}
cf_item_based = KNNBasic(sim_options=sim_options)
cv_ib = cross_validate(cf_item_based, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);

print('Mean MAE for Item-based Collaborative Filtering: ', cv_ib['test_mae'].mean())
print('Mean RMSE for Item-based Collaborative Filtering: ', cv_ib['test_rmse'].mean())


#COMPARISON OF SIMILARITY METRICS

#User Based Collaborative Filtering

#Cosine
sim_options = {'name':'cosine', 'user_based': True}
cosine_ub = KNNBasic(sim_options=sim_options);
cosine__cv_ub = cross_validate(cosine_ub, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);

#MSD
sim_options = {'name':'msd', 'user_based': True}
ub_msd = KNNBasic(sim_options=sim_options);
cv_ub_msd = cross_validate(ub_msd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);

#Pearson
sim_options = {'name':'pearson', 'user_based': True}
ub_pearson = KNNBasic(sim_options=sim_options);
cv_ub_pearson = cross_validate(ub_pearson, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);

#Observation
x = [0,1,2]
y_mae = [cosine__cv_ub['test_mae'].mean(),cv_ub_msd['test_mae'].mean(),cv_ub_pearson['test_mae'].mean()]
y_rmse = [cosine__cv_ub['test_rmse'].mean(),cv_ub_msd['test_rmse'].mean(),cv_ub_pearson['test_rmse'].mean()]
plt.plot(x, y_mae)
plt.plot(x, y_rmse)
plt.title('User-based Collaborative Filtering(With 5-fold CV)')
plt.legend(['MAE','RMSE'])
plt.xticks(x,['Cosine','MSD','Pearson'])
plt.xlabel('Similarity')
plt.ylabel('Average Test(MAE & RMSE)')
plt.show()

#Item Based Collaborative Filtering

#Cosine
sim_options = {'name':'cosine', 'user_based': False}
ib_cosine = KNNBasic(sim_options=sim_options);
cv_ib_cosine = cross_validate(ib_cosine, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);

#MSD
sim_options = {'name':'msd', 'user_based': False}
ib_msd = KNNBasic(sim_options=sim_options);
cv_ib_msd = cross_validate(ib_msd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);

#Pearson
sim_options = {'name':'pearson', 'user_based': False}
ib_pearson = KNNBasic(sim_options=sim_options);
cv_ib_pearson = cross_validate(ib_pearson, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);

#Observation
x = [0,1,2]
y_mae = [cv_ib_cosine['test_mae'].mean(),cv_ib_msd['test_mae'].mean(),cv_ib_pearson['test_mae'].mean()]
y_rmse = [cv_ib_cosine['test_rmse'].mean(),cv_ib_msd['test_rmse'].mean(),cv_ib_pearson['test_rmse'].mean()]
plt.plot(x, y_mae)
plt.plot(x, y_rmse)
plt.title('Item-based Collaborative Filtering(With 5-fold CV)')
plt.legend(['MAE','RMSE'])
plt.xticks(x,['Cosine','MSD','Pearson'])
plt.xlabel('Similarity')
plt.ylabel('Average Test(MAE & RMSE)')
plt.show()

#IMPACT OF NUMBER OF NEIGHBORS ON MODEL

trainset, testset = train_test_split(data, test_size = 0.25, random_state = 42)

#User based Collaborative Filtering

ubc_nn_mae = []
ubc_nn_rmse = []
k1 = list(np.arange(1,100,1))
for i in k1:
  ubc_nn = KNNBasic(k = i, sim_options = {'user_based' : True})
  ubc_nn.fit(trainset)
  predictions = ubc_nn.test(testset)
  ubc_nn_mae.append(accuracy.mae(predictions))
  ubc_nn_rmse.append(accuracy.rmse(predictions))
  
plt.plot(k1,ubc_nn_mae)
plt.plot(k1,ubc_nn_rmse)
plt.xlabel('Number of Neighbors')
plt.ylabel('Testset(MAE & RMSE)')
plt.legend(['MAE','RMSE'])
plt.title('User-based Collaborative Filtering')
plt.show()

#Best K
k_ubc = ubc_nn_rmse.index(min(ubc_nn_rmse))+1
print('Best Value of K : ', k_ubc)
print('Minimum RMSE : ', min(ubc_nn_rmse))

#Item Based Collaborative Filtering

ibc_nn_mae = []
ibc_nn_rmse = []
for i in k1:
  ibc_nn = KNNBasic(k = i, sim_options = {'user_based' : False})
  ibc_nn.fit(trainset)
  predictions = ibc_nn.test(testset)
  ibc_nn_mae.append(accuracy.mae(predictions))
  ibc_nn_rmse.append(accuracy.rmse(predictions))
  
plt.plot(k1,ibc_nn_mae)
plt.plot(k1,ibc_nn_rmse)
plt.xlabel('Number of Neighbors')
plt.ylabel('Testset(MAE & RMSE)')
plt.legend(['MAE','RMSE'])
plt.title('Item-based Collaborative Filtering')
plt.show()

#Best K
k_ibc = ibc_nn_rmse.index(min(ibc_nn_rmse))+1
print('Best Value of K : ', k_ibc)
print('Minimum RMSE : ', min(ibc_nn_rmse))