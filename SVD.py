# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:41:46 2019

@author: MOHAMMAD.SHAMOON
"""

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate


data = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(data, test_size=.25)

algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)

uid = str(879)   #user id 
iid = str(294)   #item id 

pred = algo.predict(uid, iid, r_ui=5, verbose=True)

# Then compute RMSE
accuracy.rmse(predictions)

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

