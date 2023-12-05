from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error, r2_score
import time
import pandas as pd
import numpy as np


def RealDataExperiments(data_files, targets, estimators={}, n_iterations=100, test_prop = 0.3, seed = None, polynomial = None, classification = False):
    results = {}
    for j, data_file in enumerate(data_files):
        # Read X and y csv files
        data_name = data_file.split('.')[1].split('\\')[-1]
        data = pd.read_csv(data_file)
        target = targets[j]
        X = data.drop([target], axis = 1)
        y = data[target]
        
        print(data_name)

        # Perform one hot encoding on categorical variables in X
        categorical_cols = [col for col in X.columns if X[col].dtype in ['object', 'category']]
        encoder = OneHotEncoder(drop = 'first')
        encoded_X = encoder.fit_transform(X[categorical_cols])
        X = pd.concat([X.drop(categorical_cols, axis=1), pd.DataFrame(encoded_X.toarray(), columns=encoder.get_feature_names_out(categorical_cols))], axis=1)
        
        if polynomial is not None:
            poly = PolynomialFeatures(degree=polynomial, include_bias=False)
            X_poly = poly.fit_transform(X)
            X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
            
            ppoly = X_poly.shape[1]
            npoly = X_poly.shape[0]
            
            if npoly*ppoly > 35000000 :
                #for col in X.columns:
                #    del X_poly[col]
                X_poly = X_poly.drop(X.columns, axis=1)
                pnew = np.ceil(35000000/npoly)
                X_poly = X_poly.iloc[:, np.random.choice(X_poly.shape[1], size=int(pnew), replace=False)]
                X = pd.concat([X, X_poly], axis=1)
            else:
                X = X_poly
            print(X.shape)
      
        estimator_results = {}
        for estimator_name, estimator in estimators.items():
            estimator_results[estimator_name] = {'mse': [], 'r2': [], 'time': [], 'p':[], 'lambda':[], 'iter':[], 'CA':[], 'q':[]}
            
        if seed is not None:
            np.random.seed(seed)
            print('.', end = '')
        
        for i in range(n_iterations):
            print(i, end = '')
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop)
            std = X_train.std()
            non_zero_std_cols = std[std != 0].index   # Select the columns with a non-zero standard deviation
            X_train = X_train[non_zero_std_cols]
            X_test = X_test[non_zero_std_cols]
            
            for estimator_name, estimator in estimators.items():
                
                start_time = time.time()
                estimator.fit(X_train, y_train)
                end_time = time.time()
                
                # Make predictions on the test data
                
                
                if classification:
                    score = estimator.score(X_test, y_test)
                    estimator_results[estimator_name]['CA'].append(score)
                    estimator_results[estimator_name]['p'].append(X_train.shape[1])
                    estimator_results[estimator_name]['q'].append(len(estimator.classes))
                    
                    print(score, end = '--')
                    
                else:
                    y_pred = estimator.predict(X_test)

                    # Compute the prediction error (root mean squared error)
                    mse = mean_squared_error(y_test, y_pred)
                    estimator_results[estimator_name]['mse'].append(mse)

                    # Compute the R2 score
                    r2 = r2_score(y_test, y_pred)
                    estimator_results[estimator_name]['r2'].append(r2)
                    
                    estimator_results[estimator_name]['p'].append(len(estimator.coef_))
                    estimator_results[estimator_name]['lambda'].append(estimator.alpha_)

                # Record the time taken to fit the estimator
                elapsed_time = end_time - start_time
                estimator_results[estimator_name]['time'].append(elapsed_time)
                
                
                
                if estimator_name == "EM":
                    estimator_results[estimator_name]['iter'].append(estimator.iterations_)
                
        # Compute the average results for each estimator
        data_results = {}
        for estimator_name, estimator_result in estimator_results.items():
            avg_mse = np.mean(estimator_result['mse'])
            avg_r2 = np.mean(estimator_result['r2'])
            avg_time = np.mean(estimator_result['time'])
            avg_p = np.mean(estimator_result['p'])
            avg_lambda = np.mean(estimator_result['lambda'])
            avg_CA = np.mean(estimator_result['CA'])
            avg_q = np.mean(estimator_result['q'])
            
            if estimator_name == "EM":
                avg_iter = np.mean(estimator_result['iter'])
            else:
                avg_iter = 100
                
            data_results[estimator_name] = {'mse': avg_mse, 'r2': avg_r2, 'time': avg_time, 'p': avg_p, 'n_train': X_train.shape[0],
                                           'lambda': avg_lambda, 'iter': avg_iter, 'CA': avg_CA, 'q': avg_q}
                
            

        # Store the data and its results in the dictionary
        results[data_name] = data_results

    return results


