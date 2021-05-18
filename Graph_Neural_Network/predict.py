"""
Computationally expensive because it predicts each gene separately;
We can use this when we predict on a small number of genes.
"""
import torch
from sklearn.metrics import mean_squared_error as scimse
from torch_geometric.utils import to_undirected
import numpy as np
from sklearn.model_selection import KFold
from train_test import train_epoch, test
import copy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from utils.functions import index_to_mask


def predict(model_class, data, opts):

    # Define lists used to store the training and test losses
    train_losses,test_losses = [],[]
    # Define lists used to store the mean MSE (across all response features) for the training and test sets
    mean_train_losses,mean_test_losses = [],[]

    # Define the objective function (MSE)
    criterion = torch.nn.MSELoss()

    # Define the splits for the genes (train/test) and the splits for the features (used to train the model)
    kf_genes = KFold(n_splits=3, random_state=opts.seed)
    kf_features = KFold(n_splits=3, random_state=opts.seed)

    # Iterate over every gene fold (train/test split)
    for k, train_test_indices in enumerate(kf_genes.split(data.x)):
        print('Fold number: {:d}'.format(k))

        # Define lists used to store the training and test set MSE scores for each response feature
        train_mse,test_mse = [],[]

        # Define a list used to store the test set predictions for each feature
        y_pred = []

        eval_data = copy.deepcopy(data)

        # Define the training and test set indices
        train_index, test_index = train_test_indices
        # Define the predictor and response feature indices (these are different for each train/test split)
        x_feat_indices, y_feat_indices = next(kf_features.split(np.arange(data.y.size(1))))

        # Define the predictor and response data (WHERE DOES HE DEFINE data.y BEFORE THIS)
        eval_data.x = data.x[:, x_feat_indices]
        eval_data.y = data.x[:, y_feat_indices]
        
        # Define the masks for the training and test sets
        train_mask = index_to_mask(train_index, eval_data.x.size(0))
        test_mask = index_to_mask(test_index, eval_data.x.size(0))

        # Iterate over every response feature
        for exp_num in range(eval_data.y.size(1)):

            # If performing linear regression or random forest regression
            if (model_class == LinearRegression) | (model_class == RandomForestRegressor):
                model = model_class()

                # Fit the model the the training set (a separate model is fit for each response feature)
                model.fit(eval_data.x[train_mask], eval_data.y[train_mask, exp_num])
                # Create predictions for the training and test sets
                train_pred = model.predict(eval_data.x[train_mask])
                test_pred = model.predict(eval_data.x[test_mask])

                # Determine the training and test set MSE scores
                train_mse.append(scimse(train_pred, eval_data.y[train_mask, exp_num]))
                test_mse.append(scimse(test_pred, eval_data.y[test_mask, exp_num]))

                print('Exp: {:03d}, Loss: {:.5f}'.format(exp_num, test_mse[-1]))

                # Append the predictions for the given feature
                y_pred.append(pred)
            else:
                torch.manual_seed(opts.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(opts.seed)

                # Define the model
                model = model_class(eval_data.num_features, opts).to(opts.device)
                # Define the optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
        
                best_loss = 1e9

                # Iterate over every epoch
                for epoch in range(1, opts.epochs + 1):

                    # Train the model and compute the training set loss
                    train_loss = train_epoch(model, eval_data, optimizer, opts, exp_num, criterion)

                    # Determine which set set of weights (model) is associated with the smallest training set loss
                    if train_loss < best_loss:
                        best_loss = train_loss
                        best_model = copy.deepcopy(model)
                
                # Compute the test set loss
                test_loss = test(best_model, eval_data, exp_num, criterion, opts)
                print('Exp: {:03d}, Loss: {:.5f}, TestLoss: {:.5f}'.format(exp_num, loss_train, loss_test))

                # Append the predictions for the given feature
                with torch.no_grad():
                    y_pred.append(best_model(eval_data))

        # Iterate over every response feature
        for i in range(eval_data.y.size(1)):
            # If linear regression or random forest regression was used to create the predictions
            if (model_class == LinearRegression) | (model_class == RandomForestRegressor):
                # Compute the training and test set MSE scores
                train_mse.append(scimse(y_pred[i], eval_data.y[train_mask, i]))
                test_mse.append(scimse(y_pred[i], eval_data.y[test_mask, i]))
            # If end-to-end prediction is performed
            else:
                # Compute the training and test set MSE scores
                train_mse.append(scimse(y_pred[i][train_mask.cpu().numpy()].cpu().numpy(), eval_data.y[train_mask, i].cpu().numpy().reshape([-1, 1])))
                test_mse.append(scimse(y_pred[i][test_mask.cpu().numpy()].cpu().numpy(), eval_data.y[test_mask, i].cpu().numpy().reshape([-1, 1])))

        # Compute the mean MSE (across all response features) for the training and test sets
        mean_train_losses.append(np.mean(train_mse))
        mean_test_losses.append(np.mean(test_mse))

    print(f'The MSE scores for each fold for the training set are as follows: {mean_train_losses}')
    print(f'The MSE scores for each fold for the test set are as follows: {mean_tmean_test_lossesrain_losses}')
    return mean_train_losses,mean_test_losses