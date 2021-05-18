import torch
from sklearn.model_selection import KFold
from train_test import train_epoch, test
import copy

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def impute(model_class, data, opts):

    # Define the loss (MSE)
    criterion = torch.nn.MSELoss()

    # Define the splits for train/test (masked and unmasked values in the expression matrix)
    kf = KFold(n_splits=3, random_state=opts.seed, shuffle=True)

    # Define a list used to store the test MSE for each fold
    loss_test = []


    indices = np.indices([data.x.size(0), data.x.size(1)]).reshape(2, -1)
        
    # Iterate over every train/test split; indices go up to the total number of elements
    for k, train_test_indices in enumerate(kf.split(np.arange(data.x.shape(0)))):
        print('Fold number: {:d}'.format(k))
        
        # two lists of indices up to num of non-zero elements, which is roughly 480*10k
        train_index, test_index = train_test_indices
        eval_data = copy.deepcopy(data)

        # Define the train and test masks (the test values are the ones that are masked)
        # In imputation, train-test is like X-y
        eval_data.train_mask = index_to_mask([indices[0, train_index], indices[1, train_index]],
                                             eval_data.x.size()).to(opts.device)
        eval_data.test_mask = index_to_mask([indices[0, test_index], indices[1, test_index]],
                                            eval_data.x.size()).to(opts.device)

        print(eval_data.num_features)
        model = model_class(eval_data.num_features, opts).to(opts.device)
        # add milestones to decay the learning rate so can continue to reduce training loss
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
        best_loss = 1e9

        # Iterate over every epoch
        for epoch in range(1, opts.epochs + 1):
            loss_train = train_epoch(model, eval_data, optimizer, opts, criterion=criterion)

            # Determine the best loss and the associated weights (this is used as the final model)
            if loss_train < best_loss:
                best_loss = loss_train
                best_model = copy.deepcopy(model)
            if epoch % 10 == 0:
                print('Epoch number: {:03d}, Train_loss: {:.5f}'.format(epoch, loss_train))
                
        # Determine the test MSE
        loss_test.append(test(best_model, eval_data, None, criterion, opts))
        print('Loss: {:.5f}, TestLoss: {:.5f}'.format(loss_train, loss_test[k]))
        
    print('Average+-std Error for test RNA values: {:.5f}+-{:.5f}'.format(np.mean(loss_test), np.std(loss_test)))
    return np.mean(loss_test)


"""
np.indices([3,5]).reshape(2,-1)
>>> array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
           [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])
"""
