import torch


def train_epoch(model, data, optimizer, opts, exp_num=None, criterion=None):
    model.train()
    optimizer.zero_grad()

    if opts.embedding:
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
    elif opts.problem == 'Prediction':
        output = model(data)
        loss = criterion(output[data.train_mask], data.y[data.train_mask, exp_num].reshape([-1, 1]))
    elif opts.problem == 'Imputation_eval':
        # forward pass, random guess
        output = model(data)
        # begin backward pass, how different is the guessed output and y
        loss = criterion(output * (data.train_mask), data.y * (data.train_mask))
    else:
        output = model(data)
        loss = criterion(output * (data.nonzeromask), data.y * (data.nonzeromask))
    # backward pass: updates the weights using chain rule / back propagation
    loss.backward()
    optimizer.step()
    return loss.item()

# decorator: takes in function, returns modified version of function
@torch.no_grad()
def test(model, data, exp_num, criterion, opts):
    model.eval()

    if opts.embedding:
        lr_out, rf_out = model.predict(data.x, data.edge_index)
        loss_lr = criterion(lr_out[data.test_mask], data.y[data.test_mask, exp_num].cpu().data.numpy())
        loss_rf = criterion(rf_out[data.test_mask], data.y[data.test_mask, exp_num].cpu().data.numpy())
        return loss_lr, loss_rf
    elif opts.problem == 'Prediction':
        output = model(data)
        loss = criterion(output[data.test_mask], data.y[data.test_mask, exp_num].reshape([-1, 1]))
    else:
        output = model(data)
        loss = criterion(output*data.test_mask, data.y*data.test_mask)
    return loss.item()
