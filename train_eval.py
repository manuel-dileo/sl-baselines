from sklearn.metrics import root_mean_squared_error, mean_absolute_error
def train(model, data, optimizer, criterion, num_epochs=100, verbose=False):
    mae_val_max = 0
    tol = 5e-2
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_predicted = model(data)
        loss = criterion(y_predicted[data.train_mask], data.y[data.train_mask].type_as(y_predicted))
        if verbose:
            print(f'Epoch: {epoch}, loss: {loss}')
        loss.backward()
        optimizer.step()
        mae_val, _ = test(model, data, data.val_mask)
        if mae_val_max - tol <= mae_val:
            mae_val_max = mae_val
        else:
            break

    return model

def test(model, data, mask):
    model.eval()
    out = model(data)
    y_pred = out[mask].cpu().detach().numpy()
    y_true = data.y[mask].cpu().detach().numpy()
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mae, rmse