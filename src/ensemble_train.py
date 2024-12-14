import random
import numpy as np
import os
import torch as torch
import wandb  # Import wandb
from load_data import load_EOD_data
from evaluator import evaluate
from model import get_loss, StockMixer
import pickle

np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

data_path = '../dataset'
market_name = 'SP500'  # NASDAQ / SP500
relation_name = 'wikidata'
stock_num = 497  # NASDAQ (1026) / SP500 (474) / SP500_2024 (497)
lookback_length = 16
epochs = 100
valid_index = 1560  # NASDAQ (756) / SP500 (1006)
test_index = 1948  # NASDAQ (756 + 252 = 1008) / SP500 (1006 + 253 = 1259)
fea_num = 11  # Number of features in consideration
market_num = 8  # NASDAQ (20) / SP500 (8)
steps = 1
learning_rate = 0.001
alpha = 0.1
scale_factor = 3
activation = 'GELU'

if fea_num > 8:
    political = "-political-ensemble-20-Donald-80-Kamala"
else:
    political = ""

# Initialize wandb
wandb.init(project='stock-prediction',
           name='stock-mixer-run-' + market_name + political)
wandb.config.update({
    "epochs": epochs,
    "learning_rate": learning_rate,
    "lookback_length": lookback_length,
    "alpha": alpha,
    "scale_factor": scale_factor,
    "activation": activation,
    "market_name": market_name,
    "fea_num": fea_num,
})

dataset_path = '../dataset/' + market_name
if market_name == "SP500":
    data_model_A = np.load('../dataset/SP500/reduced_sp500_2024_political.npy')
    data_model_B = np.load('../dataset/SP500/reduced_sp500_2024_political.npy')

    # Assign candidate_context = 1 for Model A for training and validation datasets
    data_model_A[:, :test_index, -2] = 1  # Training and validation are candidate 1
    # Test dataset should be candidate 1 (Donald Trump) since he won the election
    data_model_A[:, test_index:, -2] = 1

    # Assign candidate_context = 2 for Model B for training and validation datasets
    data_model_B[:, :test_index, -2] = 2  # Training and validation are candidate 2
    # Test dataset should be candidate 1 (Donald Trump) since he won the election
    data_model_B[:, test_index:, -2] = 1

    # -- Both models have the same information
    price_data = data_model_A[:, :, -1]
    mask_data = np.ones((data_model_A.shape[0], data_model_A.shape[1]))
    gt_data = np.zeros((data_model_A.shape[0], data_model_A.shape[1]))

    # Compute ground truth data (gt_data) for both models
    for ticket in range(data_model_A.shape[0]):
        for row in range(1, data_model_A.shape[1]):
            gt_data[ticket][row] = (data_model_A[ticket][row][-1] - data_model_A[ticket][row - steps][-1]) / \
                data_model_A[ticket][row - steps][-1]
else:
    with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f)
    with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
        mask_data = pickle.load(f)
    with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
        gt_data = pickle.load(f)
    with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
        price_data = pickle.load(f)

trade_dates = mask_data.shape[1]

# ---- Model A
model_A = StockMixer(
    stocks=stock_num,
    time_steps=lookback_length,
    channels=fea_num,
    market=market_num,
    scale=scale_factor
).to(device)

# ---- Model B
model_B = StockMixer(
    stocks=stock_num,
    time_steps=lookback_length,
    channels=fea_num,
    market=market_num,
    scale=scale_factor
).to(device)

optimizer_A = torch.optim.Adam(model_A.parameters(), lr=learning_rate)
optimizer_B = torch.optim.Adam(model_B.parameters(), lr=learning_rate)

best_valid_loss = np.inf
best_valid_perf = None
best_test_perf = None
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)


def validate(model, start_index, end_index, model_A=True):
    with torch.no_grad():
        cur_valid_pred = np.zeros(
            [stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros(
            [stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros(
            [stock_num, end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            if model_A:
                data_batch, mask_batch, price_batch, gt_batch = map(
                    lambda x: torch.Tensor(x).to(device),
                    get_batch_A(cur_offset)
                )
            else:
                data_batch, mask_batch, price_batch, gt_batch = map(
                    lambda x: torch.Tensor(x).to(device),
                    get_batch_B(cur_offset)
                )

            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     stock_num, alpha)
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset -
                           (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset -
                         (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset -
                           (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
    return cur_valid_pred, cur_valid_gt, cur_valid_mask, loss, reg_loss, rank_loss


def get_batch_A(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        data_model_A[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1)
    )


def get_batch_B(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        data_model_B[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1)
    )


for epoch in range(epochs):
    print(
        f"Epoch {epoch + 1} ##########################################################")

    # Train Model A
    print("Training Model A...")
    np.random.shuffle(batch_offsets)
    tra_loss_A, tra_reg_loss_A, tra_rank_loss_A = 0.0, 0.0, 0.0
    for j in range(valid_index - lookback_length - steps + 1):
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch_A(batch_offsets[j])
        )
        optimizer_A.zero_grad()
        prediction_A = model_A(data_batch)
        cur_loss_A, cur_reg_loss_A, cur_rank_loss_A, _ = get_loss(prediction_A, gt_batch, price_batch, mask_batch,
                                                                  stock_num, alpha)
        cur_loss_A.backward()
        optimizer_A.step()

        tra_loss_A += cur_loss_A.item()
        tra_reg_loss_A += cur_reg_loss_A.item()
        tra_rank_loss_A += cur_rank_loss_A.item()

    print(f"Model A Train: Loss = {tra_loss_A:.2e}")

    # Train Model B
    print("Training Model B...")
    np.random.shuffle(batch_offsets)
    tra_loss_B, tra_reg_loss_B, tra_rank_loss_B = 0.0, 0.0, 0.0
    for j in range(valid_index - lookback_length - steps + 1):
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch_B(batch_offsets[j])
        )
        optimizer_B.zero_grad()
        prediction_B = model_B(data_batch)
        cur_loss_B, cur_reg_loss_B, cur_rank_loss_B, _ = get_loss(prediction_B, gt_batch, price_batch, mask_batch,
                                                                  stock_num, alpha)
        cur_loss_B.backward()
        optimizer_B.step()

        tra_loss_B += cur_loss_B.item()
        tra_reg_loss_B += cur_reg_loss_B.item()
        tra_rank_loss_B += cur_rank_loss_B.item()

    print(f"Model B Train: Loss = {tra_loss_B:.2e}")

    # ----------------------
    # Validate both models separately
    val_predictions_A, val_gt, val_mask, val_loss_A, val_reg_loss_A, val_rank_loss_A = validate(
        model_A, valid_index, test_index, model_A=True)
    val_predictions_B, _, _, val_loss_B, val_reg_loss_B, val_rank_loss_B = validate(
        model_B, valid_index, test_index, model_A=False)

    # Weighted ensemble for validation predictions
    weight_A = 0.2  # Weight for candidate 1 (Trump)
    weight_B = 0.8  # Weight for candidate 2 (Harris)
    val_ensemble_predictions = weight_A * val_predictions_A + weight_B * val_predictions_B

    # Evaluate the ensemble
    val_perf = evaluate(val_ensemble_predictions, val_gt, val_mask)

    # Print validation performance
    print('Validation (Model A): loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(
        val_loss_A, val_reg_loss_A, val_rank_loss_A))
    print('Validation (Model B): loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(
        val_loss_B, val_reg_loss_B, val_rank_loss_B))
    print('Validation (Ensemble): mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
        val_perf['mse'], val_perf['IC'], val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))

    # Test both models separately
    test_predictions_A, test_gt, test_mask, test_loss_A, test_reg_loss_A, test_rank_loss_A = validate(
        model_A, test_index, trade_dates)
    test_predictions_B, _, _, test_loss_B, test_reg_loss_B, test_rank_loss_B = validate(
        model_B, test_index, trade_dates)

    # Weighted ensemble for test predictions
    test_ensemble_predictions = weight_A * \
        test_predictions_A + weight_B * test_predictions_B

    # Evaluate the ensemble
    test_perf = evaluate(test_ensemble_predictions, test_gt, test_mask)

    # Print test performance
    print('Test (Model A): loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(
        test_loss_A, test_reg_loss_A, test_rank_loss_A))
    print('Test (Model B): loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(
        test_loss_B, test_reg_loss_B, test_rank_loss_B))
    print('Test (Ensemble): mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
        test_perf['mse'], test_perf['IC'], test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']))
    # ----------------------

    # Log metrics for individual models and ensemble
    wandb.log({
        'epoch': epoch + 1,
        'Train Loss Model A': tra_loss_A,
        'Train Loss Model B': tra_loss_B,
        'Train Reg Loss Model A': tra_reg_loss_A,
        'Train Reg Loss Model B': tra_reg_loss_B,
        'Train Rank Loss Model A': tra_rank_loss_A,
        'Train Rank Loss Model B': tra_rank_loss_B,
        'Valid Loss Model A': val_loss_A,
        'Valid Loss Model B': val_loss_B,
        'Test Loss Model A': test_loss_A,
        'Test Loss Model B': test_loss_B,
        'Valid MSE': val_perf['mse'],
        'Test MSE': test_perf['mse'],
        'Valid IC': val_perf['IC'],
        'Test IC': test_perf['IC'],
        'Valid RIC': val_perf['RIC'],
        'Test RIC': test_perf['RIC'],
        'Valid prec@10': val_perf['prec_10'],
        'Test prec@10': test_perf['prec_10'],
        'Valid SR': val_perf['sharpe5'],
        'Test SR': test_perf['sharpe5']
    })

    # Update the best validation and test performance based on ensemble
    if val_perf['mse'] < best_valid_loss:  # Use ensemble's validation loss
        best_valid_loss = val_perf['mse']
        best_valid_perf = val_perf
        best_test_perf = test_perf

    # Print debugging performance metrics
    print("\n=== Validation Performance ===")

    print(f"\nEnsemble:")
    print(f"    MSE:     {val_perf['mse']:.2e}")
    print(f"    IC:      {val_perf['IC']:.2e}")
    print(f"    RIC:     {val_perf['RIC']:.2e}")
    print(f"    Prec@10: {val_perf['prec_10']:.2e}")
    print(f"    SR:      {val_perf['sharpe5']:.2e}")

    print("\n=== Test Performance ===")

    print(f"\nEnsemble:")
    print(f"    MSE:     {test_perf['mse']:.2e}")
    print(f"    IC:      {test_perf['IC']:.2e}")
    print(f"    RIC:     {test_perf['RIC']:.2e}")
    print(f"    Prec@10: {test_perf['prec_10']:.2e}")
    print(f"    SR:      {test_perf['sharpe5']:.2e}")
    print("\n")

# End of training: Print the best validation and test performance
print("\n" + "=" * 20 + " Training Complete " + "=" * 20)
print(f"Best Validation Loss: {best_valid_loss:.2e}")

print("\nBest Validation Performance:")
print(f"    MSE:     {best_valid_perf['mse']:.2e}")
print(f"    IC:      {best_valid_perf['IC']:.2e}")
print(f"    RIC:     {best_valid_perf['RIC']:.2e}")
print(f"    Prec@10: {best_valid_perf['prec_10']:.2e}")
print(f"    SR:      {best_valid_perf['sharpe5']:.2e}")

print("\nBest Test Performance (corresponding to best validation):")
print(f"    MSE:     {best_test_perf['mse']:.2e}")
print(f"    IC:      {best_test_perf['IC']:.2e}")
print(f"    RIC:     {best_test_perf['RIC']:.2e}")
print(f"    Prec@10: {best_test_perf['prec_10']:.2e}")
print(f"    SR:      {best_test_perf['sharpe5']:.2e}")

# Finish the wandb run
wandb.finish()
