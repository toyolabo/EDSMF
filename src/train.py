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
market_name = 'SP500' # NASDAQ / SP500 
relation_name = 'wikidata'
stock_num = 497 # NASDAQ (1026) / SP500 (474) / SP500_2024 (497)
lookback_length = 16
epochs = 100
clip = 0 # SP500 (915)
valid_index = 1560 # NASDAQ (756) / SP500 (1006)
test_index = 1948 # NASDAQ (756 + 252 = 1008) / SP500 (1006 + 253 = 1259)
fea_num = 11 # Number of features in consideration
market_num = 8 # NASDAQ (20) / SP500 (8)
steps = 1
learning_rate = 0.001
alpha = 0.1
scale_factor = 3
activation = 'GELU'

if fea_num > 8:
    political = "-political-new-architecure"
else:
    political = ""

# Initialize wandb
wandb.init(project='stock-prediction', name='stock-mixer-run-' + market_name + political)
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
    # data = np.load('../dataset/SP500/my_SP500.npy')
    # data = np.load('../dataset/SP500/reduced_sp500_2024.npy')
    data = np.load('../dataset/SP500/reduced_sp500_2024_political.npy')
    # data = data[:, 915:, :]
    data = data[:, clip:, :]
    price_data = data[:, :, -1]
    mask_data = np.ones((data.shape[0], data.shape[1]))
    eod_data = data
    gt_data = np.zeros((data.shape[0], data.shape[1]))
    for ticket in range(0, data.shape[0]):
        for row in range(1, data.shape[1]):
            gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                                   data[ticket][row - steps][-1]
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
model = StockMixer(
    stocks=stock_num,
    time_steps=lookback_length,
    channels=fea_num,
    market=market_num,
    scale=scale_factor
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_valid_loss = np.inf
best_valid_perf = None
best_test_perf = None
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)


def validate(start_index, end_index):
    with torch.no_grad():
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     stock_num, alpha)
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return loss, reg_loss, rank_loss, cur_valid_perf


def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))


for epoch in range(epochs):
    print("epoch{}##########################################################".format(epoch + 1))
    np.random.shuffle(batch_offsets)
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0
    for j in range(valid_index - lookback_length - steps + 1):
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(batch_offsets[j])
        )
        optimizer.zero_grad()
        prediction = model(data_batch)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                            stock_num, alpha)
        cur_loss = cur_loss
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()
    tra_loss = tra_loss / (valid_index - lookback_length - steps + 1)
    tra_reg_loss = tra_reg_loss / (valid_index - lookback_length - steps + 1)
    tra_rank_loss = tra_rank_loss / (valid_index - lookback_length - steps + 1)
    print('Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
    print('Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss))

    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
    print('Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss))

    # Log metrics to wandb
    wandb.log({
        'epoch': epoch + 1,
        'Train Loss': tra_loss,
        'Train Reg Loss': tra_reg_loss,
        'Train Rank Loss': tra_rank_loss,
        'Valid Loss': val_loss,
        'Test Loss': test_loss,
        'Valid MSE': val_perf['mse'],
        'Valid IC': val_perf['IC'],
        'Valid RIC': val_perf['RIC'],
        'Valid prec@10': val_perf['prec_10'],
        'Valid SR': val_perf['sharpe5'],
        'Test MSE': test_perf['mse'],
        'Test IC': test_perf['IC'],
        'Test RIC': test_perf['RIC'],
        'Test prec@10': test_perf['prec_10'],
        'Test SR': test_perf['sharpe5']
    })

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        best_valid_perf = val_perf
        best_test_perf = test_perf

    print('Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(val_perf['mse'], val_perf['IC'],
                                                     val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
    print('Test performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(test_perf['mse'], test_perf['IC'],
                                                                            test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']), '\n\n')

# End of training: Print the best validation and test performance
print("\n==================== Training Complete ====================")
print("Best Validation Loss: {:.2e}".format(best_valid_loss))
print("Best Validation Performance:")
print("    mse: {:.2e}, IC: {:.2e}, RIC: {:.2e}, prec@10: {:.2e}, SR: {:.2e}".format(
    best_valid_perf['mse'], best_valid_perf['IC'], best_valid_perf['RIC'],
    best_valid_perf['prec_10'], best_valid_perf['sharpe5']))
print("Best Test Performance (corresponding to best validation):")
print("    mse: {:.2e}, IC: {:.2e}, RIC: {:.2e}, prec@10: {:.2e}, SR: {:.2e}".format(
    best_test_perf['mse'], best_test_perf['IC'], best_test_perf['RIC'],
    best_test_perf['prec_10'], best_test_perf['sharpe5']))


# Finish the wandb run
wandb.finish()
