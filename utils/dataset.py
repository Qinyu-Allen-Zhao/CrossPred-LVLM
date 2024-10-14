import numpy as np
from torch.utils.data import Dataset, DataLoader

class RecommDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        user_id, item_id, user_profile, item_profile, label = self.X[idx]["user_id"], self.X[idx]["item_id"], self.X[idx]["user_profile"], self.X[idx]["item_profile"], self.y[idx]
        return user_id, item_id, user_profile, item_profile, label


def get_dataloaders(result_summary, num_user, num_item, model_profile, data_profile, batch_size=128):
    # Mask some parts of the matrix
    mask_matrix = np.random.choice([0, 1], size=(num_user, num_item), p=[0.8, 0.2])

    data = result_summary.values.copy()
    data[mask_matrix == 1] = np.nan

    # Normalize data based on observed values
    mean_train, std_train = np.nanmean(data, axis=0), np.nanstd(data, axis=0)
    data_norm = (result_summary.values.copy() - mean_train) / std_train

    # Split data into train and test
    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(num_user):
        for j in range(num_item):
            ins = {
                    "user_id": i, 
                    "item_id": j, 
                    "user_profile": model_profile.get_profile(i),
                    "item_profile": data_profile.get_profile(j)
                }
            

            if mask_matrix[i, j] == 0:
                X_train.append(ins)
                y_train.append(data_norm[i, j])
            else:
                X_test.append(ins)
                y_test.append(data_norm[i, j])

    print("Train size: ", len(X_train))
    print("Test size: ", len(X_test))

    train_set = RecommDataset(X_train, y_train)
    test_set = RecommDataset(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, mean_train, std_train