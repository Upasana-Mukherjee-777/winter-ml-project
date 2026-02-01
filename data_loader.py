import pandas as pd

def load_fd002_data(data_dir):
    """
    Load NASA C-MAPSS FD002 dataset with robust whitespace handling.
    """

    columns = (
        ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] +
        [f'sensor_{i}' for i in range(1, 22)]
    )

    train_df = pd.read_csv(
        f"{data_dir}/train_FD002.txt",
        sep=r"\s+",
        header=None,
        names=columns
    )

    test_df = pd.read_csv(
        f"{data_dir}/test_FD002.txt",
        sep=r"\s+",
        header=None,
        names=columns
    )

    rul_df = pd.read_csv(
        f"{data_dir}/RUL_FD002.txt",
        header=None,
        names=["RUL_actual"]
    )

    return train_df, test_df, rul_df
