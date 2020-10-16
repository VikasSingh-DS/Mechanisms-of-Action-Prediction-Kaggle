import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("")
    
    folds = df.copy()
    mskf = MultilabelStratifiedKFold(n_splits=5)

    for f, (t_idx, v_idx) in enumerate(mskf.split(X=df, y=df.target.values)):
        folds.loc[v_idx, 'kfold'] = int(f)

    folds['kfold'] = folds['kfold'].astype(int)
    folds

    folds.to_csv("/home/vikasunix/workspace/MoA-kaggle/input/folds.csv", index=False)