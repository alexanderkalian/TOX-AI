import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Choose whether using 'tanimoto' or 'fragments' feature engineering.
feature_engineering = 'fragments'

# Number of folds.
num_folds = 5

# Ensures correct specification of feature_engineering.
if feature_engineering not in ['tanimoto', 'fragments']:
    raise ValueError(f'Expected either "tanimoto" or "fragments" for feature_engineering variable - instead received "{feature_engineering}"')
else:
    filename = '../data/processed/' + feature_engineering + '_PCA_fold_{}.csv'
    output_file_txt = '../data/results/' + feature_engineering + '_results.txt'
    output_file_csv = '../data/results/' + feature_engineering + '_results.csv'


# Function for running MLPs with k-fold cross-validation,
def run_cross_fold_evaluation(k=5, base_filename='dataset_{}.csv', results_file_txt='../data/results/results.txt', results_file_csv='../data/results/results.csv'):
    accuracies = []
    results_data = []

    # Opens results file, for recording results.
    with open(results_file_txt, 'w') as f:
        # Iterates through folds.
        for fold in range(1, k + 1):
            print(f'Loading - fold {fold}')
            df = pd.read_csv(base_filename.format(fold))
            
            feature_cols = [col for col in df.columns if col.startswith('x')]
            X_train = df[df['split'] == 'train'][feature_cols].values
            y_train = df[df['split'] == 'train']['mutagenicity'].values
            X_test = df[df['split'] == 'test'][feature_cols].values
            y_test = df[df['split'] == 'test']['mutagenicity'].values

            print('Training MLP classifier.')
            clf = MLPClassifier(hidden_layer_sizes=(500, 500), max_iter=500, random_state=42)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

            print(f'Fold {fold} - accuracy: {acc:.4f}')
            f.write(f'Fold {fold} - accuracy: {acc:.4f}\n')
            results_data.append({'fold': fold, 'accuracy': acc})

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        summary = f'\nAverage Accuracy: {mean_acc:.4f}\nStandard Deviation: {std_acc:.4f}\n'

        print(summary)
        f.write(summary)

    # Write CSV output
    results_df = pd.DataFrame(results_data)
    results_df.loc[len(results_df)] = {'fold': 'average', 'accuracy': mean_acc}
    results_df.loc[len(results_df)] = {'fold': 'std_dev', 'accuracy': std_acc}
    results_df.to_csv(results_file_csv, index=False)


# Run functions.
if __name__ == '__main__':
    run_cross_fold_evaluation(
        k=num_folds,
        base_filename=filename,
        results_file_txt=output_file_txt,
        results_file_csv=output_file_csv
    )
