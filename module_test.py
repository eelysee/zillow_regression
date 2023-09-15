from sklearn.model_selection import train_test_split

def test_split(test, target):
    '''
    splits test data to target
    TAKES one df and target column in that df in '' quotes
    RETURNS one columm df
    '''
    X_test = test.drop(columns=[target])
    y_test = test[target]
    return X_test, y_test
    