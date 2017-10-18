import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
import lightgbm as lgb


def Preprocessor(train_data, test_data):
    """
    :param train_data: a pandas dataframe representing train data set
    :param test_data: a pandas dataframe representing test dataset
    :return: 4 pandas dataframe: train_data_x, train_data_y, test_data_x, test_idx
    """

    train_data_y = train_data['IsBadBuy']
    train_data_x = train_data.drop(['RefId', 'IsBadBuy'], axis=1)

    test_idx = test_data['RefId']
    test_data_x = test_data.drop(['RefId'], axis=1)

    features = pd.concat([train_data_x, test_data_x])


    # convert to upper case
    for col in features.select_dtypes(include=["object"]).columns.values:
        features[col] = features[col].str.upper()

    # missing value imputation
    # Fill missing values in SubModel
    features.loc[[24567, 24578], 'SubModel'] = '4D SEDAN SXT FFV'
    features.loc[[70432], 'SubModel'] = 'REG CAB 2.2L FFV'
    features.loc[[70434], 'SubModel'] = '4D SEDAN'
    features.loc[[70437], 'SubModel'] = '4D SEDAN SE'
    features.loc[[70445], 'SubModel'] = '2D COUPE'
    features.loc[[70446], 'SubModel'] = '4D SUV 4.6L'
    features.loc[[70450], 'SubModel'] = 'PASSENGER 3.9L SE'
    features.loc[[19863], 'SubModel'] = 'SPORT UTILITY'
    features.loc[[19864], 'SubModel'] = '4D SUV LS'
    features.loc[[19867], 'SubModel'] = '4D SEDAN CLASSIC'
    features.loc[[30915], 'SubModel'] = 'MINIVAN 3.3L'
    features.loc[[37737], 'SubModel'] = '4D SEDAN'

    # Fill missing values in Transmission
    features.loc[[15906, 19863, 19864, 19867, 24567, 24578, 70432, 70434, 70437, 70445, 70446, 70450],
                 ["Transmission"]] = "AUTO"

    # Fill missing values in WheelType
    for index, row in features.iterrows():
        if str(row['WheelType']) == 'nan':
            tmp = features[(features['Model'] == row['Model'])
                           & (features['SubModel'] == row['SubModel'])]['WheelType'].value_counts()
            if (tmp.size > 0):
                features.loc[[index], 'WheelType'] = tmp.index[0]
            else:
                features.loc[[index], 'WheelType'] = 'ALLOY'

    # Fill missing values in Color using Mode
    features.loc[features['Color'].isnull(), 'Color'] = 'SILVER'
    features.loc[features['Color'] == 'NOT AVAIL', 'Color'] = 'SILVER'

    # Fill missing values in nationality, size and topthreeamericans
    features.loc[[10888], 'Nationality'] = 'AMERICAN'
    features.loc[[10888], 'Size'] = 'LARGE TRUCK'
    features.loc[[10888], 'TopThreeAmericanName'] = 'GM'
    features.loc[[25169, 2082], 'Nationality'] = 'AMERICAN'
    features.loc[[25169, 2082], 'Size'] = 'MEDIUM SUV'
    features.loc[[25169, 2082], 'TopThreeAmericanName'] = 'CHRYSLER'
    features.loc[[37986], 'Nationality'] = 'OTHER ASIAN'
    features.loc[[37986], 'Size'] = 'MEDIUM'
    features.loc[[37986], 'TopThreeAmericanName'] = 'OTHER'
    features.loc[[69948, 69958], 'Nationality'] = 'AMERICAN'
    features.loc[[69948, 69958], 'Size'] = 'SMALL SUV'
    features.loc[[69948, 69958], 'TopThreeAmericanName'] = 'CHRYSLER'
    features.loc[[20588], 'Nationality'] = 'AMERICAN'
    features.loc[[20588], 'Size'] = 'LARGE TRUCK'
    features.loc[[20588], 'TopThreeAmericanName'] = 'CHRYSLER'
    features.loc[[20589], 'Nationality'] = 'AMERICAN'
    features.loc[[20589], 'Size'] = 'MEDIUM SUV'
    features.loc[[20589], 'TopThreeAmericanName'] = 'GM'
    features.loc[[20594], 'Nationality'] = 'AMERICAN'
    features.loc[[20594], 'Size'] = 'COMPACT'
    features.loc[[20594], 'TopThreeAmericanName'] = 'GM'
    features.loc[[20595], 'Nationality'] = 'AMERICAN'
    features.loc[[20595], 'Size'] = 'LARGE'
    features.loc[[20595], 'TopThreeAmericanName'] = 'FORD'
    features.loc[[20597], 'Nationality'] = 'AMERICAN'
    features.loc[[20597], 'Size'] = 'MEDIUM'
    features.loc[[20597], 'TopThreeAmericanName'] = 'GM'
    features.loc[[20600], 'Nationality'] = 'AMERICAN'
    features.loc[[20600], 'Size'] = 'MEDIUM'
    features.loc[[20600], 'TopThreeAmericanName'] = 'CHRYSLER'

    # Fill missing value in MMRAcquisitionAuctionAveragePrice,MMRAcquisitionAuctionCleanPrice,
    # MMRAcquisitionRetailAveragePrice, MMRAcquisitonRetailCleanPrice
    # MMRAcquisitionAuctionAveragePrice
    mean_auc_avg_price = features['MMRAcquisitionAuctionAveragePrice'].dropna().values.mean()
    features.loc[features['MMRAcquisitionAuctionAveragePrice'].isnull(),
                 'MMRAcquisitionAuctionAveragePrice'] = mean_auc_avg_price

    # MMRAcquisitionAuctionCleanPrice
    mean_auc_clean_price = features['MMRAcquisitionAuctionCleanPrice'].dropna().values.mean()
    features.loc[features['MMRAcquisitionAuctionCleanPrice'].isnull(),
                 'MMRAcquisitionAuctionCleanPrice'] = mean_auc_clean_price

    # MMRAcquisitionRetailAveragePrice
    mean_retail_avg_price = features['MMRAcquisitionRetailAveragePrice'].dropna().values.mean()
    features.loc[features['MMRAcquisitionRetailAveragePrice'].isnull(),
                 'MMRAcquisitionRetailAveragePrice'] = mean_retail_avg_price

    # MMRAcquisitonRetailCleanPrice
    mean_retail_clean_price = features['MMRAcquisitonRetailCleanPrice'].dropna().values.mean()
    features.loc[features['MMRAcquisitonRetailCleanPrice'].isnull(),
                 'MMRAcquisitonRetailCleanPrice'] = mean_retail_clean_price

    # Fill Missing value in MMRCurrentAuctionAveragePrice, MMRCurrentAuctionCleanPrice,
    # MMRCurrentRetailAveragePrice, MMRCurrentRetailCleanPrice
    # MMRAcquisitionAuctionAveragePrice
    mean_auc_avg_price = features['MMRCurrentAuctionAveragePrice'].dropna().values.mean()
    features.loc[features['MMRCurrentAuctionAveragePrice'].isnull(),
                 'MMRCurrentAuctionAveragePrice'] = mean_auc_avg_price

    # MMRAcquisitionAuctionCleanPrice
    mean_auc_clean_price = features['MMRCurrentAuctionCleanPrice'].dropna().values.mean()
    features.loc[features['MMRCurrentAuctionCleanPrice'].isnull(),
                 'MMRCurrentAuctionCleanPrice'] = mean_auc_clean_price

    # MMRAcquisitionRetailAveragePrice
    mean_retail_avg_price = features['MMRCurrentRetailAveragePrice'].dropna().values.mean()
    features.loc[features['MMRCurrentRetailAveragePrice'].isnull(),
                 'MMRCurrentRetailAveragePrice'] = mean_retail_avg_price

    # MMRAcquisitonRetailCleanPrice
    mean_retail_clean_price = features['MMRCurrentRetailCleanPrice'].dropna().values.mean()
    features.loc[features['MMRCurrentRetailCleanPrice'].isnull(),
                 'MMRCurrentRetailCleanPrice'] = mean_retail_clean_price

    # Fill missing values in PRIMEUNIT and AUCGUART
    features.loc[features['PRIMEUNIT'].isnull(), 'PRIMEUNIT'] = 'OTHER'
    features.loc[features['AUCGUART'].isnull(), 'AUCGUART'] = 'OTHER'

    # Extract Features
    # Odometer reading Per Year
    features['OdoPerYear'] = features['VehOdo'] / (features['VehicleAge'] + 1)

    features['AuctionAverageChange'] = features['MMRCurrentAuctionAveragePrice'] - features[
        'MMRAcquisitionAuctionAveragePrice']
    features['AuctionCleanChange'] = features['MMRCurrentAuctionCleanPrice'] - features[
        'MMRAcquisitionAuctionCleanPrice']
    features['RetailAverageChange'] = features['MMRCurrentRetailAveragePrice'] - features[
        'MMRAcquisitionRetailAveragePrice']
    features['RetailCleanChange'] = features['MMRCurrentRetailCleanPrice'] - features['MMRAcquisitonRetailCleanPrice']
    features['AcqRetailAuctionAverageDiff'] = features['MMRAcquisitionRetailAveragePrice'] - features[
        'MMRAcquisitionAuctionAveragePrice']
    features['AcqRetailAuctionCleanDiff'] = features['MMRAcquisitonRetailCleanPrice'] - features[
        'MMRAcquisitionAuctionCleanPrice']
    features['CurRetailAuctionAverageDiff'] = features['MMRCurrentRetailAveragePrice'] - features[
        'MMRCurrentAuctionAveragePrice']
    features['CurRetailAuctionCleanDiff'] = features['MMRCurrentRetailCleanPrice'] - features[
        'MMRCurrentAuctionCleanPrice']

    features['AcqBCostAvgDiff'] = features['MMRAcquisitionAuctionAveragePrice'] - features['VehBCost']
    features['AcqBCostCleanDiff'] = features['MMRCurrentAuctionCleanPrice'] - features['VehBCost']
    features['AcqWarrantyCostDiff'] = features['WarrantyCost'] - features['VehBCost']

    # Handle categorical features
    # Zipcode: VNZIP1
    avg_percentage = 0.12298754504473644
    zipcodes = features['VNZIP1'].value_counts().index.tolist()
    for zipcode in zipcodes:
        tmp = train_data[train_data['VNZIP1'] == zipcode]['IsBadBuy'].value_counts().to_dict()
        if len(tmp) == 2:
            percentage = tmp[1] / (tmp[0] + tmp[1])
        elif len(tmp) == 1:
            if 1 in tmp:
                percentage = 1
            else:
                percentage = 0
        else:
            percentage = avg_percentage
        features.loc[features['VNZIP1'] == zipcode, 'VNZIP1'] = percentage

    # Model
    avg_percentage = 0.12298754504473644
    models = features['Model'].value_counts().index.tolist()
    for model in models:
        tmp = train_data[train_data['Model'] == model]['IsBadBuy'].value_counts().to_dict()
        if len(tmp) == 2:
            percentage = tmp[1] / (tmp[0] + tmp[1])
        elif len(tmp) == 1:
            if 1 in tmp:
                percentage = 1
            else:
                percentage = 0
        else:
            percentage = avg_percentage
        features.loc[features['Model'] == model, 'Model'] = percentage

    # SubModel
    avg_percentage = 0.12298754504473644
    sub_models = features['SubModel'].value_counts().index.tolist()
    for sub_model in sub_models:
        tmp = train_data[train_data['SubModel'] == sub_model]['IsBadBuy'].value_counts().to_dict()
        if len(tmp) == 2:
            percentage = tmp[1] / (tmp[0] + tmp[1])
        elif len(tmp) == 1:
            if 1 in tmp:
                percentage = 1
            else:
                percentage = 0
        else:
            percentage = avg_percentage
        features.loc[features['SubModel'] == sub_model, 'SubModel'] = percentage

    features['PurchDate'] = (pd.to_datetime(features['PurchDate'], format="%m/%d/%Y")
                             - pd.datetime(2009, 12, 7)).dt.total_seconds()

    reduce_make_list = features['Make'].value_counts().index.tolist()[-15:]
    features.loc[features['Make'].isin(reduce_make_list), 'Make'] = 'OTHER'

    reduce_vnst_list = features['VNST'].value_counts().index.tolist()[-18:]
    features.loc[features['VNST'].isin(reduce_vnst_list), 'VNST'] = 'OTHER'

    # Handle Numerical Features
    numeric_features = features.select_dtypes(include=["float64", "int64"]).columns.values.tolist()
    numeric_features.remove('IsOnlineSale')

    # Normalize features into range (0, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    for feature in numeric_features:
        tmp = features[feature].values
        scaled = min_max_scaler.fit_transform(tmp)
        features[feature] = scaled

    # Convert categorical data type to category
    cat_features = features.select_dtypes(include=["object"]).columns.values
    encoder = preprocessing.LabelEncoder()
    for cat_feature in cat_features:
        features[cat_feature] = encoder.fit_transform(features[cat_feature])
        features[cat_feature] = features[cat_feature].astype('category')
    features['IsOnlineSale'] = features['IsOnlineSale'].astype('category')

    # Balance Train Data
    train_data_x = features.iloc[:72983]
    test_data_x = features.iloc[72983:]

    # Downsampling
    good_buy = train_data_x[train_data_y.isin([0])].index.tolist()
    good_buy = np.random.choice(good_buy, 15000).tolist()

    # Upsampling
    bad_buy = train_data_x[train_data_y.isin([1])].index.tolist()
    bad_buy = np.random.choice(bad_buy, 10000).tolist()

    # Randomly shuffle
    selected_data = good_buy + bad_buy
    random.shuffle(selected_data)

    train_data_x = train_data_x.iloc[selected_data]
    train_data_y = train_data_y.iloc[selected_data]

    return train_data_x, train_data_y, test_data_x, test_idx

def train():
    train_data = pd.read_csv('../data/training.csv')
    test_data = pd.read_csv('../data/test.csv')

    # Trim feature is highly correlated to Model and SubModel, and it contains lots of missing value
    train_data.drop(['Trim'], axis=1, inplace=True)
    test_data.drop(['Trim'], axis=1, inplace=True)

    # WheelTypeId is duplicate of WheelType
    train_data.drop(['WheelTypeID'], axis=1, inplace=True)
    test_data.drop(['WheelTypeID'], axis=1, inplace=True)

    # VehYear is duplicate of VehicleAge
    train_data.drop(['VehYear'], axis=1, inplace=True)
    test_data.drop(['VehYear'], axis=1, inplace=True)



    train_data_x, train_data_y, test_data_x, test_idx = Preprocessor(train_data, test_data)

    base_classifier = lgb.LGBMClassifier(num_leaves=127, max_depth=8, learning_rate=0.1, n_estimators=500,
                                         max_bin=255, subsample_for_bin=20, objective='binary', min_split_gain=0.0,
                                         min_child_weight=2, min_child_samples=10, subsample=0.7, subsample_freq=1,
                                         colsample_bytree=0.7, reg_alpha=0.01, reg_lambda=0.1)

    base_booster = base_classifier.fit(train_data_x, train_data_y)

    selected_features = []
    columns = train_data_x.columns.values
    feature_importance = base_booster.feature_importances_
    mean_importance = np.mean(feature_importance)
    for i in range(len(columns)):
        if feature_importance[i] >= mean_importance:
            selected_features.append(columns[i])

    selected_train_data_x = train_data_x[selected_features]
    selected_test_data_x = test_data_x[selected_features]


    final_classifier = lgb.LGBMClassifier(num_leaves=256, learning_rate=0.1, max_depth=-1, n_estimators=500,
                                          max_bin=255, subsample_for_bin=20,
                                          objective='binary', min_split_gain=0.0, colsample_bytree=0.6, subsample=0.9,
                                          subsample_freq=4, min_child_weight=1, min_child_samples=10, reg_alpha=0.01,
                                          reg_lambda=0.1)

    final_classifier.fit(selected_train_data_x, train_data_y)

    preds = final_classifier.predict(selected_test_data_x)
    res = pd.DataFrame({'RefId': test_idx, 'IsBadBuy': preds})
    res.to_csv('../output/submission.csv', index=False)

def main():
    train()

if __name__ == '__main__':
    main()











