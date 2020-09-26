import pandas as pd
import numpy as np

class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        features = self.dataset.copy()
        features.drop(
            ['Id','Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
             'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
             'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
             'MiscVal'],
            axis=1, inplace=True)
        features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])
        features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())
        features['Alley'] = features['Alley'].fillna('NOACCESS')
        features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            features[col] = features[col].fillna('NoBSMT')
        features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)
        features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])

        features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])
        features['FireplaceQu'] = features['FireplaceQu'].fillna('NoFP')
        for col in ('GarageType', 'GarageFinish', 'GarageQual'):
            features[col] = features[col].fillna('NoGRG')
        features['GarageCars'] = features['GarageCars'].fillna(0)
        features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
        features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
        features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
        numeric_features = features.loc[:, ['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]
        numeric_features_standardized = (numeric_features - numeric_features.mean()) / numeric_features.std()
        features['MSSubClass'] = features['MSSubClass'].astype(str)
        features['YrSold'] = features['YrSold'].astype(str)
        features['MoSold'] = features['MoSold'].astype(str)
        features['KitchenAbvGr'] = features['KitchenAbvGr'].astype(str)
        features.OverallCond = features.OverallCond.astype(str)
        for col in features.dtypes[features.dtypes == 'object'].index:
            for_dummy = features.pop(col)
            features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)
        features.update(numeric_features_standardized)
        self.dataset=features.loc[:].select_dtypes(include=[np.number])
        return self.dataset



