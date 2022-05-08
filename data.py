import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from itertools import product


class Data:

  def __init__(self, data_path):
    self.data_path = data_path

  def get_data(self):
      self.sales_train = pd.read_csv(self.data_path + 'sales_train.csv')
      self.shops = pd.read_csv(self.data_path + 'shops.csv')
      self.items = pd.read_csv(self.data_path + 'items.csv')
      self.item_categories = pd.read_csv(self.data_path + 'item_categories.csv')
      self.test = pd.read_csv(self.data_path + 'test.csv')

      self.sales_train = self.sales_train.rename(columns={'date': 'date',
                                                'date_block_num': 'month_id',
                                                'shop_id': 'shop_id',
                                                'item_id': 'item_id',
                                                'item_price': 'item_price',
                                                'item_cnt_day': 'item_cnt'})

  def downcast(self, dataFrame, verbose=True):
      start_mem = dataFrame.memory_usage().sum() / 1024 ** 2
      for col in dataFrame.columns:
          dtype_name = dataFrame[col].dtype.name
          if dtype_name == 'object':
              pass
          elif dtype_name == 'bool':
              dataFrame[col] = dataFrame[col].astype('int8')
          elif dtype_name.startswith('int') or (dataFrame[col].round() == dataFrame[col]).all():
              dataFrame[col] = pd.to_numeric(dataFrame[col], downcast='integer')
          else:
              dataFrame[col] = pd.to_numeric(dataFrame[col], downcast='float')
      end_mem = dataFrame.memory_usage().sum() / 1024 ** 2
      if verbose:
          print('{:.1f}% compressed'.format(100 * (start_mem - end_mem) / start_mem))
      return dataFrame

  def add_mean_features(self, dataFrame, mean_features, idx_features):
      assert (idx_features[0] == 'month_id') and \
             len(idx_features) in [2, 3]

      if len(idx_features) == 2:
          feature_name = idx_features[1] + 'by_avg_sold_count'
      else:
          feature_name = idx_features[1] + '_' + idx_features[2] + 'by_avg_sold_count'

      group = dataFrame.groupby(idx_features).agg({'item_cnt_month': 'mean'})
      group = group.reset_index()
      group = group.rename(columns={'item_cnt_month': feature_name})

      dataFrame = dataFrame.merge(group, on=idx_features, how='left')
      mean_features.append(feature_name)
      return dataFrame, mean_features

  def add_lag_features(self, dataFrame, lag_features_to_clip, idx_features,
                       lag_feature, nlags=3, clip=False):

      df_temp = dataFrame[idx_features + [lag_feature]].copy()
      for i in range(1, nlags + 1):
          lag_feature_name = lag_feature + '_time_gap' + str(i)
          df_temp.columns = idx_features + [lag_feature_name]
          df_temp['month_id'] += i
          dataFrame = dataFrame.merge(df_temp.drop_duplicates(),
                        on=idx_features,
                        how='left')

          dataFrame[lag_feature_name] = dataFrame[lag_feature_name].fillna(0)
          if clip:
              lag_features_to_clip.append(lag_feature_name)
      return dataFrame, lag_features_to_clip

  def preprocess_setp1(self):
      self.sales_train = self.sales_train[self.sales_train['item_price'] > 0]
      self.sales_train = self.sales_train[self.sales_train['item_price'] < 50000]
      self.sales_train = self.sales_train[self.sales_train['item_cnt'] > 0]
      self.sales_train = self.sales_train[self.sales_train['item_cnt'] < 1000]

      self.sales_train.loc[self.sales_train['shop_id'] == 0, 'shop_id'] = 57
      self.sales_train.loc[self.sales_train['shop_id'] == 1, 'shop_id'] = 58
      self.sales_train.loc[self.sales_train['shop_id'] == 10, 'shop_id'] = 11
      self.sales_train.loc[self.sales_train['shop_id'] == 39, 'shop_id'] = 40

      self.test.loc[self.test['shop_id'] == 0, 'shop_id'] = 57
      self.test.loc[self.test['shop_id'] == 1, 'shop_id'] = 58
      self.test.loc[self.test['shop_id'] == 10, 'shop_id'] = 11
      self.test.loc[self.test['shop_id'] == 39, 'shop_id'] = 40

      self.shops['city'] = self.shops['shop_name'].apply(lambda x: x.split()[0])
      self.shops.loc[self.shops['city'] == '!Якутск', 'city'] = 'Якутск'

      label_encoder = LabelEncoder()
      self.shops['city'] = label_encoder.fit_transform(self.shops['city'])

      self.shops = self.shops.drop('shop_name', axis=1)
      self.items = self.items.drop(['item_name'], axis=1)

      self.items['first_sell_month'] = self.sales_train.groupby('item_id').agg({'month_id': 'min'})['month_id']
      self.items['first_sell_month'] = self.items['first_sell_month'].fillna(34)
      self.item_categories['major_category'] = self.item_categories['item_category_name'].apply(lambda x: x.split()[0])
      self.item_categories['major_category'] = self.item_categories['major_category'].map(lambda x : x if len(x) >= 5 else 'etc')

      label_encoder = LabelEncoder()
      self.item_categories['major_category'] = label_encoder.fit_transform(self.item_categories['major_category'])
      self.item_categories = self.item_categories.drop('item_category_name', axis=1)

  def preprocess_setp2(self):
      train = []

      for i in self.sales_train['month_id'].unique():
          all_shop = self.sales_train.loc[self.sales_train['month_id'] == i, 'shop_id'].unique()
          all_item = self.sales_train.loc[self.sales_train['month_id'] == i, 'item_id'].unique()
          train.append(np.array(list(product([i], all_shop, all_item))))

      idx_features = ['month_id', 'shop_id', 'item_id']  # 기준 피처
      train = pd.DataFrame(np.vstack(train), columns=idx_features)
      group = self.sales_train.groupby(idx_features).agg({'item_cnt': 'sum',
                                                     'item_price': 'mean'})
      group = group.reset_index()
      group = group.rename(columns={'item_cnt': 'item_cnt_month', 'item_price': 'avg_item_price'})

      train = train.merge(group, on=idx_features, how='left')
      group = self.sales_train.groupby(idx_features).agg({'item_cnt': 'count'})
      group = group.reset_index()

      train = train.merge(group, on=idx_features, how='left')
      self.test['month_id'] = 34
      all_data = pd.concat([train, self.test.drop('ID', axis=1)],
                           ignore_index=True,
                           keys=idx_features)
      self.all_data = all_data.fillna(0)

  def preprocess_setp3(self):
      self.all_data = self.all_data.merge(self.shops, on='shop_id', how='left')
      self.all_data = self.all_data.merge(self.items, on='item_id', how='left')
      self.all_data = self.all_data.merge(self.item_categories, on='item_category_id', how='left')

      item_mean_features = []

      self.all_data, item_mean_features = self.add_mean_features(self.all_data,
                                                       mean_features=item_mean_features,
                                                       idx_features=['month_id', 'item_id'])

      self.all_data, item_mean_features = self.add_mean_features(self.all_data,
                                                       mean_features=item_mean_features,
                                                       idx_features=['month_id', 'item_id', 'city'])

      shop_mean_features = []
      self.all_data, shop_mean_features = self.add_mean_features(self.all_data,
                                                       mean_features=shop_mean_features,
                                                       idx_features=['month_id', 'shop_id', 'item_category_id'])

      lag_features_to_clip = []
      idx_features = ['month_id', 'shop_id', 'item_id']

      # idx_features를 기준으로 월간 판매량의 세 달치 시차 피처 생성
      self.all_data, lag_features_to_clip = self.add_lag_features(self.all_data,
                                                        lag_features_to_clip=lag_features_to_clip,
                                                        idx_features=idx_features,
                                                        lag_feature='item_cnt_month',
                                                        nlags=3,
                                                        clip=True)

      self.all_data, lag_features_to_clip = self.add_lag_features(self.all_data,
                                                        lag_features_to_clip=lag_features_to_clip,
                                                        idx_features=idx_features,
                                                        lag_feature='item_cnt',
                                                        nlags=3)


      self.all_data, lag_features_to_clip = self.add_lag_features(self.all_data,
                                                        lag_features_to_clip=lag_features_to_clip,
                                                        idx_features=idx_features,
                                                        lag_feature='avg_item_price',
                                                        nlags=3)

      for item_mean_feature in item_mean_features:
          self.all_data, lag_features_to_clip = self.add_lag_features(self.all_data,
                                                            lag_features_to_clip=lag_features_to_clip,
                                                            idx_features=idx_features,
                                                            lag_feature=item_mean_feature,
                                                            nlags=3,
                                                            clip=True)
      self.all_data = self.all_data.drop(item_mean_features, axis=1)

      for shop_mean_feature in shop_mean_features:
          self.all_data, self.lag_features_to_clip = self.add_lag_features(self.all_data,
                                                            lag_features_to_clip=lag_features_to_clip,
                                                            idx_features=['month_id', 'shop_id', 'item_category_id'],
                                                            lag_feature=shop_mean_feature,
                                                            nlags=3,
                                                            clip=True)

      self.all_data = self.all_data.drop(shop_mean_features, axis=1)
      self.all_data = self.all_data.drop(self.all_data[self.all_data['month_id'] < 3].index)

  def preprocess_setp4(self):
      self.all_data['item_idby_avg_sold_count_time_gap_avg'] = self.all_data[['item_idby_avg_sold_count_time_gap1',
                                                                    'item_idby_avg_sold_count_time_gap2',
                                                                    'item_idby_avg_sold_count_time_gap3']].mean(axis=1)

      self.all_data[self.lag_features_to_clip + ['item_cnt_month',
                                                 'item_idby_avg_sold_count_time_gap_avg']] =\
          self.all_data[self.lag_features_to_clip + ['item_cnt_month', 'item_idby_avg_sold_count_time_gap_avg']].clip(0, 20)

      self.all_data['time_gap_difference1'] = self.all_data['item_cnt_month_time_gap1'] / self.all_data['item_cnt_month_time_gap2']
      self.all_data['time_gap_difference1'] = self.all_data['time_gap_difference1'].replace([np.inf, -np.inf],
                                                                                  np.nan).fillna(0)

      self.all_data['time_gap_difference2'] = self.all_data['item_cnt_month_time_gap2'] / self.all_data['item_cnt_month_time_gap3']
      self.all_data['time_gap_difference2'] = self.all_data['time_gap_difference2'].replace([np.inf, -np.inf],
                                                                                  np.nan).fillna(0)

      self.all_data['is_new_item'] = self.all_data['first_sell_month'] == self.all_data['month_id']
      self.all_data['passed_month'] = self.all_data['month_id'] - self.all_data['first_sell_month']
      self.all_data['month'] = self.all_data['month_id'] % 12
      self.all_data = self.all_data.drop(['first_sell_month', 'avg_item_price', 'item_cnt'], axis=1)

  def make_ml_data(self):
      X_train = self.all_data[self.all_data['month_id'] < 33]
      X_train = X_train.drop(['item_cnt_month'], axis=1)
      # 검증 데이터 (피처)
      X_valid = self.all_data[self.all_data['month_id'] == 33]
      X_valid = X_valid.drop(['item_cnt_month'], axis=1)
      # 테스트 데이터 (피처)
      X_test = self.all_data[self.all_data['month_id'] == 34]
      X_test = X_test.drop(['item_cnt_month'], axis=1)

      # 훈련 데이터 (타깃값)
      y_train = self.all_data[self.all_data['month_id'] < 33]['item_cnt_month']
      # 검증 데이터 (타깃값)
      y_valid = self.all_data[self.all_data['month_id'] == 33]['item_cnt_month']

      return X_train, X_valid, y_train, y_valid
