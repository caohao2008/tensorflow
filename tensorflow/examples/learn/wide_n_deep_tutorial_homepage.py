# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from six.moves import urllib

import pandas as pd
import tensorflow as tf
import re

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")


COLUMNS = ["label","ADISTANCE","APPOINTMENT","DEAL_COLLECTED","TYPE_COLLECTED","COUPON0"
,"COUPON1","COUPON2","COUPON3","TYPE_VIEWED_PASSED_TIME","COUPON5"
,"DAYOFWEEK1","DAYOFWEEK2","DAYOFWEEK3","DAYOFWEEK4","DAYOFWEEK5"
,"DAYOFWEEK6","DAYOFWEEK7","DEAL_ORDERED,POI_ORDERED","DEAL_VIEWED,POI_VIEWED","DISCOUNT,POI_DISCOUNT"
,"DISTANCE","BUZ_DELETED","TYPE_DELTED","CDISTANCE","USER_LEVEL"
,"USER_FEEDBACK_SCORE","IN_ORDER_SIM","IN_VIEW_SIM","RTCLKNUM,POI_DEAL_RTCNUM","HOUROFDAY0"
,"HOUROFDAY1","HOUROFDAY10","HOUROFDAY11","HOUROFDAY12","HOUROFDAY13"
,"HOUROFDAY14","HOUROFDAY15","HOUROFDAY16","HOUROFDAY17","HOUROFDAY18"
,"HOUROFDAY19","HOUROFDAY2","HOUROFDAY20","HOUROFDAY21","HOUROFDAY22"
,"HOUROFDAY23","HOUROFDAY3","HOUROFDAY4","HOUROFDAY5","HOUROFDAY6"
,"HOUROFDAY7","HOUROFDAY8","HOUROFDAY9","NEWDEAL","NUMRESULTS"
,"ORDERCOUNT,POI_HISTORYCOUPONCOUNT","PRICE,POI_LOWESTPIRCE","BUZ_COLLECTED","DEAL_VIEWED_PASSED_TIME,POI_VIEWED_PASSED_TIME","RATECOUNT"
,"RATEVAL","STGNUM","BUZ_VIEWED_PASSED_TIME","TYPE_ORDERED","ISREMOTE"
,"TYPE_VIEWED","ISPROMOTION","CLASS2,POI_CLASS2","CLASS3,POI_CLASS3","CLASS4,POI_CLASS4"
,"CLASS5,POI_CLASS5","MEAL_COUNT","AVGPRICEPERPERSON","CLASS206,POI_CLASS206","CLASS207,POI_CLASS207"
,"CLASS208,POI_CLASS208","CLASS209,POI_CLASS209","CLASS217,POI_CLASS217","CLASS226,POI_CLASS226","BUZ_ORDERED"
,"BUZ_VIEWED","CTR,POI_CTR","CVR,POI_CVR","CXR,POI_CXR","DEAL_POI_NUM"
,"2hourSale","24hourSale","staticSaleScore","YOUFANGTAI","TODAYNOTAVAI"
,"HAVENOTAVAI","CLIENT_ANDROID","CLIENT_IPHONE","CLIENT_WAP","CLIENT_IPAD"
,"CLIENT_ANDROIDHD","CLASS_FAVOR","CATE_FAVOR","USERID0","ISLOWPRICE"
,"ISNEWUSERPROMOTION","ISCAMPAINBRAND","DEAL_ORDERED_PASSED_TIME,POI_ORDERED_PASSED_TIME","TYPE_ORDERED_PASSED_TIME","BUZ_ORDERED_PASSED_TIME"
,"CTR_TEST","CXR_TEST","CVR_TEST","COMMENT_NUM","RELAY_RATIO"
,"PICTURE_RATIO","GCOMMENT_RATIO","BRAND_SCORE","BRAND_GRADE","PAYMENT_INCR"
,"AREA_AMOUNT","DEAL_LEFT_DELETE","WEATHER_QING","WEATHER_YIN","WEATHER_YUN"
,"WEATHER_FENG","WEATHER_XUE","WEATHER_WU","WEATHER_YU","WEATHER_TMP"
,"WEATHER_TMP_MAX","WEATHER_TMP_MIN","WEATHER_PM2P5","NEG_TAG_NUM","POS_TAG_NUM"
,"ALL_TAG_NUM","NEG_TAG_RATIO","oftenBuy","RT_Book","viewedPoi_Deals"
,"viewedPb","viewed","rt-search","llr_usercf_new","collected"
,"userbased_orderpoi","llr_usercf_timectx","bookmark","lprice","rtub_v"
,"rt_area_mix","rtub_c","rtub_o","rt_area_view","item_ocf"
,"querybased","item_vcf","rt_geo","rtal_home","llr_usercf_poi"
,"rtal_work","rt_geo_rtFavor","bak","geo_rtFavor","bak_rtFavor"
,"geo","hot_rtFavor","hot","rt-loc-new","homepage-hot"
,"CLUSTER_12","CLUSTER_5","CLUSTER_4","CLUSTER_18","CLUSTER_14"
,"CLUSTER_OTHER","PROMOTION_TYPE_A","PROMOTION_TYPE_B","PROMOTION_TYPE_C","PROMOTION_TYPE_D"
,"PROMOTION_VALUE","RECOMMEND_CONSUME_CNT","rt_area_mix_RATIO","viewed_RATIO","rt_area_view_RATIO"
,"llr_usercf_timectx_RATIO","viewedPoi_Deals_RATIO","llr_usercf_new_RATIO","viewedPb_RATIO","userbased_orderpoi_RATIO"
,"lprice_RATIO","rtal_home_RATIO","item_vcf_RATIO","RT_Book_RATIO","rt-search_RATIO"
,"rtal_work_RATIO","hot_RATIO","collected_RATIO","item_ocf_RATIO","oftenBuy_RATIO"
,"ISMULTICITY","HPRECSALECNT","USER_CITYPREF","USER_RTGEO_PREF","USER_DEALGEO_PREF"
,"USER_HOME_DIS","USER_WORK_DIS","USER_CONSUME_DIS","SLCT_RETURNED","SLCT_RETURN_TIME"
,"rt_area_mix_score","rt_area_view_score","rtal_home_score","rtal_work_score","llr_usercf_timectx_score"
,"llr_usercf_new_score","rt-search_score","userbased_orderpoi_score","item_ocf_score","querybased_score"
,"item_vcf_score","DEAL_USER_REC_HIS","DEAL_USER_CLI_HIS","GEOHASH_DEAL_TRANSFER_RATIO","CSSCORE_TOP3"
,"CSSCORE_TOP10","CSSCORE_TOP30","rt_area_mix_score_ratio3","rt_area_mix_score_ratio10","rt_area_mix_score_ratio30"
,"IS_WIFI_POI_DEAL","IS_CONSUME_POI_DEAL","USER_RESI_CITY","USER_RESI_PREF","chooseDiffCity-chooseNotResiCity"
,"chooseDiffCity-chooseResiCity","chooseDiffCity-unkown","chooseSameCity-chooseNotResiCity","chooseSameCity-chooseResiCity","chooseSameCity-unkown"
,"男","女","AGE_20以下","AGE_20~25","AGE_25~30"
,"AGE_30~35","AGE_35~40","AGE_40以上","金牛座","天蝎座"
,"处女座","牡羊座","水瓶座","射手座","狮子座"
,"双子座","天秤座","巨蟹座","双鱼座","摩羯座"
,"MARRIED","HAS_CHILD","JOB_白领","JOB_学生","JOB_其他"
,"HAS_CAR","NEG_FEEDBACK","ONLINE_DAYS","TOTAL_SHOW","CXR_CITY"
,"CXR_DIS","CXR_TIME","CXR_TIME_20_9","CXR_TIME_10_11","CXR_TIME_12_13"
,"CXR_TIME_14_17","CXR_TIME_18_19","CVR_CITY","CVR_DIS","CVR_TIME"
,"CVR_TIME_20_9","CVR_TIME_10_11","CVR_TIME_12_13","CVR_TIME_14_17","CVR_TIME_18_19"
,"CTR_CITY","CTR_DIS","CTR_TIME","CTR_TIME_20_9","CTR_TIME_10_11"
,"CTR_TIME_12_13","CTR_TIME_14_17","CTR_TIME_18_19","REC_NUM_CITY","REC_NUM_DIS"
,"REC_NUM_TIME","REC_NUM_TIME_20_9","REC_NUM_TIME_10_11","REC_NUM_TIME_12_13","REC_NUM_TIME_14_17"
,"REC_NUM_TIME_18_19","CLICK_NUM_CITY","CLICK_NUM_DIS","CLICK_NUM_TIME","CLICK_NUM_TIME_20_9"
,"CLICK_NUM_TIME_10_11","CLICK_NUM_TIME_12_13","CLICK_NUM_TIME_14_17","CLICK_NUM_TIME_18_19","ORDER_NUM_CITY"
,"ORDER_NUM_DIS","ORDER_NUM_TIME","ORDER_NUM_TIME_20_9","ORDER_NUM_TIME_10_11","ORDER_NUM_TIME_12_13"
,"ORDER_NUM_TIME_14_17","ORDER_NUM_TIME_18_19","CXR_CLASS_TIME","CVR_CLASS_TIME","CTR_CLASS_TIME"
,"CXR_CLASS_DIS","CVR_CLASS_DIS","CTR_CLASS_DIS","CXR_CLASS_CITY","CVR_CLASS_CITY"
,"CTR_CLASS_CITY","CLASS5005,POI_CLASS5005","TYPE756","CLASS_ORDERED_PASSED_TIME","CATE_ORDERED_PASSED_TIME"
,"BRAND_ORDERED_PASSED_TIME","CLASS_VIEWED_PASSED_TIME","CATE_VIEWED_PASSED_TIME","BRAND_VIEWED_PASSED_TIME","BRAND_COLLECTED"
,"BRAND_ORDERED_NEW","BRAND_VIEWED_NEW","DEAL_ORDERED_NEW","DEAL_VIEWED_NEW","TYPE_ORDERED_NEW"
,"TYPE_VIEWED_NEW","BUZ_ORDERED_NEW","BUZ_VIEWED_NEW","WAIMAI_VIEWED","WAIMAI_ORDERED"
,"CLASS_ORDERED_PASSED_TIME_NEW","CATE_ORDERED_PASSED_TIME_NEW","TYPE_ORDERED_PASSED_TIME_NEW","BUZ_ORDERED_PASSED_TIME_NEW","BRAND_ORDERED_PASSED_TIME_NEW"
,"CLASS_VIEWED_PASSED_TIME_NEW","CATE_VIEWED_PASSED_TIME_NEW","TYPE_VIEWED_PASSED_TIME_NEW","BRAND_VIEWED_PASSED_TIME_NEW","BUZ_VIEWED_PASSED_TIME_NEW"
,"WAIMAI_ORDERED_PASSED_TIME","WAIMAI_VIEWED_PASSED_TIME","POI_MARKNUMBERS","POI_LATESTWEEKCOUPON","POI_ISHOT"
,"POI_SCORERATIO","USER_ALL_VIEW_NUM","USER_ALL_ORDER_NUM","POI_GROUP","DEAL_GROUP"
,"POI_HOTEL","DEAL_AD","MPOI_MOVIE","POI_WAIMAI","POI_MAITON"
,"POI_LVYOU","SDEAL_SHOW","ODEAL_OVERSEAS","BIZ_AREA","IS_NEW_USER"
,"MonthSaleNum","DAOZHONG_CATE_FAVOR","SALES_COUNT_BY_GEO_DTYPE","2HOUR_RT_SALES_COUNT","OPEN_NOW"
,"MIN_PRICE","SHIPPING_FEE","SHIPPING_MEITUAN","SHIPPING_ZHONGBAO","SHIPPING_ZIPEI"
,"AVG_DELIVERY_TIME","ORDER_NUM_TODAY","AVG_AMOUNT","AVG_AMOUNT_MONTH","AVG_FACT_AMOUNT"
,"AVG_FACT_AMOUNT_MONTH","POI_SCORE","FOOD_AVG_SCORE","DELIVERY_AVG_SCORE","POI_OPEN_DAYS"
,"WAIMAI_CTR","WAIMAI_CVR","WAIMAI_CXR"]
LABEL_COLUMN = "label"

CATEGORICAL_COLUMNS = []

CONTINUOUS_COLUMNS = ["DISTANCE", "2hourSale", "24hourSale", "CTR_TEST",
                      "CXR_TEST","CVR_TEST","viewedPoi_Deals","viewedPb","viewed","rt_area_mix","rt_area_view","USER_ALL_VIEW_NUM","CLASS_VIEWED_PASSED_TIME_NEW","ADISTANCE","USER_ALL_ORDER_NUM","NUMRESULTS","SALES_COUNT_BY_GEO_DTYPE","ONLINE_DAYS","GEOHASH_DEAL_TRANSFER_RATIO",
                      "USER_WORK_DIS","USER_CITYPREF","AVGPRICEPERPERSON","DEAL_USER_REC_HIS","USER_HOME_DIS","USER_RTGEO_PREF","AREA_AMOUNT","CTR_CLASS_TIME","COMMENT_NUM","USER_LEVEL","BRAND_VIEWED_PASSED_TIME_NEW","WAIMAI_ORDERED_PASSED_TIME","CTR_CLASS_DIS","BUZ_ORDERED_PASSED_TIME_NEW","BUZ_VIEWED_PASSED_TIME",
    "CTR_CITY","USER_FEEDBACK_SCORE","CATE_VIEWED_PASSED_TIME_NEW","CSSCORE_TOP30",
    "DEAL_USER_CLI_HIS","CLICK_NUM_CITY","BRAND_ORDERED_PASSED_TIME_NEW","BUZ_ORDERED_PASSED_TIME","CSSCORE_TOP3","PICTURE_RATIO","IS_WIFI_POI_DEAL","TYPE_VIEWED_PASSED_TIME_NEW","WEATHER_TMP_MAX",
    "CTR_DIS","CLASS_VIEWED_PASSED_TIME","REC_NUM_CITY","rt_area_view_score","WEATHER_TMP_MIN","NEG_TAG_RATIO"
    ,"DEAL_POI_NUM","SLCT_RETURN_TIME","RATECOUNT","CLASS_ORDERED_PASSED_TIME","WEATHER_TMP",
    "RATEVAL","PAYMENT_INCR","item_vcf_score","CSSCORE_TOP10","REC_NUM_DIS","CXR_CITY","CXR_CLASS_TIME","CATE_ORDERED_PASSED_TIME_NEW"                
    ,"CVR_CITY","CVR_CLASS_TIME","CLIENT_ANDROID","TYPE_VIEWED_PASSED_TIME",
    ,"HPRECSALECNT","querybased_score","POI_MARKNUMBERS","CATE_VIEWED_PASSED_TIME","CLIENT_IPHONE","CTR_TIME","CTR_TIME_10_11","CATE_ORDERED_PASSED_TIME","RELAY_RATIO"
    ,"CXR_CLASS_DIS","TYPE_ORDERED_PASSED_TIME_NEW","CLICK_NUM_DIS","item_ocf_score","CTR_TIME_14_17","CTR_TIME_12_13","ORDER_NUM_CITY","TYPE_ORDERED_PASSED_TIME","CTR_TIME_18_19","MEAL_COUNT","CLICK_NUM_TIME",
    "GCOMMENT_RATIO","staticSaleScore","userbased_orderpoi_score","2HOUR_RT_SALES_COUNT"
                     ]

def maybe_download():
  """Maybe downloads training data and returns train and test file names."""
  if FLAGS.train_data:
    train_file_name = FLAGS.train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)  # pylint: disable=line-too-long
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if FLAGS.test_data:
    test_file_name = FLAGS.test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)  # pylint: disable=line-too-long
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s" % test_file_name)
  
  return train_file_name, test_file_name

features=[]

def build_estimator(model_dir):
  """Build an estimator."""
  # Sparse base columns.

  # Continuous base columns.
  ##age = tf.contrib.layers.real_valued_column("age")
  for feature in CONTINUOUS_COLUMNS:
      features.append(tf.contrib.layers.real_valued_column(feature))

  # Transformations.
  # Wide columns and deep columns.
  wide_columns = features
  deep_columns = features
  #print("wide_columns =",wide_columns)
  #print("deep_columns =",deep_columns)

  if FLAGS.model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif FLAGS.model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
  return m


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  ##print("arrive here");
  ##print("continuous_cols =",continuous_cols)
  # Creates a dictionary mapping from each categorcal feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  ##print("categorical_cols =",categorical_cols)
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  
  ##print("feature_cols =",feature_cols)
  # Returns the feature columns and the label.
  return feature_cols, label



def train_and_eval():
  """Train and evaluate the model."""
  train_file_name, test_file_name = maybe_download()
  ##print("train_file_name =",train_file_name)
  ##print("test_file_name =",test_file_name)
  df_train = pd.read_csv(
      tf.gfile.Open(train_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      engine="python")
  ##print("df_train = ",df_train)
  df_test = pd.read_csv(
      tf.gfile.Open(test_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1,
      engine="python")
  ##print("df_test = ",df_test)

  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)

  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  ##print("model directory = %s" % model_dir)

  m = build_estimator(model_dir)
  m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()
