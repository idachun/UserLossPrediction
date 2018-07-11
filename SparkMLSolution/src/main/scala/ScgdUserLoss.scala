package com.starcor.userportrait.offline.plugins.user.area

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory


object ScgdUserLoss {

  private val log = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]): Unit = {

    val startTime1 = System.currentTimeMillis()

    val spark = SparkSession
      .builder()
      .master("local")
      .appName("scgdUserLoss")
      .config("spark.sql.hive.convertMetastoreParquet", "false")
      .config("spark.sql.parquet.filterPushdown", "false")
      .getOrCreate()

    //    训练数据-4月份,特征生成
    val data = trainFeatureGenerate(spark)

    //    训练数据-4月份,特征处理
    var dataAll = trainFeatureProcess(spark, data)

    //模型建立与测试
    setModel(dataAll)

    //    待预测数据-5月份,特征生成
    var data1=testFeatureGenerate(spark)

    //    待预测数据-5月份,特征处理
    data1 = testFeatureProcess(spark, data1)

    //    模型预测
    val pred = makePredictions(data1)

    //    效果评估-时间窗口一个月，以6月底的流失用户为标准，测试以6月12为准
    evaluatePredictions(spark, pred)

    val endTime1 = System.currentTimeMillis()
    log.info("Total Time:" + ((endTime1 - startTime1) / 60000) + " minutes")
  }

  def trainFeatureGenerate(spark: SparkSession): DataFrame = {

    //    创建视图
    var playlog = spark.sql("select * from scgd.play_log where day between 20180329 and 20180430")
    playlog.createOrReplaceTempView("playlog")

    var userlog = spark.sql("select * from scgd.user_log where day between 20180329 and 20180430")
    userlog.createOrReplaceTempView("userlog")

    val nnsBuyOrder = spark.sql("select * from scgd.nns_buy_order where day=20180501")
    nnsBuyOrder.createOrReplaceTempView("nnsBuyOrder")
    spark.sql("select *, from_unixtime(nns_modify_time/1000,'yyyy-MM-dd') time from nnsBuyOrder where from_unixtime(nns_modify_time/1000,'yyyy-MM-dd') between '2018-03-24' and '2018-04-30'").createOrReplaceTempView("nnsBuyOrder")

    val nnsUser = spark.sql("select * from scgd.nns_user where day=20180501")
    nnsUser.createOrReplaceTempView("nnsUser")

    //    log.info("训练数据：开始用户流失相关特征生成")

    //用户观看特征
    spark.sql("select user_id,video_type,server_time,row_number() over(partition by user_id order by server_time) rank from playlog ").createOrReplaceTempView("a")
    spark.sql("select user_id,video_type,server_time,rank-1 as rank from a order by user_id,server_time").createOrReplaceTempView("b")
    spark.sql("select a.user_id,a.video_type,( b.server_time-a.server_time) watch_time from a left join b on a.user_id=b.user_id and a.rank=b.rank order by a.server_time").createOrReplaceTempView("c")
    spark.sql("select user_id,video_type, watch_time/60000.0 watch_time from c where watch_time is not null").createOrReplaceTempView("d")

    var userWatch = spark.sql("select user_id,sum(watch_time) as user_time from d group by user_id")
    var userDevice = spark.sql("select user_id,count(device_id) as device_freq from playlog group by user_id")
    var userDeviceUni = spark.sql("select user_id,count(distinct(device_id)) as device_uni from playlog group by user_id")
    var userVod = spark.sql("select user_id,sum(watch_time) as vod_time from d where video_type='vod' group by user_id")
    var userLive = spark.sql("select user_id,sum(watch_time) as live_time from d where video_type='live' group by user_id")

    var dfTrain = userWatch.join(userDevice, "user_id")
    dfTrain = dfTrain.join(userDeviceUni, "user_id")

    dfTrain.createOrReplaceTempView("dfa")
    userVod.createOrReplaceTempView("dfb")
    dfTrain = spark.sql("select dfa.user_id,user_time,device_freq,device_uni,vod_time from dfa left join dfb on dfa.user_id=dfb.user_id")

    dfTrain.createOrReplaceTempView("dfc")
    userLive.createOrReplaceTempView("dfd")
    dfTrain = spark.sql("select dfc.user_id,user_time,device_freq,device_uni,vod_time,live_time from dfc left join dfd on dfc.user_id=dfd.user_id")

    //    用户搜索次数特征
    var userSearch = spark.sql("select user_id, count(1) userSearch  from userlog where event_name='search' group by user_id ")

    dfTrain.createOrReplaceTempView("dfh")
    userSearch.createOrReplaceTempView("user_search")
    dfTrain = spark.sql("select dfh.user_id,user_time,device_freq,device_uni,vod_time,live_time,userSearch from dfh left join user_search on dfh.user_id=user_search.user_id")

    //    产品类特征
    //    20180501的数据中，注释的前3个统计量均为0，不作为特征；共有2703452条记录，支付状态为1的记录个数有1744104条；共有999815个用户，已支付用户数有680693，退订用户数396407；
    //    var userProductBuy = spark.sql("select nns_user_id user_id, count(1) userProductBuy from nnsBuyOrder where nns_order_type='product_buy' and nns_order_state=1 group by nns_user_id")
    //    var userPay = spark.sql("select nns_user_id user_id, count(1) userPay from nnsBuyOrder where nns_order_type='user_pay' and nns_order_state=1 group by nns_user_id")
    //    var userProductUpgrade = spark.sql("select nns_user_id user_id, count(1) userProductUpgrade from nnsBuyOrder where nns_order_type='product_upgrade' and nns_order_state=1 group by nns_user_id")

    //    每个用户id支付的次数
    var userOther = spark.sql("select nns_user_id user_id, count(1) userOther from nnsBuyOrder where nns_order_type='other' and nns_order_state=1 group by nns_user_id")
    //每个用户id支付总价格
    var userOffer = spark.sql("select nns_user_id user_id,sum(nns_order_price) userOffer from nnsBuyOrder where nns_order_state=1 group by nns_user_id")
    //    每个用户id退订的次数
    var userCancel = spark.sql("select nns_user_id user_id,count(1) userCancel from nnsBuyOrder where nns_order_state=5 group by nns_user_id")
    //    每个用户id退订总价格

    dfTrain.createOrReplaceTempView("dfe")
    userOther.createOrReplaceTempView("user_other")
    dfTrain = spark.sql("select dfe.user_id,user_time,device_freq,device_uni,vod_time,live_time,userSearch,userOther from dfe left join user_other on dfe.user_id=user_other.user_id")

    dfTrain.createOrReplaceTempView("dff")
    userOffer.createOrReplaceTempView("user_offer")
    dfTrain = spark.sql("select dff.user_id,user_time,device_freq,device_uni,vod_time,live_time,userSearch,userOther,userOffer from dff left join user_offer on dff.user_id=user_offer.user_id")

    dfTrain.createOrReplaceTempView("dfg")
    userCancel.createOrReplaceTempView("user_cancel")
    dfTrain = spark.sql("select dfg.user_id,user_time,device_freq,device_uni,vod_time,live_time,userSearch,userOther,userOffer,userCancel from dfg left join user_cancel on dfg.user_id=user_cancel.user_id")

    //    用户label，0正常，4停机，5销户
    var userLabel = spark.sql("select nns_id user_id,nns_state from nnsUser where nns_state=0 or nns_state=4 or nns_state=5")
    dfTrain.createOrReplaceTempView("dfi")
    userLabel.createOrReplaceTempView("user_label")
    dfTrain = spark.sql("select dfi.user_id,user_time,device_freq,device_uni,vod_time,live_time,userSearch,userOther,userOffer,userCancel,nns_state from dfi join user_label on dfi.user_id=user_label.user_id")

    dfTrain
  }

  def testFeatureGenerate(spark: SparkSession): DataFrame = {

    //    创建视图
    var playlog = spark.sql("select * from scgd.play_log where day between 20180501 and 20180531")
    playlog.createOrReplaceTempView("playlog")

    var userlog = spark.sql("select * from scgd.user_log where day between 20180501 and 20180531")
    userlog.createOrReplaceTempView("userlog")

    val nnsBuyOrder = spark.sql("select * from scgd.nns_buy_order where day=20180601")
    nnsBuyOrder.createOrReplaceTempView("nnsBuyOrder")
    spark.sql("select *, from_unixtime(nns_modify_time/1000,'yyyy-MM-dd') time from nnsBuyOrder where from_unixtime(nns_modify_time/1000,'yyyy-MM-dd') between '2018-03-24' and '2018-04-30'").createOrReplaceTempView("nnsBuyOrder")

    val nnsUser = spark.sql("select * from scgd.nns_user where day=20180601")
    nnsUser.createOrReplaceTempView("nnsUser")

    //    log.info("训练数据：开始用户流失相关特征生成")

    //用户观看特征
    spark.sql("select user_id,video_type,server_time,row_number() over(partition by user_id order by server_time) rank from playlog ").createOrReplaceTempView("a")
    spark.sql("select user_id,video_type,server_time,rank-1 as rank from a order by user_id,server_time").createOrReplaceTempView("b")
    spark.sql("select a.user_id,a.video_type,( b.server_time-a.server_time) watch_time from a left join b on a.user_id=b.user_id and a.rank=b.rank order by a.server_time").createOrReplaceTempView("c")
    spark.sql("select user_id,video_type, watch_time/60000.0 watch_time from c where watch_time is not null").createOrReplaceTempView("d")

    var userWatch = spark.sql("select user_id,sum(watch_time) as user_time from d group by user_id")
    var userDevice = spark.sql("select user_id,count(device_id) as device_freq from playlog group by user_id")
    var userDeviceUni = spark.sql("select user_id,count(distinct(device_id)) as device_uni from playlog group by user_id")
    var userVod = spark.sql("select user_id,sum(watch_time) as vod_time from d where video_type='vod' group by user_id")
    var userLive = spark.sql("select user_id,sum(watch_time) as live_time from d where video_type='live' group by user_id")

    var dfTrain = userWatch.join(userDevice, "user_id")
    dfTrain = dfTrain.join(userDeviceUni, "user_id")

    dfTrain.createOrReplaceTempView("dfa")
    userVod.createOrReplaceTempView("dfb")
    dfTrain = spark.sql("select dfa.user_id,user_time,device_freq,device_uni,vod_time from dfa left join dfb on dfa.user_id=dfb.user_id")

    dfTrain.createOrReplaceTempView("dfc")
    userLive.createOrReplaceTempView("dfd")
    dfTrain = spark.sql("select dfc.user_id,user_time,device_freq,device_uni,vod_time,live_time from dfc left join dfd on dfc.user_id=dfd.user_id")

    //    用户搜索次数特征
    var userSearch = spark.sql("select user_id, count(1) userSearch  from userlog where event_name='search' group by user_id ")

    dfTrain.createOrReplaceTempView("dfh")
    userSearch.createOrReplaceTempView("user_search")
    dfTrain = spark.sql("select dfh.user_id,user_time,device_freq,device_uni,vod_time,live_time,userSearch from dfh left join user_search on dfh.user_id=user_search.user_id")

    //    产品类特征
    //    20180501的数据中，注释的前3个统计量均为0，不作为特征；共有2703452条记录，支付状态为1的记录个数有1744104条；共有999815个用户，已支付用户数有680693，退订用户数396407；
    //    var userProductBuy = spark.sql("select nns_user_id user_id, count(1) userProductBuy from nnsBuyOrder where nns_order_type='product_buy' and nns_order_state=1 group by nns_user_id")
    //    var userPay = spark.sql("select nns_user_id user_id, count(1) userPay from nnsBuyOrder where nns_order_type='user_pay' and nns_order_state=1 group by nns_user_id")
    //    var userProductUpgrade = spark.sql("select nns_user_id user_id, count(1) userProductUpgrade from nnsBuyOrder where nns_order_type='product_upgrade' and nns_order_state=1 group by nns_user_id")

    //    每个用户id支付的次数
    var userOther = spark.sql("select nns_user_id user_id, count(1) userOther from nnsBuyOrder where nns_order_type='other' and nns_order_state=1 group by nns_user_id")
    //每个用户id支付总价格
    var userOffer = spark.sql("select nns_user_id user_id,sum(nns_order_price) userOffer from nnsBuyOrder where nns_order_state=1 group by nns_user_id")
    //    每个用户id退订的次数
    var userCancel = spark.sql("select nns_user_id user_id,count(1) userCancel from nnsBuyOrder where nns_order_state=5 group by nns_user_id")
    //    每个用户id退订总价格

    dfTrain.createOrReplaceTempView("dfe")
    userOther.createOrReplaceTempView("user_other")
    dfTrain = spark.sql("select dfe.user_id,user_time,device_freq,device_uni,vod_time,live_time,userSearch,userOther from dfe left join user_other on dfe.user_id=user_other.user_id")

    dfTrain.createOrReplaceTempView("dff")
    userOffer.createOrReplaceTempView("user_offer")
    dfTrain = spark.sql("select dff.user_id,user_time,device_freq,device_uni,vod_time,live_time,userSearch,userOther,userOffer from dff left join user_offer on dff.user_id=user_offer.user_id")

    dfTrain.createOrReplaceTempView("dfg")
    userCancel.createOrReplaceTempView("user_cancel")
    dfTrain = spark.sql("select dfg.user_id,user_time,device_freq,device_uni,vod_time,live_time,userSearch,userOther,userOffer,userCancel from dfg left join user_cancel on dfg.user_id=user_cancel.user_id")

    //    用户label，0正常，4停机，5销户
    var userLabel = spark.sql("select nns_id user_id,nns_state from nnsUser where nns_state=0 or nns_state=4 or nns_state=5")
    dfTrain.createOrReplaceTempView("dfi")
    userLabel.createOrReplaceTempView("user_label")
    dfTrain = spark.sql("select dfi.user_id,user_time,device_freq,device_uni,vod_time,live_time,userSearch,userOther,userOffer,userCancel,nns_state from dfi join user_label on dfi.user_id=user_label.user_id")

    dfTrain
  }

  def trainFeatureProcess(spark: SparkSession, data: sql.DataFrame): DataFrame = {

    import spark.implicits._
    //    填充缺失值
    var data1 = data.na.fill(Map(
      "vod_time" -> 10215.79,
      "live_time" -> 20872.1,
      "userSearch" -> -1,
      "userOther" -> -1,
      "userOffer" -> -1,
      "userCancel" -> -1
    ))

    //更改字段类型
    data1 = data1.select(data1("user_id"), data1("user_time").cast(DoubleType), data1("device_freq").cast(DoubleType), data1("device_uni").cast(DoubleType), data1("vod_time").cast(DoubleType), data1("live_time").cast(DoubleType), data1("userSearch").cast(DoubleType), data1("userOther").cast(DoubleType), data1("userOffer").cast(DoubleType), data1("userCancel").cast(DoubleType), data1("nns_state").cast(IntegerType))


    //    对label进行处理
    var data2 = data1.map(row => {
      val user_id = row.getAs[String](0)
      val row0 = row.getAs[Integer]("nns_state")
      val value = if (row0 == 4 || row0 == 5) 1 else 0
      (user_id, value)
    }).toDF("user_id", "label")
    data1 = data1.join(data2, "user_id")
    data1 = data1.drop("nns_state")

    //转换为向量类型
    val assembler = new VectorAssembler().setInputCols(Array("user_time", "device_freq", "device_uni", "vod_time", "live_time", "userSearch", "userOther", "userOffer", "userCancel")).setOutputCol("features")
    var output = assembler.transform(data1)
    val dataAll = output.select("user_id", "features", "label")

    dataAll
  }

  def testFeatureProcess(spark: SparkSession, data: sql.DataFrame): DataFrame = {
    //    填充缺失值
    var data1 = data.na.fill(Map(
      "vod_time" -> 10215.79,
      "live_time" -> 20872.1,
      "userSearch" -> -1,
      "userOther" -> -1,
      "userOffer" -> -1,
      "userCancel" -> -1
    ))

    //更改字段类型
    data1 = data1.select(data1("user_id"), data1("user_time").cast(DoubleType), data1("device_freq").cast(DoubleType), data1("device_uni").cast(DoubleType), data1("vod_time").cast(DoubleType), data1("live_time").cast(DoubleType), data1("userSearch").cast(DoubleType), data1("userOther").cast(DoubleType), data1("userOffer").cast(DoubleType), data1("userCancel").cast(DoubleType))
    data1.show(10)


    //转换为向量类型
    val assembler = new VectorAssembler().setInputCols(Array("user_time", "device_freq", "device_uni", "vod_time", "live_time", "userSearch", "userOther", "userOffer", "userCancel")).setOutputCol("features")
    var output = assembler.transform(data1)
    val dataAll = output.select("user_id", "features")
    dataAll.show()
    dataAll
  }

  def setModel(data: DataFrame): Unit = {

    log.info("模型建立与预测")

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(10).fit(data)

    val lr = new LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    //    管道
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lr, labelConverter))

    //    建立模型
    val model = pipeline.fit(trainingData)

    //测试
    val predictions = model.transform(testData)
    predictions.show()

    var evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")
    val precision = evaluator.evaluate(predictions)
    println("precision=" + precision)

    evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("weightedRecall")
    val recall = evaluator.evaluate(predictions)
    println("recall=" + recall)

    //    持久化
    model.write.overwrite().save("file:\\F:\\MyDocuments\\starcor\\user_loss_1.1\\model")
  }

  def makePredictions(data: sql.DataFrame): DataFrame = {

    val model = PipelineModel.load("file:\\F:\\MyDocuments\\starcor\\user_loss_1.1\\model")

    var testLabel = model.transform(data)

    testLabel = testLabel.select("user_id", "predictedLabel")

    testLabel

  }

  def evaluatePredictions(spark: SparkSession, pred: sql.DataFrame): Unit = {

    import spark.implicits._

    var nnsUser = spark.sql("select * from nns_user where day=20180612")

    var predEvaluate = pred.join(nnsUser, pred("user_id") === nnsUser("nns_id"))
    print(predEvaluate.count())
    predEvaluate.show(5)

    predEvaluate = predEvaluate.select(predEvaluate("user_id"), predEvaluate("predictedLabel").cast(DoubleType), predEvaluate("nns_state").cast(DoubleType))
    predEvaluate.show(5)

    //    对label进行处理
    var data2 = predEvaluate.map(row => {
      val user_id = row.getAs[String](0)
      val row0 = row.getAs[Double]("nns_state")
      val value = if (row0 == 4 || row0 == 5) 1.0 else 0.0
      (user_id, value)
    }).toDF("user_id", "label")
    predEvaluate = predEvaluate.join(data2, "user_id")
    predEvaluate = predEvaluate.drop("nns_state")
    predEvaluate.show()
    predEvaluate.repartition(1).write.option("header", "true").csv("file:\\F:\\MyDocuments\\starcor\\user_loss_1.1\\result\\result2")

    var evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("predictedLabel")
      .setMetricName("weightedPrecision")
    val precision = evaluator.evaluate(predEvaluate)
    println("precision=" + precision)

    evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("predictedLabel")
      .setMetricName("weightedPrecision")
    val recall = evaluator.evaluate(predEvaluate)
    println("recall=" + recall)

  }

}
