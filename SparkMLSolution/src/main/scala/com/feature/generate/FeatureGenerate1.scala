package com.feature.generate

import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

object FeatureGenerate1 {

  private val log = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]): Unit = {
    val startTime1 = System.currentTimeMillis()
    val spark = SparkSession.builder().appName("scgdUserLoss").enableHiveSupport().config("spark.sql.hive.convertMetastoreParquet", "false").config("spark.sql.parquet.filterPushdown", "false").getOrCreate()
    var playlog=spark.sql("select * from scgd.play_log where day between 20180501 and 20180531")
    var userlog=spark.sql("select * from scgd.user_log where day between 20180501 and 20180531")
    var nnsBuyOrder=spark.sql("select * from scgd.nns_buy_order where day=20180531")

    playlog.createOrReplaceTempView("playlog")
    userlog.createOrReplaceTempView("userlog")

    nnsBuyOrder.createOrReplaceTempView("nnsBuyOrder")
    spark.sql("select *, from_unixtime(nns_modify_time/1000,'yyyy-MM-dd') time from nnsBuyOrder where from_unixtime(nns_modify_time/1000,'yyyy-MM-dd') between '2018-03-24' and '2018-04-30'").createOrReplaceTempView("nnsBuyOrder")


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

    // 产品类特征
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

    dfTrain.write.parquet("/user/portrait/scgd/data1")

    val endTime1 = System.currentTimeMillis()
    log.info("Total Time:" + ((endTime1 - startTime1) / 60000) + " minutes")
  }
}
