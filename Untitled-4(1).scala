def getDKHKEventFeature(df: DataFrame, sampleDF:DataFrame, watch_time: String, max_window: Int=366, event_type: String="DKHK_event", tmp_path :String="/path/to/save/medium_result"): DataFrame = {
    import org.apache.spark.sql.DataFrame
    import org.apache.hadoop.fs.{FileSystem, Path,FileStatus,FileUtil}
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val hdfs = FileSystem.get(hadoopConf)
    val outputPath = new Path(tmp_path)
    if(!hdfs.exists(outputPath)){
      throw new Exception("!!!! Error, tmp_path is invalid !!!!")
    }
    
    var df2=df
    /*Step 1:先判断传入的sampleDF是否有误，如无误进行样本过滤*/
    val tmp_cust=sampleDF.select("cust_id").dropDuplicates()
    val tmp_prt=sampleDF.select("prt_dt").dropDuplicates()
    val cust_num=sampleDF.select("cust_id").distinct().count()
    if(tmp_prt.count()==1 && tmp_prt.head(1)(0)(0).toString==watch_time){
      df2=df2.join(tmp_cust,Seq("cust_id"),"inner")
    }
    else{
      throw new Exception("!!!! Error, Input sampleDF is invalid, the prt_dt is incorrect !!!!")
    }
    println("********Starting DKHKEvent Feature Module...********")
    /*Step 2:提前进行与时间窗不相关的操作*/
    df2=df2.orderBy("event_tm")
    if(df2.columns.contains("shld_prncp") && df2.columns.contains("real_prncp"))
      df2 = df2.withColumn("unpay_prncp",col("shld_prncp")-col("real_prncp"))
    if(df2.columns.contains("shld_intrst") && df2.columns.contains("real_intrst"))
      df2 = df2.withColumn("unpay_intrst",col("shld_intrst")-col("real_intrst"))
    if(df2.columns.contains("shld_pent_intrst") && df2.columns.contains("real_pent_intrst"))
      df2 = df2.withColumn("unpay_pent_intrst",col("shld_pent_intrst")-col("real_pent_intrst"))
    /**Step 3:进行时间窗相关的操作
      * 注意：
      * 1）判断字段是否存在，存在才进行对应衍生
      * 2）字段命名需要带有所选时间窗信息
      */
    var final_fea=tmp_cust.withColumn("prt_dt",to_date(lit(watch_time))) //LY:to_date
    val window_array = Array(7,30,60,90,180,365)
    val feat_data_array = new Array[org.apache.spark.sql.DataFrame](window_array.length)
    for (i <- 0 until window_array.length){
      var fea_data = spark.emptyDataFrame
      if(max_window>=window_array(i)){
        val hist_time = getPreviousdate(watch_time,window_array(i))
        var his_df=df2.filter(($"event_tm">=hist_time) && ($"event_tm"<=watch_time))
        if(his_df.head(1).isEmpty){
            throw new Exception("!!!! Error, his_df is empty !!!!")
        }

        val num_cols = Array("real_prncp","real_intrst","real_pent_intrst","real_total","shld_prncp","shld_intrst","shld_pent_intrst","unpay_prncp","unpay_intrst","unpay_pent_intrst","prncp_bal")
        val num_cols1 = num_cols.filter(c=> his_df.columns.contains(c))
        //num_cols1.foreach(println)
        var num_feaFunc = new Array[org.apache.spark.sql.Column](0)
        if(num_cols1.length!=0){
            num_feaFunc = num_feaFunc ++ num_cols1.flatMap(col=>{Array(min(col),max(col),sum(col))})
        }
        val cat_cols = Array("loan_id")
        val cat_cols1 = cat_cols.filter(c=> his_df.columns.contains(c))
        
        var cat_feaFunc = new Array[org.apache.spark.sql.Column](0)
        if(cat_cols1.length!=0){
            cat_feaFunc = cat_feaFunc ++ cat_cols1.flatMap(col=>{Array(countDistinct(col).as("unique_"+col))})
        }

        var intersect_feaFunc = new Array[org.apache.spark.sql.Column](0)
        if(his_df.columns.contains("prncp_bal")){
            intersect_feaFunc = intersect_feaFunc++Array((col("min(prncp_bal)")/(col("max(prncp_bal)")+1e-10)).alias("prncp_bal_ratio"))
        }
        if(his_df.columns.contains("real_total") && his_df.columns.contains("real_intrst")){
            intersect_feaFunc = intersect_feaFunc++Array((col("sum(real_intrst)")/(col("sum(real_total)")+1e-10)).alias("intrst_total_ratio"))
        }

        val feaFunc1 = num_feaFunc ++ cat_feaFunc
        val feaFunc2 = intersect_feaFunc

        if(feaFunc1.length!=0){
            fea_data = his_df.groupBy("cust_id").agg(feaFunc1.head,feaFunc1.tail:_*).repartition((0.00001*cust_num).toInt)
        }
        if(feaFunc2.length!=0){
            fea_data = fea_data.select($"*"+:intersect_feaFunc:_*).repartition((0.00001*cust_num).toInt)
        }

        val cols=fea_data.columns.map(z=>if(z=="cust_id") z else z+"_"+window_array(i)+"Days")//添加时间窗作为标识
        fea_data=fea_data.toDF(cols:_*)
        val window_size = window_array(i)
        println(raw"window: $window_size feature finished")
        //中间结果落盘
        val alias_cols_window = fea_data.columns.map(z=>if(Array("cust_id","prt_dt").contains(z)) z else event_type+"_"+z.replace("(","*").replace(")","*"))
        fea_data = fea_data.toDF(alias_cols_window:_*)
        println("开始落盘")
        fea_data.write.mode("overwrite").parquet(tmp_path+event_type+"_tmp_"+window_array(i)+"Days")
        println("完成落盘")
      }
      //feat_data_array(i) = fea_data
      
      //if(! fea_data.head(1).isEmpty){
      //  final_fea=final_fea.join(fea_data,Seq("cust_id"),"left").na.fill(0)
      //}
    }
    //时间窗口3：历史所有
    var fea_data_all_his=spark.emptyDataFrame
    var his_df=df2.filter($"event_tm"<=watch_time)
    fea_data_all_his=his_df.select("cust_id").dropDuplicates()
    if(his_df.columns.contains("prncp_bal")){
      val tmp_fea_all1=his_df.groupBy("cust_id").agg(last("prncp_bal").as("last_prncp_bal"))
      fea_data_all_his=fea_data_all_his.join(tmp_fea_all1,Seq("cust_id"),"left")
    }

    if(his_df.columns.contains("event_tm")){
      var tmp_fea_all2=his_df.groupBy("cust_id").agg(last("event_tm", false)).withColumn("watch_time",lit(watch_time))
      tmp_fea_all2 = tmp_fea_all2.withColumn("event_tm"+"_last"+"_interval",datediff(col("last(event_tm, false)"),col("watch_time")))
      tmp_fea_all2 = tmp_fea_all2.withColumn("event_tm"+"_last"+"_interval"+"_transform",lit(366.0)/(lit(1.0)+col("event_tm"+"_last"+"_interval"))) //LY:为什么不添加366/1+interval
      fea_data_all_his=fea_data_all_his.join(tmp_fea_all2.select("cust_id","event_tm"+"_last"+"_interval"+"_transform"),Seq("cust_id"),"left")
    }
    val cols=fea_data_all_his.columns.map(z=>if(z=="cust_id") z else z+"_"+"AllDays")
    fea_data_all_his=fea_data_all_his.toDF(cols:_*).repartition((0.00001*cust_num).toInt)
    println("his_all feature finished")
    //中间结果落盘
    val alias_cols_all = fea_data_all_his.columns.map(z=>if(Array("cust_id","prt_dt").contains(z)) z else event_type+"_"+z.replace("(","*").replace(")","*"))
    fea_data_all_his = fea_data_all_his.toDF(alias_cols_all:_*)
    fea_data_all_his.write.mode("overwrite").parquet(tmp_path+event_type+"_tmp_"+"AllDays")

    /**读入数据 **/
    for (i <- 0 until window_array.length){
      if(max_window>=window_array(i)){
        feat_data_array(i) = spark.read.parquet(tmp_path+event_type+"_tmp_"+window_array(i)+"Days")
        final_fea=final_fea.join(feat_data_array(i),Seq("cust_id"),"left").na.fill(0)
      }
    }
    fea_data_all_his = spark.read.parquet(tmp_path+event_type+"_tmp_"+"AllDays")

    /*Step 4:处理最终特征宽表的字段形式*/
    //加入his_all

    if(! fea_data_all_his.head(1).isEmpty){
      final_fea=final_fea.join(fea_data_all_his,Seq("cust_id"),"left").na.fill(0)
    }
    println("拼接各窗口特征完成")
    val alias_cols = final_fea.columns.map(z=>if(Array("cust_id","prt_dt").contains(z)) z else event_type+"_"+z.replace("(","*").replace(")","*"))
    final_fea = final_fea.toDF(alias_cols:_*)

    val fea_cols = final_fea.columns.filter(c=> !Array("cust_id","prt_dt").contains(c))  //保证统一字段顺序"cust_id", "prt_dt", "Feature1~N"
    val fea_cols_length = fea_cols.length
    println(raw"feature size: $fea_cols_length")
    println("feature name:")
    fea_cols.foreach(println)
    println("****Finish DKHKEvent Feature Module...****")
    final_fea
  }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def getSJYHJYEventFeature(df: DataFrame, sampleDF:DataFrame, watch_time: String, max_window: Int=366, event_type: String="SJYHJY_event", tmp_path :String="/path/to/save/medium_result"): DataFrame = {
    import org.apache.spark.sql.DataFrame
    import org.apache.hadoop.fs.{FileSystem, Path,FileStatus,FileUtil}
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val hdfs = FileSystem.get(hadoopConf)
    val outputPath = new Path(tmp_path)
    if(!hdfs.exists(outputPath)){
      throw new Exception("!!!! Error, tmp_path is invalid !!!!")
    }

    var df2=df
    /*Step 1:先判断传入的sampleDF是否有误，如无误进行样本过滤*/
    val tmp_cust=sampleDF.select("cust_id").dropDuplicates()
    val tmp_prt=sampleDF.select("prt_dt").dropDuplicates()
    val cust_num=sampleDF.select("cust_id").distinct().count()
    if(tmp_prt.count()==1 && tmp_prt.head(1)(0)(0).toString==watch_time){
      df2=df2.join(tmp_cust,Seq("cust_id"),"inner")
    }
    else{
      throw new Exception("!!!! Error, Input sampleDF is invalid, the prt_dt is incorrect !!!!")
    }
    println("********Starting SJYHJYEvent Feature Module...********")
    /*Step 2:提前进行与时间窗不相关的操作*/
    df2=df2.orderBy("event_tm")

    /**Step 3:进行时间窗相关的操作
      * 注意：
      * 1）判断字段是否存在，存在才进行对应衍生
      * 2）字段命名需要带有所选时间窗信息
      */
    var final_fea=tmp_cust.withColumn("prt_dt",to_date(lit(watch_time))) //LY:to_date
    val window_array = Array(7,30,60,90,180,365)
    val feat_data_array = new Array[org.apache.spark.sql.DataFrame](window_array.length)
    for (i <- 0 until window_array.length){
        var fea_data = spark.emptyDataFrame
        if(max_window>=window_array(i)){
          val hist_time = getPreviousdate(watch_time,window_array(i))
          var his_df=df2.filter(($"event_tm">=hist_time) && ($"event_tm"<=watch_time))
          fea_data=his_df.select("cust_id").dropDuplicates()

          if(his_df.columns.contains("txt_amt")){
            val tmp_fea_txtAmtSta=his_df.groupBy("cust_id").agg(max("txt_amt"),min("txt_amt"),sum("txt_amt"),avg("txt_amt"),stddev("txt_amt"),count("txt_amt"))
            fea_data=fea_data.join(tmp_fea_txtAmtSta,Seq("cust_id"),"left")
          }

          val window_size = window_array(i)
          println(raw"window: $window_size feature finished")

          //中间结果落盘
          var alias_cols_window=fea_data.columns.map(z=>if(z=="cust_id") z else z+"_"+window_array(i)+"Days")//添加时间窗作为标识
          fea_data=fea_data.toDF(alias_cols_window:_*)

          alias_cols_window = fea_data.columns.map(z=>if(Array("cust_id","prt_dt").contains(z)) z else event_type+"_"+z.replace("(","*").replace(")","*"))
          fea_data = fea_data.toDF(alias_cols_window:_*).repartition((0.00001*cust_num).toInt)
          println("开始落盘")
          fea_data.write.mode("overwrite").parquet(tmp_path+event_type+"_tmp_"+window_array(i)+"Days")
          println("完成落盘")
          }
          //feat_data_array(i) = fea_data
    }
    //时间窗口2：历史所有
    var fea_data_all_his=spark.emptyDataFrame
    var his_df2=df2.filter($"event_tm"<=watch_time)
    fea_data_all_his=his_df2.select("cust_id").dropDuplicates()
    if(his_df2.columns.contains("txt_amt")){
      val tmp_fea_txtAmtLast=his_df2.groupBy("cust_id").agg(last("txt_amt")).withColumnRenamed("last(txt_amt, false)","last_txt_amt")
      fea_data_all_his=fea_data_all_his.join(tmp_fea_txtAmtLast,Seq("cust_id"),"left")
    }
    
    if(his_df2.columns.contains("event_tm")){
      var tmp_fea_tmLastIntv=his_df2.groupBy("cust_id").agg(last("event_tm", false)).withColumn("watch_time",lit(watch_time))
      tmp_fea_tmLastIntv = tmp_fea_tmLastIntv.withColumn("event_tm"+"_last"+"_interval",datediff(col("last(event_tm, false)"),col("watch_time")))
      tmp_fea_tmLastIntv = tmp_fea_tmLastIntv.withColumn("event_tm"+"_last"+"_interval"+"_transform",lit(366.0)/(lit(1.0)+col("event_tm"+"_last"+"_interval"))) //LY:为什么不添加366/1+interval
      fea_data_all_his=fea_data_all_his.join(tmp_fea_tmLastIntv.select("cust_id","event_tm"+"_last"+"_interval"+"_transform"),Seq("cust_id"),"left")
    }
    val cols2=fea_data_all_his.columns.map(z=>if(z=="cust_id") z else z+"_"+"AllDays")
    fea_data_all_his=fea_data_all_his.toDF(cols2:_*).repartition((0.00001*cust_num).toInt)
    //中间结果落盘
    val alias_cols_all = fea_data_all_his.columns.map(z=>if(Array("cust_id","prt_dt").contains(z)) z else event_type+"_"+z.replace("(","*").replace(")","*"))
    fea_data_all_his = fea_data_all_his.toDF(alias_cols_all:_*)
    fea_data_all_his.write.mode("overwrite").parquet(tmp_path+event_type+"_tmp_"+"AllDays")
    println("his_all finished")
    /*Step 4:处理最终特征宽表的字段形式*/
    /**读入数据 **/
    for (i <- 0 until window_array.length){
      if(max_window>=window_array(i)){
        feat_data_array(i) = spark.read.parquet(tmp_path+event_type+"_tmp_"+window_array(i)+"Days")
        final_fea=final_fea.join(feat_data_array(i),Seq("cust_id"),"left").na.fill(0)
      }
    }
    fea_data_all_his = spark.read.parquet(tmp_path+event_type+"_tmp_"+"AllDays")

    //加入his_all
    if(! fea_data_all_his.head(1).isEmpty){
      final_fea=final_fea.join(fea_data_all_his,Seq("cust_id"),"left").na.fill(0)
    }

    val alias_cols = final_fea.columns.map(z=>if(Array("cust_id","prt_dt").contains(z)) z else event_type+"_"+z.replace("(","*").replace(")","*"))
    final_fea = final_fea.toDF(alias_cols:_*)

    val fea_cols = final_fea.columns.filter(c=> !Array("cust_id","prt_dt").contains(c))  //LY:统一字段顺序"cust_id", "prt_dt", "Feature1~N"
    val fea_cols_length = fea_cols.length
    println(raw"feature size: $fea_cols_length")
    println("feature name:")
    fea_cols.foreach(println)
    println("****Finish SJYHJYEvent Feature Module...****")
    final_fea
  }
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
