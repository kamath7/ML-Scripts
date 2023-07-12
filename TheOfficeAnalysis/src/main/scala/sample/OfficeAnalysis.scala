package sample
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{avg, col, desc, floor, regexp_replace, when}

object OfficeAnalysis {

  def main (args: Array[String]) : Unit = {
    val spark = SparkSession.builder().appName("Office").master("local[*]").getOrCreate()

    val df = spark.read.option("header","true").csv("D:\\Code\\MachineLearning-Scripts\\TheOfficeAnalysis\\src\\main\\scala\\sample\\the_office_series.csv")
//    df.printSchema()

    import spark.implicits._

    val cleanedUpDf = df.withColumn("season_cleaned", regexp_replace(col("Season"), "[^0-9.]", ""))
      .withColumn("season_cleaned", when(col("season_cleaned").isNull, null)
        .otherwise(floor(col("season_cleaned").cast("double") % 10)))
      .filter(col("season_cleaned").isNotNull)

    cleanedUpDf.show(truncate = false)

    val episodeCountsBySeason = cleanedUpDf.groupBy("season_cleaned").count().orderBy("season_cleaned")
    episodeCountsBySeason.show()

    val averageRatingsBySeason = cleanedUpDf.groupBy("season_cleaned").agg(avg("ratings").alias("Average_Ratings"), avg("Viewership").alias("Average_Viewership"))
    averageRatingsBySeason.show()

    val topRatedEpisodes = cleanedUpDf.orderBy(desc("ratings")).limit(10)
    topRatedEpisodes.show()
    spark.stop()
  }
}
