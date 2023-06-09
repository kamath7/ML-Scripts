package sample
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{avg, col, desc, floor, max, regexp_replace, struct, when, min}
import org.apache.log4j._
object OfficeAnalysis {

  def main (args: Array[String]) : Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
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


    val bestEpisodeBySeason = cleanedUpDf.groupBy("season_cleaned")
      .agg(max(struct(col("Ratings"), col("EpisodeTitle"))).alias("best_episode"))
      .select("season_cleaned", "best_episode.*")
      .orderBy("season_cleaned")

    bestEpisodeBySeason.show(truncate = false)

    /*
    * +--------------+-------+----------------+
|season_cleaned|Ratings|EpisodeTitle    |
+--------------+-------+----------------+
|1             |8.4    |Basketball      |
|2             |9.4    |Casino Night    |
|3             |9.3    |The Job         |
|4             |9.5    |Dinner Party    |
|5             |9.7    |Stress Relief   |
|6             |9.4    |Niagara: Part 2 |
|7             |9.8    |Goodbye, Michael|
|8             |8.1    |The List        |
|9             |9.8    |Finale          |
+--------------+-------+----------------+

    *
    * */

    val worstEpisodeBySeason = cleanedUpDf.groupBy("season_cleaned")
      .agg(min(struct(col("Ratings"), col("EpisodeTitle"))).alias("worst_episode"))
      .select("season_cleaned", "worst_episode.*")
      .orderBy("season_cleaned")

    worstEpisodeBySeason.show(truncate = false)
    spark.stop()
  }
}
