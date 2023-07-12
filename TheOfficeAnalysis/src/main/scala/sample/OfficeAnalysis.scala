package sample
import org.apache.spark.sql.SparkSession

object OfficeAnalysis {

  def main (args: Array[String]) : Unit = {
    val spark = SparkSession.builder().appName("Office").master("local[*]").getOrCreate()

    val df = spark.read.option("header","true").csv("D:\\Code\\MachineLearning-Scripts\\TheOfficeAnalysis\\src\\main\\scala\\sample\\the_office_series.csv")
    df.printSchema()
  }
}
