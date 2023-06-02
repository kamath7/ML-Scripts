package sample

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession


object MyMain {
  def main(args: Array[String]):Unit = {

    val spark = SparkSession.builder().appName("MyAwesomeModel").getOrCreate()

    val dataset = spark.read.format("csv").option("header","true").option("inferSchema","true").load("D:\\Code\\MachineLearning-Scripts\\Spark-Simple_LR\\dataset\\Salary_Data.csv")


  }
}
