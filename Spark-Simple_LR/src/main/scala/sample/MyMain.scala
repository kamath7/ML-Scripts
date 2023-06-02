package sample

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession


object MyMain {
  def main(args: Array[String]):Unit = {

    val spark = SparkSession.builder().appName("MyAwesomeModel").getOrCreate()

    val dataset = spark.read.format("csv").option("header","true").option("inferSchema","true").load("D:\\Code\\MachineLearning-Scripts\\Spark-Simple_LR\\dataset\\Salary_Data.csv")

    val assembler = new VectorAssembler().setInputCols(Array("YearsExperience")).setOutputCol("features")

    val assembledData = assembler.transform(dataset).select("features","Salary")

    //Train-test split

    val Array(trainingData, testData) = assembledData.randomSplit(Array(0.8, 0.2), seed = 123)

    val lrModel = new LinearRegression().setLabelCol("Salary").setFeaturesCol("features")

    val model = lrModel.fit(trainingData)


  }
}
