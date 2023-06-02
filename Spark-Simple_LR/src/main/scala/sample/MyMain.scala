package sample

import org.apache.spark.SparkContext
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}


object MyMain {
  def main(args: Array[String]):Unit = {

    val spark = SparkSession.builder().appName("MyAwesomeModel").master("local[*]").getOrCreate()

    val dataset = spark.read.format("csv").option("header","true").option("inferSchema","true").load("D:\\Code\\MachineLearning-Scripts\\Spark-Simple_LR\\src\\main\\scala\\sample\\Salary_Data.csv")

    val assembler = new VectorAssembler().setInputCols(Array("YearsExperience")).setOutputCol("features")

    val assembledData = assembler.transform(dataset).select("features","Salary")

    //Train-test split

    val Array(trainingData, testData) = assembledData.randomSplit(Array(0.8, 0.2), seed = 123)

    val lrModel = new LinearRegression().setLabelCol("Salary").setFeaturesCol("features")

    val model = lrModel.fit(trainingData)

    val prediction = model.transform(testData)

//    prediction.select("features","Salary","prediction").show()

    import spark.implicits._

    // Create a new DataFrame with input features for prediction
    val inputFeatures = Seq(Vectors.dense(13.0))
    val schema = StructType(Seq(StructField("features", org.apache.spark.ml.linalg.SQLDataTypes.VectorType, nullable = false)))
    val inputData = spark.createDataFrame(inputFeatures.map(Tuple1.apply)).toDF("features").select("features")

    // Use the trained model to make predictions
    val predictions = model.transform(inputData)

    // Access the predicted salary
    val predictedSalary = predictions.select("prediction").first().getDouble(0)

    // Print the predicted salary
    println(s"Predicted Salary for 13 years of experience: $predictedSalary")



  }
}
