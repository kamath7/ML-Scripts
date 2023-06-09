package sample

import org.apache.log4j._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{ StructField, StructType}
import org.apache.spark.sql.{SparkSession}


object MyMain {
  def main(args: Array[String]):Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    //Creating spark session
    val spark = SparkSession.builder().appName("MyAwesomeModel").master("local[*]").getOrCreate()

    //Reading the dataset
    val dataset = spark.read.format("csv").option("header","true").option("inferSchema","true").load("D:\\Code\\MachineLearning-Scripts\\Spark-Simple_LR\\src\\main\\scala\\sample\\Salary_Data.csv")

    //Preprocessing
    val assembler = new VectorAssembler().setInputCols(Array("YearsExperience")).setOutputCol("features")

    val assembledData = assembler.transform(dataset).select("features","Salary")

    //Train-test split

    val Array(trainingData, testData) = assembledData.randomSplit(Array(0.8, 0.2), seed = 123)

    val lrModel = new LinearRegression().setLabelCol("Salary").setFeaturesCol("features")

    val model = lrModel.fit(trainingData)

    //Testing predictions with our test data
    val prediction = model.transform(testData)

    import spark.implicits._
    def makePrediction(yoe: Double): Unit = {
      val inputFeatures = Seq(Vectors.dense(yoe))
      val schema = StructType(Seq(StructField("features", org.apache.spark.ml.linalg.SQLDataTypes.VectorType, nullable = false)))
      val inputData = spark.createDataFrame(inputFeatures.map(Tuple1.apply)).toDF("features").select("features")

      // Use the trained model to make predictions
      val predictions = model.transform(inputData)

      // Access the predicted salary
      val predictedSalary = predictions.select("prediction").first().getDouble(0)

      // Print the predicted salary
      println(s"Predicted Salary for $yoe years of experience: $predictedSalary")
    }

    makePrediction(3)
    makePrediction(6)
    makePrediction(10)
  }
}
