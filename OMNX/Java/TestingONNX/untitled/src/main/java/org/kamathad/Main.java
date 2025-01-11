package org.kamathad;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.util.Collection;
import java.util.Collections;
import java.util.Scanner;

public class Main {

    //To reverse-scale. I don't know how I can do this seamlessly yet
    private static final double MEAN = 5.7;
    private static final double STD = 2.5;

    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);

        try {
            String modelPath = "C:\\Users\\adith\\OneDrive\\Desktop\\Code Work\\ML-Scripts\\OMNX\\Java\\TestingONNX\\untitled\\src\\main\\resources\\lalle_model.onnx";
            OrtEnvironment environment = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            try (OrtSession session = environment.createSession(modelPath, options)) {
                System.out.println("Enter your years of experince");
                float yearsOfEx = scanner.nextFloat();
                double scaledInput = (yearsOfEx - MEAN) / STD;
                float[][] inputTensorData = new float[][]{{yearsOfEx}};

                OnnxTensor inputTensor = OnnxTensor.createTensor(environment, inputTensorData);

                try (OrtSession.Result result = session.run(Collections.singletonMap("x", inputTensor))) {
                    float[][] prediction = (float[][]) result.get(0).getValue();
                    double predictedSalary = (prediction[0][0] * STD) + MEAN;
                    System.out.printf("Predicted Salary: %.2f\n", predictedSalary);
                }
            }

        } catch (Exception e){
            e.printStackTrace();
        }finally {
            scanner.close();
        }
    }
}