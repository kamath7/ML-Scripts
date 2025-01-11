package org.kamathad;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);

        try {
            String modelPath = "C:\\Users\\adith\\OneDrive\\Desktop\\Code Work\\ML-Scripts\\OMNX\\Java\\TestingONNX\\untitled\\src\\main\\resources\\lalle_model.onnx";
            OrtEnvironment environment = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();


        } catch (Exception e){
            e.printStackTrace();
        }finally {
            scanner.close();
        }
    }
}