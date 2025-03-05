using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx; // Kluczowe dla ApplyOnnxModel

public class ImageData
{
    public string ImagePath { get; set; }
}

public class ImagePrediction
{
    [ColumnName("output")]
    public float[] Scores { get; set; }

    public string PredictedLabel { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        var context = new MLContext();

        // Ścieżka do modelu ONNX
        var modelPath = "Models/mobilenetv2-7.onnx";

        // Ścieżka do zdjęcia testowego
        var imagePath = "Data/Test/test_image.jpg";

        // Przygotowanie danych
        var imageData = new ImageData { ImagePath = imagePath };
        var imageDataView = context.Data.LoadFromEnumerable(new[] { imageData });

        // Tworzenie pipeline
        var pipeline = context.Transforms.LoadImages(outputColumnName: "input", imageFolder: "", inputColumnName: nameof(ImageData.ImagePath))
            .Append(context.Transforms.ResizeImages(outputColumnName: "input", imageWidth: 224, imageHeight: 224, inputColumnName: "input"))
            .Append(context.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: true, offsetImage: 117))
            .Append(context.Transforms.ApplyOnnxModel(modelFile: modelPath, outputColumnNames: new[] { "output" }, inputColumnNames: new[] { "input" }));

        // Trenowanie modelu (w tym przypadku to tylko transformacja danych)
        var model = pipeline.Fit(imageDataView);

        // Przewidywanie
        var predictor = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
        var prediction = predictor.Predict(imageData);

        // Wyświetlenie wyniku
        Console.WriteLine($"Predicted Label: {prediction.PredictedLabel}");
    }
}