using System;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using Tesseract;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace ComputerPartIdentifier
{
    class Program
    {
        static void Main(string[] args)
        {
            string imagesFolder = Path.Combine(Directory.GetCurrentDirectory(), "images");
            string imagePath = Path.Combine(imagesFolder, "processor.jpg");

            if (File.Exists(imagePath))
            {
                try
                {
                    Console.WriteLine("Przetwarzanie obrazu...");
                    Mat image = CvInvoke.Imread(imagePath);
                    Mat preprocessedImage = PreprocessImage(image);
                    string processedImagePath = Path.Combine(imagesFolder, "processed.jpg");
                    CvInvoke.Imwrite(processedImagePath, preprocessedImage);
                    Console.WriteLine("Zapisano obraz przetworzony: " + processedImagePath);

                    string tessdataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "tessdata");
                    using (var engine = new TesseractEngine(tessdataPath, "eng", EngineMode.Default))
                    using (var img = Pix.LoadFromFile(processedImagePath))
                    using (var page = engine.Process(img))
                    {
                        string recognizedText = page.GetText();
                        Console.WriteLine("Rozpoznany tekst:\n" + recognizedText);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Wystąpił błąd: " + ex.Message);
                }
            }
            else
            {
                Console.WriteLine("Plik nie istnieje. Upewnij się, że zdjęcie jest w folderze 'images' i nazywa się 'processor.jpg'.");
            }
        }

        static Mat PreprocessImage(Mat img)
        {
            Mat gray = new Mat();
            CvInvoke.CvtColor(img, gray, ColorConversion.Bgr2Gray);
            gray = Deskew(gray);

            CvInvoke.CLAHE(gray, 2.0, new Size(8, 8), gray);

            Mat binary = new Mat();
            CvInvoke.AdaptiveThreshold(gray, binary, 255, AdaptiveThresholdType.GaussianC, ThresholdType.BinaryInv, 21, 5);

            Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(3, 3), new Point(-1, -1));
            Mat gradient = new Mat();
            CvInvoke.MorphologyEx(binary, gradient, MorphOp.Gradient, kernel, new Point(-1, -1), 1, BorderType.Reflect, new MCvScalar());

            Mat cleaned = new Mat();
            CvInvoke.MedianBlur(gradient, cleaned, 3);

            return cleaned;
        }

        static Mat Deskew(Mat img)
        {
            Moments m = CvInvoke.Moments(img);
            if (Math.Abs(m.Mu02) < 1e-2)
                return img;

            double skew = m.Mu11 / m.Mu02;

            Mat warpMat = new Mat(2, 3, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
            float[] data = { 1, (float)skew, (float)(-0.5 * skew * img.Rows), 0, 1, 0 };
            CvInvoke.SetIdentity(warpMat, new MCvScalar(0));
            warpMat.SetTo(new MCvScalar(0));
            Marshal.Copy(data, 0, warpMat.DataPointer, data.Length);

            Mat deskewed = new Mat();
            CvInvoke.WarpAffine(img, deskewed, warpMat, img.Size(), Inter.Linear, Warp.Default, new MCvScalar(255));

            return deskewed;
        }
    }
}