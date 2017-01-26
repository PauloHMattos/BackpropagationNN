using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using NeuralNetwork.Activation;
using NeuralNetwork.Nets;

namespace NeuralNetwork.Tests
{
    public static partial class TestCases
    {
        public static void BirdRecognition()
        {
            var br = new BirdRecon();
            br.LoadTrainData();
            br.LoadTestData();
            
            int numInput = br.TestData[0].Length - 2;
            const int numHidden = 2;
            const int numOutput = 2;

            var trainData = br.TrainData.ToArray();
            var testData = br.TestData.ToArray();

            Normalize(trainData, numInput);
            Normalize(testData, numInput);

            var hiddenActivation = new SigmoidActivation();
            var outputActivation = new SigmoidActivation();
            NeuralNetData(numInput, numHidden, numOutput, hiddenActivation, outputActivation);
            var nn = new BackpropNeuralNet(numInput, numHidden, numOutput, hiddenActivation, outputActivation);
            nn.InitializeWeights();

            const int maxEpochs = 500;
            const double minSquaredError = 0.1;
            const double learnRate = 0.4;
            const double momentum = 0.6;
            const double weightDecay = 0;

            int epoch;
            double mse;

            var watch = new Stopwatch();

            ReportStart(maxEpochs, minSquaredError, learnRate, momentum, weightDecay);

            watch.Start();
            nn.Train(trainData, maxEpochs, minSquaredError, learnRate,
                momentum, weightDecay, out mse, out epoch);
            watch.Stop();

            ReportEnd(watch, epoch, mse);
            
            Console.WriteLine("Precisão com os dados de treinamento: " + nn.Accuracy(trainData).ToString("F4"));
            Console.WriteLine("Precisão com os dados de teste: " + nn.Accuracy(testData).ToString("F4"));

            do
            {
                while (!Console.KeyAvailable)
                {
                    Console.WriteLine("\nEscolha uma imagem para testar: ");
                    var path = Console.ReadLine();
                    if (!File.Exists("CIFAR10/" + path))
                    {
                        Console.WriteLine("O arquivo " + path + " não existe");
                        continue;
                    }
                    bool positive;
                    var data = br.GetData(path, out positive);
                    var outputs = nn.GetOutputs(data);
                    Utils.ShowVector(outputs, 2, 5, false);
                }
            } while (Console.ReadKey(true).Key != ConsoleKey.Escape);
        }

        private class BirdRecon
        {
            private readonly List<double[]> _trainData = new List<double[]>();
            private readonly List<double[]> _testData = new List<double[]>();

            public List<double[]> TestData
            {
                get { return _testData; }
            }

            public List<double[]> TrainData
            {
                get { return _trainData; }
            }

            public void LoadTrainData()
            {
                var prefix = "CIFAR10/data_batch_";
                var extension = ".bin";

                for (int i = 1; i <= 5; i++)
                {
                    using (var stream = File.Open(prefix + i + extension, FileMode.Open))
                    {
                        while (stream.Position != stream.Length)
                        {
                            var label = stream.ReadByte();
                            var pixels = new byte[3072];
                            stream.Read(pixels, 0, pixels.Length);

                            var data = new double[1026];
                            for (int j = 0; j < 1024; j++)
                            {
                                var grayScale = (double)pixels[j] / byte.MaxValue;
                                grayScale += (double)pixels[j + 1024] / byte.MaxValue;
                                grayScale += (double)pixels[j + 2048] / byte.MaxValue;
                                data[j] = grayScale/3;
                            }
                            data[1024] = (label == 2) ? 1.0 : 0.0;
                            data[1025] = (label == 2) ? 0.0 : 1.0;

                            _trainData.Add(data);
                        }
                    }
                }
            }

            public double[] GetData(string path, out bool positive)
            {
                var data = new double[32*32];
                var colors = new Color[32*32];
                var bmp = new Bitmap("CIFAR10/" + path);
                for (int y = 0; y < 32; y++)
                {
                    for (int x = 0; x < 32; x++)
                    {
                        colors[y * 32 + x] = bmp.GetPixel(x, y);
                    }
                }

                for (int i = 0; i < colors.Length; i++)
                {
                    var grayScale = (double)colors[i].R / byte.MaxValue;
                    grayScale += (double)colors[i].G / byte.MaxValue;
                    grayScale += (double)colors[i].B / byte.MaxValue;
                    data[i] = grayScale / 3;
                    //data[i] = colors[i].R;
                    //data[i + 1024] = colors[i].G;
                    //data[i + 2048] = colors[i].B;
                }
                positive = path.Contains("bird");
                return data;
            }

            public void LoadTestData()
            {
                using (var stream = File.Open("CIFAR10/test_batch.bin", FileMode.Open))
                {
                    while (stream.Position != stream.Length)
                    {
                        var label = stream.ReadByte();
                        var pixels = new byte[3072];
                        stream.Read(pixels, 0, pixels.Length);

                        var data = new double[1026];
                        for (int j = 0; j < 1024; j++)
                        {
                            var grayScale = (double)pixels[j] / byte.MaxValue;
                            grayScale += (double)pixels[j + 1024] / byte.MaxValue;
                            grayScale += (double)pixels[j + 2048] / byte.MaxValue;
                            data[j] = grayScale / 3;
                        }
                        data[1024] = (label == 2) ? 1.0 : 0.0;
                        data[1025] = (label == 2) ? 0.0 : 1.0;

                        //var label = stream.ReadByte();
                        //var pixels = new byte[3072];
                        //stream.Read(pixels, 0, pixels.Length);

                        //var data = new double[3074];
                        //for (int j = 0; j < pixels.Length; j++)
                        //{
                        //    data[j] = (double) pixels[j]/byte.MaxValue;
                        //}
                        //data[3072] = (label == 2) ? 1.0 : 0.0;
                        //data[3073] = (label == 2) ? 0.0 : 1.0;
                        _testData.Add(data);
                    }
                }
            }
        }
    }
}
