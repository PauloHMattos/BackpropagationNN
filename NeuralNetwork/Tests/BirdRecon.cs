using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using NeuralNetwork.Activations;
using NeuralNetwork.Layers;
using NeuralNetwork.Networks;
using NeuralNetwork.Training;
using NeuralNetwork.Utils;

namespace NeuralNetwork.Tests
{
    public static partial class TestCases
    {
        public static void BirdRecognition(string[] commands)
        {
            var br = new BirdRecon();
            br.LoadTrainData();
            br.LoadTestData();

            int numInput = br.TestData[0].Length - 2;
            const int numHidden = 2;
            const int numOutput = 2;

            var trainData = br.TrainData.ToArray();
            var testData = br.TestData.ToArray();

            var inputs = new List<double[]>();
            var targetOutputs = new List<double[]>();
            foreach (var data in trainData)
            {
                inputs.Add(data.Take(numInput).ToArray());
                targetOutputs.Add(data.Skip(numInput).Take(numOutput).ToArray());
            }

            Normalize(trainData, numInput);
            Normalize(testData, numInput);

            var inputLayer = new Layer(numInput, new IdentityActivation());
            var hiddenLayer = new Layer(numHidden, new SigmoidActivation());
            var outputLayer = new Layer(numOutput, new SigmoidActivation());

            var net = new BackpropagationNet()
                .InputLayer(inputLayer)
                .HiddenLayers(hiddenLayer)
                .OutputLayer(outputLayer);
            net.BuildConnections();

            var trainConfiguration = new TrainConfiguration
            {
                MaxEpochs = 10,
                LearnRate = 0.4,
                MinError = 0.1,
                Momentum = 0.6,
                WeightDecay = 0
            };
            
            TrainResult result;

            var watch = new Stopwatch();

            TestCasesUtils.ReportStart(trainConfiguration.MaxEpochs, trainConfiguration.MinError, trainConfiguration.LearnRate, trainConfiguration.Momentum, trainConfiguration.WeightDecay);
            watch.Start();
            net.Train(trainConfiguration, inputs, targetOutputs, out result);
            watch.Stop();

            TestCasesUtils.ReportEnd(watch, result.Epochs, result.Error);

            Console.WriteLine("Precisão com os dados de treinamento: " + net.Accuracy(trainData, numInput, numOutput).ToString("F4"));
            Console.WriteLine("Precisão com os dados de teste: " + net.Accuracy(testData, numInput, numOutput).ToString("F4"));

            var output = new List<double>();
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
                    net.FeedForward(data);
                    output.Clear();
                    net.GetOutputs(output);
                    ConsoleUtils.ShowVector(output, 2, 5, false);
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
