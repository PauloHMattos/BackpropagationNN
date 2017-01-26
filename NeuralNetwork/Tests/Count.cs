using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using NeuralNetwork.Activation;
using NeuralNetwork.Nets;

namespace NeuralNetwork.Tests
{
    public static partial class TestCases
    {
        public static void Count()
        {
            var inputs = new List<double[]>()
            {
                new double[] { 0, 0, 0, 0 }, // 0
                new double[] { 0, 0, 0, 1 }, // 1
                new double[] { 0, 0, 1, 0 }, // 2
                new double[] { 0, 0, 1, 1 }, // 3
                new double[] { 0, 1, 0, 0 },
                new double[] { 0, 1, 0, 1 },
                new double[] { 0, 1, 1, 0 },
                new double[] { 0, 1, 1, 1 },

                new double[] { 1, 0, 0, 0 },
                new double[] { 1, 0, 0, 1 },
                new double[] { 1, 0, 1, 0 },
                new double[] { 1, 0, 1, 1 },
                new double[] { 1, 1, 0, 0 },
                new double[] { 1, 1, 0, 1 },
                new double[] { 1, 1, 1, 0 },
                new double[] { 1, 1, 1, 1 }, // 15
            };

            var targetOutputs = new List<double[]>()
            {
                new double[] { 0, 0, 0, 1 }, // 0
                new double[] { 0, 0, 1, 0 }, // 1
                new double[] { 0, 0, 1, 1 }, // 2
                new double[] { 0, 1, 0, 0 }, // 3
                new double[] { 0, 1, 0, 1 },
                new double[] { 0, 1, 1, 0 },
                new double[] { 0, 1, 1, 1 },
                new double[] { 1, 0, 0, 0 },

                new double[] { 1, 0, 0, 1 },
                new double[] { 1, 0, 1, 0 },
                new double[] { 1, 0, 1, 1 },
                new double[] { 1, 1, 0, 0 },
                new double[] { 1, 1, 0, 1 },
                new double[] { 1, 1, 1, 0 },
                new double[] { 1, 1, 1, 1 },
                new double[] { 0, 0, 0, 0 }, // 15
            };

            var inputLayer = new Layer(4, new IdentityActivation());
            var hiddenLayer = new Layer(8, new HyperbolicTanActivation());
            var outputLayer = new Layer(4, new SigmoidActivation());

            var net = new BackpropagationNet()
                .InputLayer(inputLayer)
                .HiddenLayers(hiddenLayer)
                .OutputLayer(outputLayer);
            net.BuildConnections();

            var trainConfiguration = new TrainConfiguration
            {
                MaxEpochs = 1000,
                LearnRate = 0.6,
                MinError = 0.0005,
                Momentum = 0.1,
                WeightDecay = 0
            };

            TrainResult result;

            var watch = new Stopwatch();

            ReportStart(trainConfiguration.MaxEpochs, trainConfiguration.MinError, trainConfiguration.LearnRate, trainConfiguration.Momentum, trainConfiguration.WeightDecay);
            watch.Start();
            net.Train(trainConfiguration, inputs, targetOutputs, out result);
            watch.Stop();

            ReportEnd(watch, result.Epochs, result.Error);
            
            var output = new List<double>();
            foreach (var input in inputs)
            {
                net.FeedForward(input);
                net.GetOutputs(output);
                Utils.ShowVector(output, 4, 0, false);
                Thread.Sleep(500);
            }
        }
    }
}
