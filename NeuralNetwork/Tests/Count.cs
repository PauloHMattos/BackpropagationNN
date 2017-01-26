using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using NeuralNetwork.Activations;
using NeuralNetwork.Layers;
using NeuralNetwork.Networks;
using NeuralNetwork.Training;
using NeuralNetwork.Utils;

namespace NeuralNetwork.Tests
{
    public static partial class TestCases
    {
        public static void Count(string[] commands)
        {
            var train = true;
            TrainResult result = new TrainResult();
            TrainConfiguration trainConfiguration = new TrainConfiguration
            {
                MaxEpochs = 1000,
                LearnRate = 0.6,
                MinError = 0.0005,
                Momentum = 0.1,
                WeightDecay = 0
            }; 

            if (commands.Length > 1)
            {
                switch (commands[1].ToLower())
                {
                    case "weights":
                        if (commands.Length <= 2)
                        {
                            break;
                        }
                        train = false;
                        result = TrainResult.Deserialize(commands[2]);
                        break;

                    case "train":
                        if (commands.Length > 2)
                            trainConfiguration = TrainConfiguration.Deserialize(commands[2]);
                        break;
                }
            }

            #region Inputs
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
            #endregion

            var inputLayer = new Layer(4, new IdentityActivation());
            var hiddenLayer = new Layer(8, new HyperbolicTanActivation());
            var outputLayer = new Layer(4, new SigmoidActivation());

            var net = new BackpropagationNet()
                .InputLayer(inputLayer)
                .HiddenLayers(hiddenLayer)
                .OutputLayer(outputLayer);
            net.BuildConnections();

            if (train)
            {
                #region Outputs
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
                #endregion

                var watch = new Stopwatch();

                TestCasesUtils.ReportStart(trainConfiguration.MaxEpochs, trainConfiguration.MinError,
                    trainConfiguration.LearnRate, trainConfiguration.Momentum, trainConfiguration.WeightDecay);
                watch.Start();
                net.Train(trainConfiguration, inputs, targetOutputs, out result);
                watch.Stop();

                TestCasesUtils.ReportEnd(watch, result.Epochs, result.Error);
                result.Serialize("count");

                if (commands.Contains("saveconfig"))
                {
                    trainConfiguration.Serialize("count");
                }
            }
            else
            {
                net.SetWeights(result.Weights);
            }

            var output = new List<double>();
            foreach (var input in inputs)
            {
                net.FeedForward(input);
                net.GetOutputs(output);
                ConsoleUtils.ShowVector(output, 4, 0, false);
                Thread.Sleep(500);
            }
        }
    }
}
