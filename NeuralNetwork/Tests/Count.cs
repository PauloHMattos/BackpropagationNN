using System;
using System.Diagnostics;
using System.Threading;
using NeuralNetwork.Activation;
using NeuralNetwork.Nets;

namespace NeuralNetwork.Tests
{
    public static partial class TestCases
    {
        private static void NeuralNetData(int numInput, int numHidden, int numOutput, IActivationFunction hiddenActivationFunction, IActivationFunction outputActivationFunction)
        {
            Console.WriteLine("\nRede neural com " + numInput + "-entradas, " +
                              numHidden + "-hidden, " + numOutput + "-output");
            Console.WriteLine("   -Ativação Hidden-Input: " + hiddenActivationFunction.GetType().Name);
            Console.WriteLine("   -Ativação Hidden-Output: " + outputActivationFunction.GetType().Name);
        }

        private static void ReportStart(int maxEpochs, double minSquaredError, double learnRate, double momentum, double weightDecay)
        {
            Console.WriteLine("\nTreino iniciado");
            Console.WriteLine("   -Erro mínimo (MSE): " + minSquaredError);
            Console.WriteLine("   -Máximo de épocas: " + maxEpochs);
            Console.WriteLine("   -Taxa de aprendizado: " + learnRate);
            Console.WriteLine("   -Momento: " + momentum);
            Console.WriteLine("   -Decaimento: " + weightDecay);
        }

        private static void ReportEnd(Stopwatch watch, int epoch, double mse)
        {
            Console.WriteLine("\nTreino finalizado");
            Console.WriteLine("   -Duração: " + watch.Elapsed);
            Console.WriteLine("   -Épocas de treino: " + epoch);
            Console.WriteLine("   -Último MSE: " + mse);
            Console.WriteLine("\n");
        }

        public static void Count()
        {
            var trainSequence = new double[16][];
            trainSequence[0]  = new double[] { 0, 0, 0, 0, /**/ 0, 0, 0, 1 };
            trainSequence[1]  = new double[] { 0, 0, 0, 1, /**/ 0, 0, 1, 0 };
            trainSequence[2]  = new double[] { 0, 0, 1, 0, /**/ 0, 0, 1, 1 };
            trainSequence[3]  = new double[] { 0, 0, 1, 1, /**/ 0, 1, 0, 0 };
            trainSequence[4]  = new double[] { 0, 1, 0, 0, /**/ 0, 1, 0, 1 };
            trainSequence[5]  = new double[] { 0, 1, 0, 1, /**/ 0, 1, 1, 0 };
            trainSequence[6]  = new double[] { 0, 1, 1, 0, /**/ 0, 1, 1, 1 };
            trainSequence[7]  = new double[] { 0, 1, 1, 1, /**/ 1, 0, 0, 0 };

            trainSequence[8]  = new double[] { 1, 0, 0, 0, /**/ 1, 0, 0, 1 };
            trainSequence[9]  = new double[] { 1, 0, 0, 1, /**/ 1, 0, 1, 0 };
            trainSequence[10] = new double[] { 1, 0, 1, 0, /**/ 1, 0, 1, 1 };
            trainSequence[11] = new double[] { 1, 0, 1, 1, /**/ 1, 1, 0, 0 };
            trainSequence[12] = new double[] { 1, 1, 0, 0, /**/ 1, 1, 0, 1 };
            trainSequence[13] = new double[] { 1, 1, 0, 1, /**/ 1, 1, 1, 0 };
            trainSequence[14] = new double[] { 1, 1, 1, 0, /**/ 1, 1, 1, 1 };
            trainSequence[15] = new double[] { 1, 1, 1, 1, /**/ 0, 0, 0, 0 };


            const int numInput = 4;
            const int numHidden = 8;
            const int numOutput = 4;


            var hiddenActivation = new HyperbolicTanActivation();
            var outputActivation = new SigmoidActivation();
            NeuralNetData(numInput, numHidden, numOutput, hiddenActivation, outputActivation);
            var nn = new BackpropNeuralNet(numInput, numHidden, numOutput, hiddenActivation, outputActivation);
            nn.InitializeWeights();

            const int maxEpochs = 1000;
            const double minSquaredError = 0.0005;
            const double learnRate = 0.6;
            const double momentum = 0.1;
            const int weightDecay = 0;

            int epoch;
            double mse;
            
            var watch = new Stopwatch();

            ReportStart(maxEpochs, minSquaredError, learnRate, momentum, weightDecay);

            watch.Start();
            nn.Train(trainSequence, maxEpochs, minSquaredError, learnRate,
                momentum, weightDecay, out mse, out epoch);
            watch.Stop();

            ReportEnd(watch, epoch, mse);

            var value = new double[] { 0, 0, 0, 0 };
            for (var i = 0; i < trainSequence.Length; i++)
            {
                Utils.ShowVector(value, 4, 0, false);
                nn.ComputeOutputs(value);
                value = nn.GetOutputs();
                Thread.Sleep(500);
            }
            
        }
    }
}
