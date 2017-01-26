using System;
using System.Diagnostics;
using NeuralNetwork.Activations;

namespace NeuralNetwork.Utils
{
    public static class TestCasesUtils
    {
        public static void NeuralNetData(int numInput, int numHidden, int numOutput,
            IActivationFunction hiddenActivationFunction, IActivationFunction outputActivationFunction)
        {
            Console.WriteLine("\nRede neural com " + numInput + "-entradas, " +
                              numHidden + "-hidden, " + numOutput + "-output");
            Console.WriteLine("   -Ativação Hidden-Input: " + hiddenActivationFunction.GetType().Name);
            Console.WriteLine("   -Ativação Hidden-Output: " + outputActivationFunction.GetType().Name);
        }

        public static void ReportStart(int maxEpochs, double minSquaredError, double learnRate, double momentum,
            double weightDecay)
        {
            Console.WriteLine("\nTreino iniciado");
            Console.WriteLine("   -Erro mínimo (MSE): " + minSquaredError);
            Console.WriteLine("   -Máximo de épocas: " + maxEpochs);
            Console.WriteLine("   -Taxa de aprendizado: " + learnRate);
            Console.WriteLine("   -Momento: " + momentum);
            Console.WriteLine("   -Decaimento: " + weightDecay);
        }

        public static void ReportEnd(Stopwatch watch, int epoch, double mse)
        {
            Console.WriteLine("\nTreino finalizado");
            Console.WriteLine("   -Duração: " + watch.Elapsed);
            Console.WriteLine("   -Épocas de treino: " + epoch);
            Console.WriteLine("   -Último MSE: " + mse);
            Console.WriteLine("\n");
        }
    }
}
