using System;

namespace NeuralNetwork.Activations
{
    public class SoftSignActivation : IActivationFunction
    {
        public double Function(double x)
        {
            return x / (1 + Math.Abs(x));
        }

        public double Derivative(double x)
        {
            return 1 / Math.Pow(1 + Math.Abs(x), 2);
        }
    }
}