using System;

namespace NeuralNetwork.Activations
{
    public class HyperbolicTanActivation : IActivationFunction
    {
        public double Function(double x)
        {
            return Math.Tanh(x);
        }

        public double Derivative(double x)
        {
            return 1 - Math.Pow(Function(x), 2);
        }
    }
}