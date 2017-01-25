using System;

namespace NeuralNetwork.Activation
{
    public class ArcTanActivation : IActivationFunction
    {
        public double Function(double x)
        {
            return Math.Atan(x);
        }

        public double Derivative(double x)
        {
            return 1 / (Math.Pow(x, 2) + 1);
        }
    }
}