using System;

namespace NeuralNetwork.Activation
{
    public class GaussianActivation : IActivationFunction
    {
        public double Function(double x)
        {
            return Math.Exp(-Math.Pow(x, 2));
        }

        public double Derivative(double x)
        {
            return -2 * x * Function(x);
        }
    }
}