using System;

namespace NeuralNetwork.Activations
{
    public class SinusoidActivation : IActivationFunction
    {
        public double Function(double x)
        {
            return Math.Sin(x);
        }

        public double Derivative(double x)
        {
            return Math.Cos(x);
        }
    }
}