using System;

namespace NeuralNetwork.Activations
{
    public class SigmoidActivation : IActivationFunction
    {
        public double Function(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public double Derivative(double x)
        {
            var f = Function(x);
            return f*(1 - f);
        }
    }
}