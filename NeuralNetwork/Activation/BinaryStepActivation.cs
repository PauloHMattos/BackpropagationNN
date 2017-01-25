using System;

namespace NeuralNetwork.Activation
{
    /// <summary>
    /// Representa o funcionamento de um perceptron
    /// </summary>
    public class BinaryStepActivation : IActivationFunction
    {
        public double Function(double x)
        {
            return (x < 0) ? 0 : 1;
        }

        public double Derivative(double x)
        {
            if (x != 0) return 0;
            throw new ArgumentException("Derivada não definida na função BinaryStepActivation para o valor x = " + x);
        }
    }
}