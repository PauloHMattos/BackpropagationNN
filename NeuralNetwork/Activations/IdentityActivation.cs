namespace NeuralNetwork.Activations
{
    public class IdentityActivation : IActivationFunction
    {
        public double Function(double x)
        {
            return x;
        }

        public double Derivative(double x)
        {
            return 1;
        }
    }
}