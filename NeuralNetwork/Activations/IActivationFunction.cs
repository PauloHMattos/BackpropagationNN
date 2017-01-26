namespace NeuralNetwork.Activations
{
    public interface IActivationFunction
    {
        double Function(double x);
        double Derivative(double x);
    }
}