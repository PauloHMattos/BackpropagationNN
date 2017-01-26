namespace NeuralNetwork.Activations
{
    public interface ISumActivationFunction
    {
        double[] Function(double[] x);
        double[] Derivative(double[] x);
    }
}