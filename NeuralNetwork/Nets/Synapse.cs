namespace NeuralNetwork.Nets
{
    public class Synapse
    {
        public double Weight;
        public double DeltaWeight;
        public Neuron ConnectedNeuron;


        public Synapse()
        {
            Weight = RandomUtils.Interpolate(-0.01, 0.01);
        }
    }
}