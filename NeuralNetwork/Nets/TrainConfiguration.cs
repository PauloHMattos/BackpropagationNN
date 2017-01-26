namespace NeuralNetwork.Nets
{
    public struct TrainConfiguration
    {
        public int MaxEpochs;
        public double MinError;
        public double LearnRate;
        public double Momentum;
        public double WeightDecay;
    }
}