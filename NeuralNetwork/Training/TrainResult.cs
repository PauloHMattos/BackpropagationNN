using System.Collections.Generic;

namespace NeuralNetwork.Training
{
    public struct TrainResult
    {
        public int Epochs;
        public double Error;
        public List<double> Weights;
    }
}