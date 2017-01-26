using System.Collections;
using System.Collections.Generic;
using NeuralNetwork.Activation;

namespace NeuralNetwork.Nets
{
    public class Layer : IEnumerable<Neuron>
    {
        public readonly IActivationFunction ActivationFunction;        
        public readonly List<Neuron> Neurons;

        public Neuron this[int id] => Neurons[id];

        public Layer(int numNeurons, IActivationFunction activationFunction)
        {
            ActivationFunction = activationFunction;
            Neurons = new List<Neuron>(numNeurons);
            for (var i = 0; i < numNeurons; i++)
            {
                var neuron = new Neuron(this);
                Neurons.Add(neuron);
            }
            Neurons.Add(new Neuron(this)
            {
                OutputValue = 1.0,
                IsBias = true
            });
        }

        public void Connect(Layer previousLayer)
        {
            foreach (var neuron in Neurons)
            {
                neuron.Connect(previousLayer);
            }
        }

        public IEnumerator<Neuron> GetEnumerator()
        {
            return Neurons.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void UpdateWeights(TrainConfiguration trainConfiguration)
        {
            foreach (var neuron in Neurons)
            {
                neuron.UpdateWeights(trainConfiguration);
            }
        }
    }
}