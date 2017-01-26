using System;
using System.Collections.Generic;
using System.Security.Permissions;

namespace NeuralNetwork.Nets
{
    public class Neuron
    {
        public bool IsBias;
        public readonly Layer Layer;
        public int Index;
        public double OutputValue;

        public double Gradient;
        public readonly List<Synapse> Synapses;

        public Neuron(Layer layer)
        {
            Layer = layer;
            Synapses = new List<Synapse>();
        }

        public void Connect(Neuron otherNeuron)
        {
            // Conecta da direita para a esquerda
            // output.Connect(hidden)
            // hidden.Connect(input)
            var synapse = new Synapse()
            {
                ConnectedNeuron = otherNeuron
            };
            Synapses.Add(synapse);
        }

        public void Connect(Layer layer)
        {
            foreach (var neuron in layer)
            {
                Connect(neuron);
            }
        }

        public void FeedForward(double? value = null)
        {
            if (IsBias)
                return;

            if (value.HasValue)
            {
                OutputValue = value.Value;
                return;
            }

            var sum = 0.0;

            // Soma as saidas do layer anterior (que são as entradas desse Neuron)
            // Inclui o bias do layer anterior
            foreach (var synapse in Synapses)
            {
                sum += synapse.Weight * synapse.ConnectedNeuron.OutputValue;
            }
            // Aplica a função de ativação
            OutputValue = Layer.ActivationFunction.Function(sum);
        }

        public void UpdateWeights(TrainConfiguration trainConfiguration)
        {
            var learnRate = trainConfiguration.LearnRate;
            var momentum = trainConfiguration.Momentum;
            var weightDecay = trainConfiguration.WeightDecay;

            foreach (var synapse in Synapses)
            {
                var otherNeuron = synapse.ConnectedNeuron;
                var oldDeltaWeight = synapse.DeltaWeight;

                // Entrada individual (saida do outro neuron), multiplicada pelo gradiente (desse neuron) e pela taxa de aprendizado
                var newDeltaWeight = learnRate * otherNeuron.OutputValue * Gradient;
                // Adiciona também uma fração do delta anterior do peso
                newDeltaWeight += momentum * oldDeltaWeight;
                newDeltaWeight -= weightDecay * synapse.Weight;

                synapse.DeltaWeight = newDeltaWeight;
                synapse.Weight += newDeltaWeight;
                //Console.WriteLine(Gradient);
            }
        }

        public bool IsConnectedTo(Neuron otherNeuron, out Synapse synapse)
        {
            foreach (var syn in Synapses)
            {
                if (syn.ConnectedNeuron == otherNeuron)
                {
                    synapse = syn;
                    return true;
                }
            }
            synapse = null;
            return false;
        }
    }
}