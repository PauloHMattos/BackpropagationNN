using System;
using System.Collections.Generic;
using NeuralNetwork.Layers;
using NeuralNetwork.Neurons;
using NeuralNetwork.Training;

namespace NeuralNetwork.Networks
{
    public class BackpropagationNet
    {
        protected Layer _inputLayer;
        protected readonly List<Layer> _hiddenLayers;
        protected Layer _outputLayer;
        protected TrainConfiguration _trainConfiguration;


        private double _error;
        public BackpropagationNet()
        {
            _hiddenLayers = new List<Layer>();
        }
        
        public BackpropagationNet InputLayer(Layer inputLayer)
        {
            _inputLayer = inputLayer;
            return this;
        }

        public BackpropagationNet OutputLayer(Layer outputLayer)
        {
            _outputLayer = outputLayer;
            // Remove o bias, já que no output não precisa
            _outputLayer.Neurons.RemoveAt(_outputLayer.Neurons.Count - 1);
            return this;
        }

        public BackpropagationNet HiddenLayers(params Layer[] layers)
        {
            _hiddenLayers.AddRange(layers);
            return this;
        }

        public void BuildConnections()
        {
            var currentLayer = _outputLayer;
            for (var i = _hiddenLayers.Count - 1; i >= 0; i--)
            {
                var previousLayer = _hiddenLayers[i];
                currentLayer.Connect(previousLayer);
                currentLayer = previousLayer;
            }
            currentLayer.Connect(_inputLayer);
        }

        public void FeedForward(double[] inputValues)
        {
            if(_inputLayer.Neurons.Count - 1 != inputValues.Length)
                throw new Exception();

            for (var i = 0; i < inputValues.Length; i++)
            {
                _inputLayer[i].FeedForward(inputValues[i]);
            }

            foreach (var hiddenLayer in _hiddenLayers)
            {
                foreach (var neuron in hiddenLayer)
                {
                    neuron.FeedForward();
                }
            }
            
            foreach (var neuron in _outputLayer)
            {
                neuron.FeedForward();
            }
        }


        private void Backpropagate(double[] targetValues)
        {
            if (_outputLayer.Neurons.Count != targetValues.Length)
                throw new Exception();

            // MSE
            _error = 0.0;
            for (var i = 0; i < targetValues.Length; i++)
            {
                var neuron = _outputLayer[i];
                // Calcula o erro
                var delta = targetValues[i] - neuron.OutputValue;
                _error += delta * delta;

                // Calcula o gradiente
                neuron.Gradient = delta * neuron.Layer.ActivationFunction.Derivative(neuron.OutputValue);
            }
            _error /= targetValues.Length;
            
            // Calcula os gradientes dos layers ocultos
            for (var i = 0; i < _hiddenLayers.Count; i++)
            {
                var hiddenLayer = _hiddenLayers[i];
                var nextLayer = (i + 1 < _hiddenLayers.Count) ? _hiddenLayers[i + 1] : _outputLayer;

                foreach (var neuron in hiddenLayer)
                {
                    var sum = 0.0;
                    foreach (var nextLayerNeuron in nextLayer)
                    {
                        Synapse synapse;
                        if (nextLayerNeuron.IsConnectedTo(neuron, out synapse))
                        {
                            sum += synapse.Weight * nextLayerNeuron.Gradient;
                        }
                    }
                    neuron.Gradient = sum * neuron.Layer.ActivationFunction.Derivative(neuron.OutputValue);
                }
            }

            _inputLayer.UpdateWeights(_trainConfiguration);
            foreach (var hiddenLayer in _hiddenLayers)
            {
                hiddenLayer.UpdateWeights(_trainConfiguration);
            }
            _outputLayer.UpdateWeights(_trainConfiguration);
        }

        public void Train(TrainConfiguration trainConfiguration, List<double[]> inputs,
            List<double[]> targetValues, out TrainResult result)
        {
            if (inputs.Count != targetValues.Count)
                throw new Exception();

            _trainConfiguration = trainConfiguration;
            result = new TrainResult {Weights = new List<double>()};

            _error = double.MaxValue;
            while (result.Epochs < _trainConfiguration.MaxEpochs && _error > _trainConfiguration.MinError)
            {
                for (var i = 0; i < inputs.Count; i++)
                {
                    FeedForward(inputs[i]);
                    Backpropagate(targetValues[i]);
                }
                result.Epochs++;
            }

            result.Error = _error;
            GetWeights(result.Weights);
        }

        public void GetWeights(List<double> weigths)
        {
            weigths.Clear();
            foreach (var neuron in _outputLayer)
            {
                foreach (var synapse in neuron.Synapses)
                {
                    weigths.Add(synapse.Weight);
                }
            }
            foreach (var layer in _hiddenLayers)
            {
                foreach (var neuron in layer)
                {
                    foreach (var synapse in neuron.Synapses)
                    {
                        weigths.Add(synapse.Weight);
                    }
                }
            }
        }

        public void SetWeights(List<double> weigths)
        {
            var i = 0;
            foreach(var neuron in _outputLayer)
            {
                foreach (var synapse in neuron.Synapses)
                {
                    synapse.Weight = weigths[i];
                    i++;
                }
            }
            foreach (var layer in _hiddenLayers)
            {
                foreach (var neuron in layer)
                {
                    foreach (var synapse in neuron.Synapses)
                    {
                        synapse.Weight = weigths[i];
                        i++;
                    }
                }
            }
        }

        public void GetOutputs(List<double> output)
        {
            output.Clear();
            foreach (var neuron in _outputLayer)
            {
                output.Add(neuron.OutputValue);
            }
        }

        public double Accuracy(IEnumerable<double[]> testData, int numInput, int numOutput)
        {
            var numCorrect = 0;
            var numWrong = 0;
            var inputValues = new double[numInput];
            var targetedOutputs = new double[numOutput];
            foreach (var data in testData)
            {
                Array.Copy(data, inputValues, numInput);
                Array.Copy(data, numInput, targetedOutputs, 0, numOutput);

                FeedForward(inputValues);
                var outputs = new List<double>();
                GetOutputs(outputs);

                var maxIndex = MaxIndex(outputs); // Qual das saídas tem o maior valor?

                if (targetedOutputs[maxIndex].Equals(1.0))
                    // Se a saída que calculamos for a certa (i.e 1.0 dos dados de treino)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            // TODO-Checar possivel divisão por 0
            return (double)numCorrect / (numCorrect + numWrong);
        }

        private static int MaxIndex(IReadOnlyList<double> vector)
        {
            var bigIndex = 0;
            var biggestVal = vector[0];
            for (var i = 0; i < vector.Count; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i]; bigIndex = i;
                }
            }
            return bigIndex;
        }
    }
}