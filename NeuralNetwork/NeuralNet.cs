using System;
using NeuralNetwork.Activation;

namespace NeuralNetwork
{
    public class NeuralNet
    {
        private static Random _rnd;

        /// <summary>
        /// Número de nós (unidades [?], neurônios) na camada de entrada (e.i número de entradas, ou variáveis)
        /// </summary>
        private readonly int _numInput;
        /// <summary>
        /// Número de nós na camada "oculta"
        /// </summary>
        private readonly int _numHidden;
        /// <summary>
        /// Número de nós na camada de saída (e.i número de saídas)
        /// </summary>
        private readonly int _numOutput;


        /// <summary>
        /// Valores da camada de entrada
        /// </summary>
        private readonly double[] _inputs;

        /// <summary>
        /// Pesos entre os nós da camada de entrada e os nós da camada "oculta"
        /// </summary>
        private readonly double[][] _inputHiddenWeights;
        /// <summary>
        /// Vieses da camada "oculta"
        /// </summary>
        private readonly double[] _hiddenBiases;
        /// <summary>
        /// Saídas da camada "oculta"
        /// </summary>
        private readonly double[] _hiddenOutputs;

        /// <summary>
        /// Pesos entre os nós da camada "oculta" e os nós da camada de saída
        /// </summary>
        private readonly double[][] _hiddenOutputWeights; // hidden-output
        /// <summary>
        /// Vieses da camada de saída
        /// </summary>
        private readonly double[] _outputBiases;
        /// <summary>
        /// Valores da camada de saída
        /// </summary>
        private readonly double[] _outputs;



        private readonly IActivationFunction _hiddenOutputActivation;
        private readonly IActivationFunction _outputActivation;

        /// <summary>
        /// Vetores e matrizes especificas do método de backpropagation
        /// 
        /// Poderiam ser locais dos métodos 'UpdateWeights' e 'Train'
        /// mas optei deixa-las globais pra evitar a alocação constante de memória
        /// </summary>
        #region Backpropagation

        // UpdateWeights
        private readonly double[] _outputGradients;
        private readonly double[] _hiddenGradients;

        // Train
        private readonly double[][] _inputHiddenPreviousWeightsDelta;  // for momentum with back-propagation
        private readonly double[] _hiddenPreviousBiasesDelta;
        private readonly double[][] _hiddenOutputPreviousWeightsDelta;
        private readonly double[] _outputPreviousBiasesDelta;

        #endregion

        public NeuralNet(int numInput, int numHidden, int numOutput, IActivationFunction hiddenOutputActivation, IActivationFunction outputActivation)
        {
            // Usado em InitializeWeights() and Shuffle()
            // Seed definida em 0 torna deterministico.
            // Com isso o mesmo conjunto de entradas vai sempre produzir exatamente as
            // mesmas saída no mesmo número de passos.
            // A seed poderia ser qualquer outra. Foi escolhido 0 por conveniencia
            _rnd = new Random(0);

            _numInput = numInput;
            _numHidden = numHidden;
            _numOutput = numOutput;
            _hiddenOutputActivation = hiddenOutputActivation;
            _outputActivation = outputActivation;

            _inputs = new double[numInput];

            _inputHiddenWeights = BuildMatrix(numInput, numHidden);
            _hiddenBiases = new double[numHidden];
            _hiddenOutputs = new double[numHidden];

            _hiddenOutputWeights = BuildMatrix(numHidden, numOutput);
            _outputBiases = new double[numOutput];

            _outputs = new double[numOutput];
            
            _hiddenGradients = new double[numHidden];
            _outputGradients = new double[numOutput];

            _inputHiddenPreviousWeightsDelta = BuildMatrix(numInput, numHidden);
            _hiddenPreviousBiasesDelta = new double[numHidden];
            _hiddenOutputPreviousWeightsDelta = BuildMatrix(numHidden, numOutput);
            _outputPreviousBiasesDelta = new double[numOutput];
        }

        /// <summary>
        /// Gera uma matriz com 'rows' linas e 'cols' colunas
        /// </summary>
        /// <param name="rows">Número de linhas na matriz</param>
        /// <param name="cols">Número de colunas na matriz</param>
        /// <returns>Matriz cols x rows</returns>
        private static double[][] BuildMatrix(int rows, int cols)
        {
            var result = new double[rows][];
            for (var r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            return result;
        }


        /// <summary>
        /// Retorna o conjunto atual de pesos, presumivelmente após o treinamento
        /// </summary>
        /// <returns>Conjunto atual de pesos</returns>
        public double[] GetWeights()
        {
            var numWeights = (_numInput * _numHidden) + (_numHidden * _numOutput) + _numHidden + _numOutput;
            var result = new double[numWeights];
            var k = 0;  // Apontador pra um valor em 'result[]'
            for (var i = 0; i < _inputHiddenWeights.Length; ++i)
                for (var j = 0; j < _inputHiddenWeights[0].Length; ++j)
                    result[k++] = _inputHiddenWeights[i][j];
            for (var i = 0; i < _hiddenBiases.Length; ++i)
                result[k++] = _hiddenBiases[i];
            for (var i = 0; i < _hiddenOutputWeights.Length; ++i)
                for (var j = 0; j < _hiddenOutputWeights[0].Length; ++j)
                    result[k++] = _hiddenOutputWeights[i][j];
            for (var i = 0; i < _outputBiases.Length; ++i)
                result[k++] = _outputBiases[i];
            return result;
        }

        //TODO - Melhorar esse sumário. Tá ridiculo
        /// <summary>
        /// Copia os pesos e vieses (?) em 'weights[]' para as arrays i-h weights, i-h biases, h-o weights, h-o biases
        /// </summary>
        /// <param name="weights"></param>
        public void SetWeights(double[] weights)
        {
            var numWeights = (_numInput * _numHidden) + (_numHidden * _numOutput) + _numHidden + _numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array length: ");

            var k = 0; // Apontador pra um valor em 'weights[]'

            for (var i = 0; i < _numInput; ++i)
                for (var j = 0; j < _numHidden; ++j)
                    _inputHiddenWeights[i][j] = weights[k++];
            for (var i = 0; i < _numHidden; ++i)
                _hiddenBiases[i] = weights[k++];
            for (var i = 0; i < _numHidden; ++i)
                for (var j = 0; j < _numOutput; ++j)
                    _hiddenOutputWeights[i][j] = weights[k++];
            for (var i = 0; i < _numOutput; ++i)
                _outputBiases[i] = weights[k++];
        }

        /// <summary>
        /// Inicializa os pesos e vieses (?) como pequenos valores aleatórios
        /// 
        /// Ele precisa receber valores inicias para começar o processo de treinamento
        /// </summary>
        public void InitializeWeights()
        {
            var numWeights = (_numInput * _numHidden) + (_numHidden * _numOutput) + _numHidden + _numOutput;
            var initialWeights = new double[numWeights];
            const double lowest = -0.01;
            const double highest = 0.01;
            for (var i = 0; i < initialWeights.Length; ++i)
            {
                // Interpola o peso linearmente entre 'lowest' e 'highest' baseado num valor randomico
                initialWeights[i] = (highest - lowest) * _rnd.NextDouble() + lowest;
            }
            SetWeights(initialWeights);
        }

        public void ComputeOutputs(double[] inputs)
        {
            if (inputs.Length != _numInput)
                throw new Exception("O tamanho do array de entradas (" + nameof(inputs) + ") não é compativel com o esperado");
            
            Array.Copy(inputs, _inputs, _numInput);
            
            // Calcula o somatório do produto entre as entradas
            // E os pesos do hidden layer
            for (var j = 0; j < _numHidden; j++)
            {
                var hiddenSum = 0.0;
                for (var i = 0; i < _numInput; i++)
                {
                    hiddenSum += _inputs[i] * _inputHiddenWeights[i][j];
                }
                // Soma o bias
                hiddenSum += _hiddenBiases[j];

                // Aplica a função de ativação
                _hiddenOutputs[j] = _hiddenOutputActivation.Function(hiddenSum);
            }

            // Calcula o somatório do produto entre a saída do hidden layer
            // E os pesos do layer de saída
            for (var j = 0; j < _numOutput; j++)
            {
                var outputSum = 0.0;
                for (var i = 0; i < _numHidden; i++)
                {
                    outputSum += _hiddenOutputs[i] * _hiddenOutputWeights[i][j];
                }
                // Soma o bias
                outputSum += _outputBiases[j];

                // Aplica a função de ativação
                _outputs[j] = _outputActivation.Function(outputSum);
            }
        }

        public double[] GetOutputs()
        {
            var result = new double[_numOutput];
            Array.Copy(_outputs, result, _numOutput);
            return result;
        }

        /// <summary>
        /// Atualiza os pesos e biases usando back-propagation, com valores-alvo,
        /// eta (taxa de aprendizado), alfa (momento) e custo (decaimento dos pesos).
        /// 
        /// ATENÇÃO: Assume que 'SetWeights' e 'ComputeOutputs' já foram chamados,
        /// portanto todos os vetores e matrizes internos foram populados,
        /// com valores diferentes de 0.0.
        /// </summary>
        /// <param name="targetValues">Valores-alvo</param>
        /// <param name="learnRate">Taxa de aprendizado (eta)</param>
        /// <param name="momentum">Momento (alfa)</param>
        /// <param name="weightDecay">Decaimento</param>
        /// TODO - Renomear para Backpropagate?
        private void UpdateWeights(double[] targetValues, double learnRate, double momentum, double weightDecay)
        {
            if (targetValues.Length != _numOutput)
                throw new Exception("targetValues não tem o comprimento esperado: " + targetValues.Length + "!=" + _numOutput);
            
            // Calcula os gradientes do output layer
            for (var i = 0; i < _numOutput; i++)
            {
                var derivative = _outputActivation.Derivative(_outputs[i]);
                var delta = targetValues[i] - _outputs[i];
                _outputGradients[i] = derivative * delta;
            }
            
            // Calcula os gradientes do hidden layer
            for (var i = 0; i < _numHidden; i++)
            {
                var delta = 0.0;
                // Cada "hidden delta" é o somatório de '_numOutput' termos
                // do produto entre os gradientes do output layer e os pesos da interface
                // hidden-output
                for (var j = 0; j < _numOutput; j++)
                {
                    delta += _outputGradients[j] * _hiddenOutputWeights[i][j];
                }
                var derivative = _hiddenOutputActivation.Derivative(_hiddenOutputs[i]);
                _hiddenGradients[i] = derivative * delta;
            }

            // Calcula os novos pesos do hidden layer
            // Os Gradientes devem ser computados da direita para a esquerda (Primeiro os output e depois os hidden)
            // mas os pesos podem ser atualizados em qualquer ordem (Aqui foi feita da esquerda para a direita)
            for (var i = 0; i < _numInput; i++)
            {
                for (var j = 0; j < _numHidden; j++)
                {
                    var newDelta = learnRate * _hiddenGradients[j] * _inputs[i];
                        newDelta += momentum * _inputHiddenPreviousWeightsDelta[i][j];
                        newDelta -= (weightDecay * _inputHiddenWeights[i][j]);

                    _inputHiddenWeights[i][j] += newDelta;
                    _inputHiddenPreviousWeightsDelta[i][j] = newDelta;
                }
            }

            // Calcula os novos bias do hidden layer
            for (var i = 0; i < _numHidden; i++)
            {
                var newDelta = learnRate * _hiddenGradients[i]; // * 1.0; -> Saída constante dos bias 
                    newDelta += momentum * _hiddenPreviousBiasesDelta[i];
                    newDelta -= weightDecay * _hiddenBiases[i];

                _hiddenBiases[i] += newDelta;
                _hiddenPreviousBiasesDelta[i] = newDelta; 
            }

            // Calcula os novos pesos da interface hidden-output
            for (var i = 0; i < _numHidden; i++)
            {
                for (var j = 0; j < _numOutput; j++)
                {
                    var newDelta = learnRate * _outputGradients[j] * _hiddenOutputs[i];
                        newDelta += momentum * _hiddenOutputPreviousWeightsDelta[i][j];
                        newDelta -= (weightDecay * _hiddenOutputWeights[i][j]);

                    _hiddenOutputWeights[i][j] += newDelta;
                    _hiddenOutputPreviousWeightsDelta[i][j] = newDelta;
                }
            }

            // Calcula os novos bias do output layer
            for (var i = 0; i < _outputBiases.Length; ++i)
            {
                var newDelta = learnRate * _outputGradients[i]; // * 1.0; -> Saída constante dos bias 
                    newDelta += momentum * _outputPreviousBiasesDelta[i];
                    newDelta -= (weightDecay * _outputBiases[i]);

                _outputBiases[i] += newDelta;
                _outputPreviousBiasesDelta[i] = newDelta;
            }
        }

        /// <summary>
        /// Treina uma rede neural com backpropagation usando
        /// taxa de aprendizado e momento
        /// </summary>
        /// <param name="trainData">Dados de treinamento contendo as entradas e as saídas esperadas</param>
        /// <param name="maxEpochs">Número máximo de "épocas" de treinamento</param>
        /// <param name="minSquaredError">Erro mínimo para interromper o treinamento</param>
        /// <param name="learnRate"></param>
        /// <param name="momentum">Multiplicador do delta da epoch anterior</param>
        /// <param name="weightDecay">Reduz a magnitude de um peso com o tempo a menos que esse sea constantemente incrementado</param>
        /// <param name="mse"></param>
        /// <param name="epoch"></param>
        public void Train(double[][] trainData, int maxEpochs, double minSquaredError, double learnRate, double momentum,
            double weightDecay, out double mse, out int epoch)
        {
            epoch = 0;
            var inputValues = new double[_numInput];
            var targetValues = new double[_numOutput];

            // Cria uma sequencia direta que vai ser embaralhada posteriormente
            var sequence = new int[trainData.Length];
            for (var i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            mse = double.MaxValue;
            while (epoch < maxEpochs)
            {
                mse = MeanSquaredError(trainData);
                if (mse < minSquaredError)
                    break;
                
                Shuffle(sequence); // A cada traino visita os dados numa ordem aleatória
                foreach (var i in sequence)
                {
                    Array.Copy(trainData[i], inputValues, _numInput);
                    Array.Copy(trainData[i], _numInput, targetValues, 0, _numOutput);
                    ComputeOutputs(inputValues);
                    UpdateWeights(targetValues, learnRate, momentum, weightDecay); // Encontra pesos melhores
                }
                epoch++;
            }
        }


        /// <summary>
        /// Calcula a porcentagem de resultados determinados corretamente usando a regra "vencedor leva tudo"
        /// 
        /// ATENÇÃO: Só funcion pra casos de classificação direta.
        /// </summary>
        /// <param name="testData">Conjunto de daos a ser testado</param>
        /// <returns>Porcentagem de acerto (0.0 ~ 1.0)</returns>
        public double Accuracy(double[][] testData)
        {
            var numCorrect = 0;
            var numWrong = 0;
            var inputValues = new double[_numInput];
            var targetedOutputs = new double[_numOutput];

            foreach (var data in testData)
            {
                Array.Copy(data, inputValues, _numInput);
                Array.Copy(data, _numInput, targetedOutputs, 0, _numOutput);

                ComputeOutputs(inputValues);
                var computedOutputs = GetOutputs();
                var maxIndex = MaxIndex(computedOutputs); // Qual das saídas tem o maior valor?

                if (targetedOutputs[maxIndex].Equals(1.0))
                    // Se a saída que calculamos for a certa (i.e 1.0 dos dados de treino)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            // TODO-Checar possivel divisão por 0
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }
        
        #region Helpers

        /// <summary>
        /// Embaralha os valores na array 'sequence[]'
        /// </summary>
        /// <param name="sequence">Array a sem embaralhada</param>
        private static void Shuffle(int[] sequence)
        {
            for (var i = 0; i < sequence.Length; ++i)
            {
                var r = _rnd.Next(i, sequence.Length);
                var tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }


        /// <summary>
        /// Calcula o erro quadrático médio por tupla de treino usando os pesos atuais
        /// É usado como condição de parada dos treinos
        /// </summary>
        /// <param name="trainData">Matriz com os dados de treino</param>
        /// <returns>Erro quadrático médio entre os valores calculados e os visados</returns>
        private double MeanSquaredError(double[][] trainData)
        {
            var sumSquaredError = 0.0;
            var inputValues = new double[_numInput]; // Primeiros n (numInput) elementos em trainData
            var targetedOutputs = new double[_numOutput]; // Ultimos n (numOutput) elementos em trainData

            // Passa por cada caso de treinamento e separa os valores 
            foreach (var data in trainData)
            {
                Array.Copy(data, inputValues, _numInput);
                Array.Copy(data, _numInput, targetedOutputs, 0, _numOutput); // Pega as saidas que estamos buscando

                ComputeOutputs(inputValues);
                var computedOutputs = GetOutputs(); // Computa as saidas usando os pesos atuais

                for (var i = 0; i < _numOutput; ++i)
                {
                    var err = targetedOutputs[i] - computedOutputs[i]; // Esperado - Encontrado = Erro
                    sumSquaredError += err * err; // Erro quadrático (torna o valor positivo, e é mais rápido que fazer ABS)...
                }
            }
            return sumSquaredError / trainData.Length;
        }

        
        /// <summary>
        /// Retorna o índice de maior valor do array vector
        /// </summary>
        /// <param name="vector"></param>
        /// <returns>Índice de maior valor do array 'vector[]'</returns>
        private static int MaxIndex(double[] vector)
        {
            var bigIndex = 0;
            var biggestVal = vector[0];
            for (var i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i]; bigIndex = i;
                }
            }
            return bigIndex;
        }

        #endregion
    }
}