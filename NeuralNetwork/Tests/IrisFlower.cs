using System;
using System.Diagnostics;
using NeuralNetwork.Activation;

namespace NeuralNetwork.Tests
{
    public static partial class TestCases
    {
        public static void IrisFlower()
        {
            #region Data

            var allData = new double[150][];
            allData[0] = new[] { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 }; // sepal length, width, petal length, width
            allData[1] = new[] { 4.9, 3.0, 1.4, 0.2, 0, 0, 1 }; // Iris setosa = 0 0 1
            allData[2] = new[] { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 }; // Iris versicolor = 0 1 0
            allData[3] = new[] { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 }; // Iris virginica = 1 0 0
            allData[4] = new[] { 5.0, 3.6, 1.4, 0.2, 0, 0, 1 };
            allData[5] = new[] { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
            allData[6] = new[] { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
            allData[7] = new[] { 5.0, 3.4, 1.5, 0.2, 0, 0, 1 };
            allData[8] = new[] { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
            allData[9] = new[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };

            allData[10] = new[] { 5.4, 3.7, 1.5, 0.2, 0, 0, 1 };
            allData[11] = new[] { 4.8, 3.4, 1.6, 0.2, 0, 0, 1 };
            allData[12] = new[] { 4.8, 3.0, 1.4, 0.1, 0, 0, 1 };
            allData[13] = new[] { 4.3, 3.0, 1.1, 0.1, 0, 0, 1 };
            allData[14] = new[] { 5.8, 4.0, 1.2, 0.2, 0, 0, 1 };
            allData[15] = new[] { 5.7, 4.4, 1.5, 0.4, 0, 0, 1 };
            allData[16] = new[] { 5.4, 3.9, 1.3, 0.4, 0, 0, 1 };
            allData[17] = new[] { 5.1, 3.5, 1.4, 0.3, 0, 0, 1 };
            allData[18] = new[] { 5.7, 3.8, 1.7, 0.3, 0, 0, 1 };
            allData[19] = new[] { 5.1, 3.8, 1.5, 0.3, 0, 0, 1 };

            allData[20] = new[] { 5.4, 3.4, 1.7, 0.2, 0, 0, 1 };
            allData[21] = new[] { 5.1, 3.7, 1.5, 0.4, 0, 0, 1 };
            allData[22] = new[] { 4.6, 3.6, 1.0, 0.2, 0, 0, 1 };
            allData[23] = new[] { 5.1, 3.3, 1.7, 0.5, 0, 0, 1 };
            allData[24] = new[] { 4.8, 3.4, 1.9, 0.2, 0, 0, 1 };
            allData[25] = new[] { 5.0, 3.0, 1.6, 0.2, 0, 0, 1 };
            allData[26] = new[] { 5.0, 3.4, 1.6, 0.4, 0, 0, 1 };
            allData[27] = new[] { 5.2, 3.5, 1.5, 0.2, 0, 0, 1 };
            allData[28] = new[] { 5.2, 3.4, 1.4, 0.2, 0, 0, 1 };
            allData[29] = new[] { 4.7, 3.2, 1.6, 0.2, 0, 0, 1 };

            allData[30] = new[] { 4.8, 3.1, 1.6, 0.2, 0, 0, 1 };
            allData[31] = new[] { 5.4, 3.4, 1.5, 0.4, 0, 0, 1 };
            allData[32] = new[] { 5.2, 4.1, 1.5, 0.1, 0, 0, 1 };
            allData[33] = new[] { 5.5, 4.2, 1.4, 0.2, 0, 0, 1 };
            allData[34] = new[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            allData[35] = new[] { 5.0, 3.2, 1.2, 0.2, 0, 0, 1 };
            allData[36] = new[] { 5.5, 3.5, 1.3, 0.2, 0, 0, 1 };
            allData[37] = new[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            allData[38] = new[] { 4.4, 3.0, 1.3, 0.2, 0, 0, 1 };
            allData[39] = new[] { 5.1, 3.4, 1.5, 0.2, 0, 0, 1 };

            allData[40] = new[] { 5.0, 3.5, 1.3, 0.3, 0, 0, 1 };
            allData[41] = new[] { 4.5, 2.3, 1.3, 0.3, 0, 0, 1 };
            allData[42] = new[] { 4.4, 3.2, 1.3, 0.2, 0, 0, 1 };
            allData[43] = new[] { 5.0, 3.5, 1.6, 0.6, 0, 0, 1 };
            allData[44] = new[] { 5.1, 3.8, 1.9, 0.4, 0, 0, 1 };
            allData[45] = new[] { 4.8, 3.0, 1.4, 0.3, 0, 0, 1 };
            allData[46] = new[] { 5.1, 3.8, 1.6, 0.2, 0, 0, 1 };
            allData[47] = new[] { 4.6, 3.2, 1.4, 0.2, 0, 0, 1 };
            allData[48] = new[] { 5.3, 3.7, 1.5, 0.2, 0, 0, 1 };
            allData[49] = new[] { 5.0, 3.3, 1.4, 0.2, 0, 0, 1 };

            allData[50] = new[] { 7.0, 3.2, 4.7, 1.4, 0, 1, 0 };
            allData[51] = new[] { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
            allData[52] = new[] { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
            allData[53] = new[] { 5.5, 2.3, 4.0, 1.3, 0, 1, 0 };
            allData[54] = new[] { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
            allData[55] = new[] { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };
            allData[56] = new[] { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
            allData[57] = new[] { 4.9, 2.4, 3.3, 1.0, 0, 1, 0 };
            allData[58] = new[] { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
            allData[59] = new[] { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };

            allData[60] = new[] { 5.0, 2.0, 3.5, 1.0, 0, 1, 0 };
            allData[61] = new[] { 5.9, 3.0, 4.2, 1.5, 0, 1, 0 };
            allData[62] = new[] { 6.0, 2.2, 4.0, 1.0, 0, 1, 0 };
            allData[63] = new[] { 6.1, 2.9, 4.7, 1.4, 0, 1, 0 };
            allData[64] = new[] { 5.6, 2.9, 3.6, 1.3, 0, 1, 0 };
            allData[65] = new[] { 6.7, 3.1, 4.4, 1.4, 0, 1, 0 };
            allData[66] = new[] { 5.6, 3.0, 4.5, 1.5, 0, 1, 0 };
            allData[67] = new[] { 5.8, 2.7, 4.1, 1.0, 0, 1, 0 };
            allData[68] = new[] { 6.2, 2.2, 4.5, 1.5, 0, 1, 0 };
            allData[69] = new[] { 5.6, 2.5, 3.9, 1.1, 0, 1, 0 };

            allData[70] = new[] { 5.9, 3.2, 4.8, 1.8, 0, 1, 0 };
            allData[71] = new[] { 6.1, 2.8, 4.0, 1.3, 0, 1, 0 };
            allData[72] = new[] { 6.3, 2.5, 4.9, 1.5, 0, 1, 0 };
            allData[73] = new[] { 6.1, 2.8, 4.7, 1.2, 0, 1, 0 };
            allData[74] = new[] { 6.4, 2.9, 4.3, 1.3, 0, 1, 0 };
            allData[75] = new[] { 6.6, 3.0, 4.4, 1.4, 0, 1, 0 };
            allData[76] = new[] { 6.8, 2.8, 4.8, 1.4, 0, 1, 0 };
            allData[77] = new[] { 6.7, 3.0, 5.0, 1.7, 0, 1, 0 };
            allData[78] = new[] { 6.0, 2.9, 4.5, 1.5, 0, 1, 0 };
            allData[79] = new[] { 5.7, 2.6, 3.5, 1.0, 0, 1, 0 };

            allData[80] = new[] { 5.5, 2.4, 3.8, 1.1, 0, 1, 0 };
            allData[81] = new[] { 5.5, 2.4, 3.7, 1.0, 0, 1, 0 };
            allData[82] = new[] { 5.8, 2.7, 3.9, 1.2, 0, 1, 0 };
            allData[83] = new[] { 6.0, 2.7, 5.1, 1.6, 0, 1, 0 };
            allData[84] = new[] { 5.4, 3.0, 4.5, 1.5, 0, 1, 0 };
            allData[85] = new[] { 6.0, 3.4, 4.5, 1.6, 0, 1, 0 };
            allData[86] = new[] { 6.7, 3.1, 4.7, 1.5, 0, 1, 0 };
            allData[87] = new[] { 6.3, 2.3, 4.4, 1.3, 0, 1, 0 };
            allData[88] = new[] { 5.6, 3.0, 4.1, 1.3, 0, 1, 0 };
            allData[89] = new[] { 5.5, 2.5, 4.0, 1.3, 0, 1, 0 };

            allData[90] = new[] { 5.5, 2.6, 4.4, 1.2, 0, 1, 0 };
            allData[91] = new[] { 6.1, 3.0, 4.6, 1.4, 0, 1, 0 };
            allData[92] = new[] { 5.8, 2.6, 4.0, 1.2, 0, 1, 0 };
            allData[93] = new[] { 5.0, 2.3, 3.3, 1.0, 0, 1, 0 };
            allData[94] = new[] { 5.6, 2.7, 4.2, 1.3, 0, 1, 0 };
            allData[95] = new[] { 5.7, 3.0, 4.2, 1.2, 0, 1, 0 };
            allData[96] = new[] { 5.7, 2.9, 4.2, 1.3, 0, 1, 0 };
            allData[97] = new[] { 6.2, 2.9, 4.3, 1.3, 0, 1, 0 };
            allData[98] = new[] { 5.1, 2.5, 3.0, 1.1, 0, 1, 0 };
            allData[99] = new[] { 5.7, 2.8, 4.1, 1.3, 0, 1, 0 };

            allData[100] = new[] { 6.3, 3.3, 6.0, 2.5, 1, 0, 0 };
            allData[101] = new[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
            allData[102] = new[] { 7.1, 3.0, 5.9, 2.1, 1, 0, 0 };
            allData[103] = new[] { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
            allData[104] = new[] { 6.5, 3.0, 5.8, 2.2, 1, 0, 0 };
            allData[105] = new[] { 7.6, 3.0, 6.6, 2.1, 1, 0, 0 };
            allData[106] = new[] { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
            allData[107] = new[] { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };
            allData[108] = new[] { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };
            allData[109] = new[] { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };

            allData[110] = new[] { 6.5, 3.2, 5.1, 2.0, 1, 0, 0 };
            allData[111] = new[] { 6.4, 2.7, 5.3, 1.9, 1, 0, 0 };
            allData[112] = new[] { 6.8, 3.0, 5.5, 2.1, 1, 0, 0 };
            allData[113] = new[] { 5.7, 2.5, 5.0, 2.0, 1, 0, 0 };
            allData[114] = new[] { 5.8, 2.8, 5.1, 2.4, 1, 0, 0 };
            allData[115] = new[] { 6.4, 3.2, 5.3, 2.3, 1, 0, 0 };
            allData[116] = new[] { 6.5, 3.0, 5.5, 1.8, 1, 0, 0 };
            allData[117] = new[] { 7.7, 3.8, 6.7, 2.2, 1, 0, 0 };
            allData[118] = new[] { 7.7, 2.6, 6.9, 2.3, 1, 0, 0 };
            allData[119] = new[] { 6.0, 2.2, 5.0, 1.5, 1, 0, 0 };

            allData[120] = new[] { 6.9, 3.2, 5.7, 2.3, 1, 0, 0 };
            allData[121] = new[] { 5.6, 2.8, 4.9, 2.0, 1, 0, 0 };
            allData[122] = new[] { 7.7, 2.8, 6.7, 2.0, 1, 0, 0 };
            allData[123] = new[] { 6.3, 2.7, 4.9, 1.8, 1, 0, 0 };
            allData[124] = new[] { 6.7, 3.3, 5.7, 2.1, 1, 0, 0 };
            allData[125] = new[] { 7.2, 3.2, 6.0, 1.8, 1, 0, 0 };
            allData[126] = new[] { 6.2, 2.8, 4.8, 1.8, 1, 0, 0 };
            allData[127] = new[] { 6.1, 3.0, 4.9, 1.8, 1, 0, 0 };
            allData[128] = new[] { 6.4, 2.8, 5.6, 2.1, 1, 0, 0 };
            allData[129] = new[] { 7.2, 3.0, 5.8, 1.6, 1, 0, 0 };

            allData[130] = new[] { 7.4, 2.8, 6.1, 1.9, 1, 0, 0 };
            allData[131] = new[] { 7.9, 3.8, 6.4, 2.0, 1, 0, 0 };
            allData[132] = new[] { 6.4, 2.8, 5.6, 2.2, 1, 0, 0 };
            allData[133] = new[] { 6.3, 2.8, 5.1, 1.5, 1, 0, 0 };
            allData[134] = new[] { 6.1, 2.6, 5.6, 1.4, 1, 0, 0 };
            allData[135] = new[] { 7.7, 3.0, 6.1, 2.3, 1, 0, 0 };
            allData[136] = new[] { 6.3, 3.4, 5.6, 2.4, 1, 0, 0 };
            allData[137] = new[] { 6.4, 3.1, 5.5, 1.8, 1, 0, 0 };
            allData[138] = new[] { 6.0, 3.0, 4.8, 1.8, 1, 0, 0 };
            allData[139] = new[] { 6.9, 3.1, 5.4, 2.1, 1, 0, 0 };

            allData[140] = new[] { 6.7, 3.1, 5.6, 2.4, 1, 0, 0 };
            allData[141] = new[] { 6.9, 3.1, 5.1, 2.3, 1, 0, 0 };
            allData[142] = new[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
            allData[143] = new[] { 6.8, 3.2, 5.9, 2.3, 1, 0, 0 };
            allData[144] = new[] { 6.7, 3.3, 5.7, 2.5, 1, 0, 0 };
            allData[145] = new[] { 6.7, 3.0, 5.2, 2.3, 1, 0, 0 };
            allData[146] = new[] { 6.3, 2.5, 5.0, 1.9, 1, 0, 0 };
            allData[147] = new[] { 6.5, 3.0, 5.2, 2.0, 1, 0, 0 };
            allData[148] = new[] { 6.2, 3.4, 5.4, 2.3, 1, 0, 0 };
            allData[149] = new[] { 5.9, 3.0, 5.1, 1.8, 1, 0, 0 };

            #endregion

            Console.WriteLine("Separa todos os dados em 80% para trainamento e 20% para teste");
            double[][] trainData;
            double[][] testData;
            MakeTrainTest(allData, out trainData, out testData);

            // Normaliza os dados (valores > 1)
            // Torna o treino mais rápido, e produz uma precisão maior
            Normalize(trainData, 4);
            Normalize(testData, 4);

            const int numInput = 4;
            const int numHidden = 7;
            const int numOutput = 3;
            
            var hiddenActivation = new HyperbolicTanActivation();
            var outputActivation = new SigmoidActivation();
            NeuralNetData(numInput, numHidden, numOutput, hiddenActivation, outputActivation);
            var nn = new NeuralNet(numInput, numHidden, numOutput, hiddenActivation, outputActivation);
            nn.InitializeWeights();

            const int maxEpochs = 2000;
            const double minSquaredError = 0.01;
            const double learnRate = 0.5;
            const double momentum = 0.2;
            const double weightDecay = 0;

            int epoch;
            double mse;
            
            var watch = new Stopwatch();

            ReportStart(maxEpochs, minSquaredError, learnRate, momentum, weightDecay);

            watch.Start();
            nn.Train(trainData, maxEpochs, minSquaredError, learnRate,
                momentum, weightDecay, out mse, out epoch);
            watch.Stop();
            
            ReportEnd(watch, epoch, mse);
            
            Console.WriteLine("Precisão com os dados de treinamento: " + nn.Accuracy(trainData).ToString("F4"));
            Console.WriteLine("Precisão com os dados de teste: " + nn.Accuracy(testData).ToString("F4"));
        }

        private static void MakeTrainTest(double[][] allData, out double[][] trainData, out double[][] testData)
        {
            // split allData into 80% trainData and 20% testData
            var rnd = new Random(0);
            var totRows = allData.Length;
            var numCols = allData[0].Length;

            var trainRows = (int)(totRows * 0.80); // hard-coded 80-20 split
            var testRows = totRows - trainRows;

            trainData = new double[trainRows][];
            testData = new double[testRows][];

            var sequence = new int[totRows]; // create a random sequence of indexes
            for (var i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            for (var i = 0; i < sequence.Length; ++i)
            {
                var r = rnd.Next(i, sequence.Length);
                var tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }

            var si = 0; // index into sequence[]
            var j = 0; // index into trainData or testData

            for (; si < trainRows; ++si) // first rows to train data
            {
                trainData[j] = new double[numCols];
                var idx = sequence[si];
                Array.Copy(allData[idx], trainData[j], numCols);
                ++j;
            }

            j = 0; // reset to start of test data
            for (; si < totRows; ++si) // remainder to test data
            {
                testData[j] = new double[numCols];
                var idx = sequence[si];
                Array.Copy(allData[idx], testData[j], numCols);
                j++;
            }
        }

        private static void Normalize(double[][] dataMatrix, int cols)
        {
            // normalize specified cols by computing (x - mean) / sd for each value
            for (var col = 0; col < cols; col++)
            {
                var sum = 0.0;
                for (var i = 0; i < dataMatrix.Length; ++i)
                    sum += dataMatrix[i][col];
                var mean = sum / dataMatrix.Length;
                sum = 0.0;
                for (var i = 0; i < dataMatrix.Length; ++i)
                    sum += (dataMatrix[i][col] - mean) * (dataMatrix[i][col] - mean);
                // thanks to Dr. W. Winfrey, Concord Univ., for catching bug in original code
                var sd = Math.Sqrt(sum / (dataMatrix.Length - 1));
                for (var i = 0; i < dataMatrix.Length; ++i)
                    dataMatrix[i][col] = (dataMatrix[i][col] - mean) / sd;
            }
        }

    }
}
