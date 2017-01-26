using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public static class ConsoleUtils
    {
        public static void ShowVector(IEnumerable<double> vector, int valsPerRow, int decimals, bool newLine)
        {
            for (var i = 0; i < vector.Count(); ++i)
            {
                if (i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector.ElementAt(i).ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine) Console.WriteLine("");
        }

        public static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool newLine)
        {
            for (var i = 0; i < numRows; ++i)
            {
                Console.Write(i.ToString().PadLeft(3) + ": ");
                for (var j = 0; j < matrix[i].Length; ++j)
                {
                    Console.Write(matrix[i][j] >= 0.0 ? " " : "-");
                    Console.Write(Math.Abs(matrix[i][j]).ToString("F" + decimals) + " ");
                }
                Console.WriteLine("");
            }
            if (newLine) Console.WriteLine("");
        }
    }
}