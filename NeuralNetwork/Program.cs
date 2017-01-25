using System;
using NeuralNetwork.Tests;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Selecione o teste digitando '-iris' ou '-count'");
            var line = Console.ReadLine();
            if (line == "-iris")
                TestCases.IrisFlower();
            else if (line == "-count")
                TestCases.Count();

            Console.WriteLine("\nPress ESC to stop");
            do
            {
                while (!Console.KeyAvailable)
                {

                }
            } while (Console.ReadKey(true).Key != ConsoleKey.Escape);
        }
    }
}
