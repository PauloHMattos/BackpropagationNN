using System;
using NeuralNetwork.Tests;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Selecione o teste digitando '-iris', '-count' ou '-bird'");
            var commands = Console.ReadLine().Split(' ');

            if (commands[0] == "-iris")
                TestCases.IrisFlower(commands);
            else if (commands[0] == "-count")
                TestCases.Count(commands);
            else if (commands[0] == "-bird")
                TestCases.BirdRecognition(commands);

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
