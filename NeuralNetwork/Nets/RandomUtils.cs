using System;

namespace NeuralNetwork.Nets
{
    public static class RandomUtils
    {
        private static readonly Random _rnd = new Random(0);

        public static double NextDouble()
        {
            return _rnd.NextDouble();
        }

        public static double Interpolate(double lowest, double highest)
        {
            return (highest - lowest) * _rnd.NextDouble() + lowest;
        }

        public static int Next(int i, int length)
        {
            return _rnd.Next(i, length);
        }
    }
}