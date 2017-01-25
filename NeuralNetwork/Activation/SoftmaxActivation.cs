using System;

namespace NeuralNetwork.Activation
{
    public class SoftmaxActivation : ISumActivationFunction
    {
        public double[] Function(double[] x)
        {
            // determine max output sum
            // does all output nodes at once so scale doesn't have to be re-computed each time
            var max = x[0];
            for (var i = 0; i < x.Length; ++i)
                if (x[i] > max) max = x[i];

            // determine scaling factor -- sum of exp(each val - max)
            var scale = 0.0;
            for (var i = 0; i < x.Length; ++i)
                scale += Math.Exp(x[i] - max);

            var result = new double[x.Length];
            for (var i = 0; i < x.Length; ++i)
                result[i] = Math.Exp(x[i] - max) / scale;

            return result; // now scaled so that xi sum to 1.0
        }

        public double[] Derivative(double[] x)
        {
            //var f = Function(x);
            var result = new double[x.Length];
            for (var i = 0; i < x.Length; i++)
            {
                //result[i] = (x[i])*f[i];
                result[i] = (1 - x[i])*x[i];
            }
            return result;
        }
    }
}