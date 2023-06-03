using System;

namespace GenderDetectionConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length > 0)
            {
                GenderDetectionService genderDetectionService = new GenderDetectionService();
                genderDetectionService.Run(args);
            }
            else
            {
                Console.WriteLine("No args");
            }
        }
    }
}
