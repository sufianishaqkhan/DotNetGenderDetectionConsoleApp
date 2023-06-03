using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.Extensions;
using Point = OpenCvSharp.Point;
using Size = OpenCvSharp.Size;

namespace GenderDetectionConsoleApp
{
    public class GenderDetectionService
    {
        private Net _faceNet;
        private Net _genderNet;

        private CascadeClassifier _haarCascade;

        private Mat _image;

        private readonly List<string> _genderList = new List<string> { "Male", "Female", "Cannot detect gender from image" };

        public void Run(string[] args)
        {
            Console.WriteLine("Gender Detection Service started");

            //string faceProto = Path.GetFullPath("D:\\Projects\\GenderDetectionConsoleApp\\GenderDetectionConsoleApp\\Models\\deploy.prototxt");
            //string faceModel = Path.GetFullPath("D:\\Projects\\GenderDetectionConsoleApp\\GenderDetectionConsoleApp\\Models\\res10_300x300_ssd_iter_140000.caffemodel");
            string genderProto = Path.GetFullPath("D:\\Projects\\GenderDetectionConsoleApp\\GenderDetectionConsoleApp\\Models\\gender_deploy.prototxt");
            string genderModel = Path.GetFullPath("D:\\Projects\\GenderDetectionConsoleApp\\GenderDetectionConsoleApp\\Models\\gender_net.caffemodel");

            //_faceNet = CvDnn.ReadNetFromCaffe(faceProto, faceModel);
            _genderNet = CvDnn.ReadNetFromCaffe(genderProto, genderModel);

            _image = new Mat();

            _haarCascade = new CascadeClassifier(Path.GetFullPath("D:\\Projects\\GenderDetectionConsoleApp\\GenderDetectionConsoleApp\\Models\\haarcascade_frontalface_default.xml"));

            string[] filePaths = Directory.GetFileSystemEntries(args[0], "*.jpg", SearchOption.AllDirectories);
            foreach (var path in filePaths)
            {
                string filePath = Path.GetFullPath(path);
                string fileName = Path.GetFileName(filePath);

                var detectedGender = DetectGender(filePath);

                Console.WriteLine("Processing; EmployeeId: " + fileName.Replace(".jpg", "") + ", Result from gender detection: " + detectedGender);
            }

            Console.WriteLine("Gender Detection Service completed");
        }

        private string DetectGender(string imagePath)
        {
            try
            {
                string detectedGender = _genderList[2];

                Mat _detectedface = DetectFace(_haarCascade, imagePath);
                Cv2.Resize(_detectedface, _detectedface, new Size(227, 227));

                var meanValues = new Scalar(78.4263377603, 87.7689143744, 114.895847746);
                var blobGender = CvDnn.BlobFromImage(_detectedface, 1.0, new Size(227, 227), mean: meanValues, swapRB: false);
                _genderNet.SetInput(blobGender);
                var genderPreds = _genderNet.Forward();
                GetMaxClass(genderPreds, out int classId, out double classProbGender);
                detectedGender = _genderList[classId];

                return detectedGender;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Exception: " + ex.InnerException);
                return _genderList[2];
            }
        }

        private Mat DetectFace(CascadeClassifier cascade, string imagePath)
        {
            Mat result;

            using (var src = new Mat(imagePath, ImreadModes.Color))
            using (var gray = new Mat())
            {
                result = src.Clone();
                Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

                // Detect faces
                Rect[] faces = cascade.DetectMultiScale(
                    gray, 1.08, 2, HaarDetectionTypes.ScaleImage, new Size(227, 227));

                // Render all detected faces
                foreach (Rect face in faces)
                {
                    var center = new Point
                    {
                        X = (int)(face.X + face.Width * 0.5),
                        Y = (int)(face.Y + face.Height * 0.5)
                    };
                    var axes = new Size
                    {
                        Width = (int)(face.Width * 0.5),
                        Height = (int)(face.Height * 0.5)
                    };
                    Cv2.Ellipse(result, center, axes, 0, 0, 360, new Scalar(255, 0, 255), 4);
                }
            }
            return result;
        }

        private void GetMaxClass(Mat probBlob, out int classId, out double classProb)
        {
            // reshape the blob to 1x1000 matrix
            using var probMat = probBlob.Reshape(1, 1);
            Cv2.MinMaxLoc(probMat, out _, out classProb, out _, out var classNumber);
            classId = classNumber.X;
        }
    }
}
