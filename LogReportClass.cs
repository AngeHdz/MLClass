using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLClass
{
    public static class LogReportClass
    {
        public static void CreateLogConfusionMatrix(string basePath, string Data)
        {
            string LogPath = $"{basePath}TrainingLog";
            if (!Directory.Exists(LogPath))
            {
                Directory.CreateDirectory(LogPath);
            }
            string ConfutionMatrixFile = $"{LogPath}\\ConfutionMatrix.txt";
            if(File.Exists(ConfutionMatrixFile))
            {
                File.Delete(ConfutionMatrixFile);
            }
            File.WriteAllText(ConfutionMatrixFile, Data);
            Process.Start("notepad.exe", ConfutionMatrixFile );
        }
    }
}
