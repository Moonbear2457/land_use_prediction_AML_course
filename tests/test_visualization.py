# test_visualization.py

import unittest
import os
from land_use_analysis.visualization import TextProcessor

class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        self.folder_path = "path/to/validation/output"
        self.save_folder = "path/to/save/validation/figures"
        self.processor = TextProcessor(self.folder_path, self.save_folder)

    def test_read_confusion_matrix(self):
        matrix = self.processor.read_confusion_matrix('path/to/confusion_matrix.txt')
        self.assertIsInstance(matrix, np.ndarray)

    def test_plot_confusion_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        self.processor.plot_confusion_matrix(matrix, 'Test Title', 'test_confusion_matrix.png')
        self.assertTrue(os.path.exists('test_confusion_matrix.png'))
        os.remove('test_confusion_matrix.png')

    def test_process_files(self):
        self.processor.process_files(as_percentage=True)
        self.assertTrue(os.path.exists(os.path.join(self.save_folder, 'total_false_negatives_stacked.png')))

    def test_process_classification_reports(self):
        report = self.processor.process_classification_reports()
        self.assertIsInstance(report, pd.DataFrame)
        self.assertTrue(os.path.exists(os.path.join(self.save_folder, 'combined_classification_report.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.save_folder, 'combined_classification_report.html')))

if __name__ == '__main__':
    unittest.main()
