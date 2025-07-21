import unittest
from app.utils.image_handler import process_image

class TestChatApp(unittest.TestCase):
    def test_image_processing(self):
        # Mock file upload for testing
        self.assertEqual(process_image(None), (None, None))

if __name__ == '__main__':
    unittest.main()