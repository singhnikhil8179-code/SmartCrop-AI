import unittest
from app import app

class SmartCropAppTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SmartCrop AI", response.data)

    def test_recommendation_form_page(self):
        response = self.client.get("/recommend")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Nitrogen", response.data)

if __name__ == "__main__":
    unittest.main()
