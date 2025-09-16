# test_app.py
import unittest
import json
from unittest.mock import patch
from app import app, is_valid_email, is_valid_phone

class ChatbotTestCase(unittest.TestCase):

    def setUp(self):
        """Set up a test client and apply testing configuration."""
        app.config['TESTING'] = True
        # Use a temporary file for the session to not interfere with the main one
        app.config["SESSION_TYPE"] = "filesystem"
        self.app = app.test_client()

    def tearDown(self):
        """Clean up after each test."""
        pass # The 'with' block handles session teardown implicitly

    def test_pii_validation_helpers(self):
        """Unit test: Prove that the email and phone validation helpers work correctly."""
        # 1. Test valid/invalid emails
        self.assertTrue(is_valid_email("test@example.com"))
        self.assertFalse(is_valid_email("test@example"))
        self.assertFalse(is_valid_email("test.example.com"))

        # 2. Test valid/invalid phone numbers
        self.assertTrue(is_valid_phone("+11234567890"))
        self.assertTrue(is_valid_phone("9876543210"))
        self.assertFalse(is_valid_phone("123-abc-7890"))
        self.assertFalse(is_valid_phone("not a number"))

    @patch('app.get_llm_response')
    def test_unknown_question_returns_safe_response(self, mock_get_llm_response):
        """E2E test: Prove an unknown question returns the safe, grounded fallback text."""
        # 1. Configure the mock to return the specific "I don't know" text
        mock_get_llm_response.return_value = "I do not have information on that topic based on the website content."

        with self.app as client:
            # 2. Ask a question that is out of scope
            response = client.post('/chat', json={'message': 'What is the square root of a pizza?'})
            data = json.loads(response.data)

            # 3. Assert that the bot's response contains the safe fallback text
            self.assertEqual(response.status_code, 200)
            self.assertIn("I do not have information on that topic", data['response'])
    
    @patch('app.get_llm_response')
    def test_chat_nudges_user_and_completes_onboarding(self, mock_get_llm_response):
        """E2E test: Prove the full Q&A -> Nudge -> Onboarding flow works."""
        # 1. Configure the mock to return a standard answer for any question
        mock_get_llm_response.return_value = "Occams Advisory is a next-generation advisory firm."

        with self.app as client:
            # Step 1: Ask a question to trigger the nudge
            response = client.post('/chat', json={'message': 'What is Occams Advisory?'})
            data = json.loads(response.data)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Would you like to get started", data['response'], "The bot should nudge the user to onboard.")

            # Step 2: User agrees to onboard
            response = client.post('/chat', json={'message': 'yes'})
            data = json.loads(response.data)
            self.assertIn("What's your full name?", data['response'], "The bot should ask for the user's name.")

            # Step 3: User provides name
            response = client.post('/chat', json={'message': 'Jane Doe'})
            data = json.loads(response.data)
            self.assertIn("What's your email address?", data['response'])

            # Step 4: User provides email
            response = client.post('/chat', json={'message': 'jane.doe@test.com'})
            data = json.loads(response.data)
            self.assertIn("what's a good phone number?", data['response'])

            # Step 5: User provides phone
            response = client.post('/chat', json={'message': '5551234567'})
            data = json.loads(response.data)
            self.assertIn("Perfect, you're all set!", data['response'], "The bot should confirm onboarding completion.")

    @patch('app.get_llm_response')
    def test_user_can_onboard_only_once(self, mock_get_llm_response):
        """E2E test: Prove a user who has completed onboarding is not nudged again."""
        mock_get_llm_response.return_value = "This is a standard answer."

        with self.app as client:
            # First, complete the entire onboarding process
            client.post('/chat', json={'message': 'get started'})
            client.post('/chat', json={'message': 'John Smith'})
            client.post('/chat', json={'message': 'john.smith@test.com'})
            response = client.post('/chat', json={'message': '1234567890'})
            data = json.loads(response.data)
            self.assertIn("you're all set", data['response'])

            # Now, ask another question
            response = client.post('/chat', json={'message': 'What are your services?'})
            data = json.loads(response.data)
            
            # Assert that the nudge is NOT present in the response
            self.assertNotIn("Would you like to get started", data['response'], "A completed user should not be nudged again.")

            # Try to explicitly start onboarding again
            response = client.post('/chat', json={'message': 'sign me up'})
            data = json.loads(response.data)
            self.assertIn("you've already completed", data['response'], "Bot should prevent re-onboarding.")


if __name__ == '__main__':
    unittest.main()