#!/usr/bin/env python3
"""
Testing and Debugging Demo
Demonstrates unit testing, TDD, and debugging techniques
"""

import unittest
from unittest.mock import Mock, patch
import logging

# Configure logging for debugging examples
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Example 1: Basic Calculator with Tests
class Calculator:
    """Simple calculator class for testing demonstration"""
    
    def add(self, a, b):
        """Add two numbers"""
        return a + b
    
    def subtract(self, a, b):
        """Subtract b from a"""
        return a - b
    
    def multiply(self, a, b):
        """Multiply two numbers"""
        return a * b
    
    def divide(self, a, b):
        """Divide a by b"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, base, exponent):
        """Calculate base raised to exponent"""
        return base ** exponent

class TestCalculator(unittest.TestCase):
    """Test cases for Calculator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calc = Calculator()
    
    def test_add_positive_numbers(self):
        """Test addition with positive numbers"""
        result = self.calc.add(3, 5)
        self.assertEqual(result, 8)
    
    def test_add_negative_numbers(self):
        """Test addition with negative numbers"""
        result = self.calc.add(-3, -5)
        self.assertEqual(result, -8)
    
    def test_add_mixed_numbers(self):
        """Test addition with mixed positive/negative"""
        result = self.calc.add(10, -3)
        self.assertEqual(result, 7)
    
    def test_subtract_basic(self):
        """Test basic subtraction"""
        result = self.calc.subtract(10, 3)
        self.assertEqual(result, 7)
    
    def test_multiply_basic(self):
        """Test basic multiplication"""
        result = self.calc.multiply(4, 5)
        self.assertEqual(result, 20)
    
    def test_multiply_by_zero(self):
        """Test multiplication by zero"""
        result = self.calc.multiply(5, 0)
        self.assertEqual(result, 0)
    
    def test_divide_basic(self):
        """Test basic division"""
        result = self.calc.divide(10, 2)
        self.assertEqual(result, 5.0)
    
    def test_divide_by_zero_raises_error(self):
        """Test that division by zero raises ValueError"""
        with self.assertRaises(ValueError) as context:
            self.calc.divide(10, 0)
        
        self.assertEqual(str(context.exception), "Cannot divide by zero")
    
    def test_power_basic(self):
        """Test basic power calculation"""
        result = self.calc.power(2, 3)
        self.assertEqual(result, 8)
    
    def test_power_zero_exponent(self):
        """Test power with zero exponent"""
        result = self.calc.power(5, 0)
        self.assertEqual(result, 1)

# Example 2: TDD - Password Validator
def validate_password(password):
    """
    Validate password based on requirements:
    - At least 8 characters long
    - Contains uppercase and lowercase letters
    - Contains at least one digit
    - Contains at least one special character
    """
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    if not has_upper:
        return False, "Password must contain at least one uppercase letter"
    if not has_lower:
        return False, "Password must contain at least one lowercase letter"
    if not has_digit:
        return False, "Password must contain at least one digit"
    if not has_special:
        return False, "Password must contain at least one special character"
    
    return True, "Password is valid"

class TestPasswordValidator(unittest.TestCase):
    """Test cases for password validator (TDD approach)"""
    
    def test_empty_password(self):
        """Test empty password"""
        valid, message = validate_password("")
        self.assertFalse(valid)
        self.assertIn("8 characters", message)
    
    def test_short_password(self):
        """Test password too short"""
        valid, message = validate_password("Abc1!")
        self.assertFalse(valid)
        self.assertIn("8 characters", message)
    
    def test_no_uppercase(self):
        """Test password without uppercase"""
        valid, message = validate_password("password123!")
        self.assertFalse(valid)
        self.assertIn("uppercase", message)
    
    def test_no_lowercase(self):
        """Test password without lowercase"""
        valid, message = validate_password("PASSWORD123!")
        self.assertFalse(valid)
        self.assertIn("lowercase", message)
    
    def test_no_digit(self):
        """Test password without digit"""
        valid, message = validate_password("Password!")
        self.assertFalse(valid)
        self.assertIn("digit", message)
    
    def test_no_special_character(self):
        """Test password without special character"""
        valid, message = validate_password("Password123")
        self.assertFalse(valid)
        self.assertIn("special", message)
    
    def test_valid_password(self):
        """Test valid password"""
        valid, message = validate_password("MyPassword123!")
        self.assertTrue(valid)
        self.assertEqual(message, "Password is valid")

# Example 3: Debugging Example - Buggy Function
def calculate_average_grade(students):
    """
    Calculate average grade for students
    This function contains bugs for debugging practice
    """
    logger.debug(f"Calculating average for {len(students)} students")
    
    total = 0
    count = 0
    
    for i, student in enumerate(students):
        logger.debug(f"Processing student {i}: {student}")
        
        # Bug: Not handling missing grades
        if 'grade' in student:
            grade = student['grade']
            logger.debug(f"Adding grade: {grade}")
            total += grade
            count += 1
        else:
            logger.warning(f"Student {student.get('name', 'Unknown')} has no grade")
    
    if count == 0:
        logger.error("No valid grades found")
        return 0
    
    average = total / count
    logger.info(f"Average calculated: {average}")
    return average

def debug_average_calculation():
    """Demonstrate debugging the average calculation"""
    print("=== Debugging Average Calculation ===")
    
    # Test data with missing grade
    test_students = [
        {'name': 'Alice', 'grade': 85},
        {'name': 'Bob', 'grade': 92},
        {'name': 'Carol'},  # Missing grade!
        {'name': 'David', 'grade': 78}
    ]
    
    print("Test data:", test_students)
    
    # This will show debug output
    average = calculate_average_grade(test_students)
    print(f"Calculated average: {average}")
    
    # Expected: (85 + 92 + 78) / 3 = 85.0
    expected = (85 + 92 + 78) / 3
    print(f"Expected average: {expected}")

# Example 4: String Utilities with Comprehensive Tests
class StringUtils:
    """String utility functions for testing"""
    
    @staticmethod
    def is_palindrome(text):
        """Check if text is a palindrome (ignoring case and spaces)"""
        if not text:
            return True
        
        # Clean text: remove spaces and convert to lowercase
        cleaned = ''.join(text.split()).lower()
        return cleaned == cleaned[::-1]
    
    @staticmethod
    def word_count(text):
        """Count words in text"""
        if not text:
            return 0
        return len(text.split())
    
    @staticmethod
    def capitalize_words(text):
        """Capitalize first letter of each word"""
        if not text:
            return ""
        return ' '.join(word.capitalize() for word in text.split())
    
    @staticmethod
    def remove_duplicates(text):
        """Remove duplicate words while preserving order"""
        if not text:
            return ""
        
        words = text.split()
        seen = set()
        result = []
        
        for word in words:
            if word.lower() not in seen:
                seen.add(word.lower())
                result.append(word)
        
        return ' '.join(result)

class TestStringUtils(unittest.TestCase):
    """Comprehensive tests for StringUtils"""
    
    def test_is_palindrome_simple(self):
        """Test simple palindrome"""
        self.assertTrue(StringUtils.is_palindrome("racecar"))
        self.assertTrue(StringUtils.is_palindrome("level"))
    
    def test_is_palindrome_with_spaces(self):
        """Test palindrome with spaces"""
        self.assertTrue(StringUtils.is_palindrome("race car"))
        self.assertTrue(StringUtils.is_palindrome("A man a plan a canal Panama"))
    
    def test_is_palindrome_case_insensitive(self):
        """Test palindrome case insensitivity"""
        self.assertTrue(StringUtils.is_palindrome("RaceCar"))
        self.assertTrue(StringUtils.is_palindrome("Madam"))
    
    def test_is_palindrome_not_palindrome(self):
        """Test non-palindromes"""
        self.assertFalse(StringUtils.is_palindrome("hello"))
        self.assertFalse(StringUtils.is_palindrome("python"))
    
    def test_is_palindrome_empty_string(self):
        """Test empty string (should be True)"""
        self.assertTrue(StringUtils.is_palindrome(""))
    
    def test_word_count_basic(self):
        """Test basic word counting"""
        self.assertEqual(StringUtils.word_count("hello world"), 2)
        self.assertEqual(StringUtils.word_count("one two three four"), 4)
    
    def test_word_count_empty_string(self):
        """Test word count with empty string"""
        self.assertEqual(StringUtils.word_count(""), 0)
    
    def test_word_count_single_word(self):
        """Test word count with single word"""
        self.assertEqual(StringUtils.word_count("hello"), 1)
    
    def test_capitalize_words_basic(self):
        """Test basic word capitalization"""
        result = StringUtils.capitalize_words("hello world")
        self.assertEqual(result, "Hello World")
    
    def test_capitalize_words_already_capitalized(self):
        """Test capitalizing already capitalized text"""
        result = StringUtils.capitalize_words("Hello World")
        self.assertEqual(result, "Hello World")
    
    def test_capitalize_words_empty_string(self):
        """Test capitalizing empty string"""
        result = StringUtils.capitalize_words("")
        self.assertEqual(result, "")
    
    def test_remove_duplicates_basic(self):
        """Test basic duplicate removal"""
        result = StringUtils.remove_duplicates("hello world hello")
        self.assertEqual(result, "hello world")
    
    def test_remove_duplicates_case_insensitive(self):
        """Test duplicate removal is case insensitive"""
        result = StringUtils.remove_duplicates("Hello world HELLO")
        self.assertEqual(result, "Hello world")
    
    def test_remove_duplicates_preserve_order(self):
        """Test that order is preserved"""
        result = StringUtils.remove_duplicates("apple banana apple cherry banana")
        self.assertEqual(result, "apple banana cherry")

# Example 5: Mocking Example
class DatabaseService:
    """Mock database service for testing"""
    
    def get_user(self, user_id):
        """Get user from database (would normally connect to DB)"""
        # This would normally make a database call
        pass
    
    def save_user(self, user_data):
        """Save user to database (would normally connect to DB)"""
        # This would normally make a database call
        pass

class UserManager:
    """User management class that depends on database"""
    
    def __init__(self, db_service):
        self.db_service = db_service
    
    def create_user(self, name, email):
        """Create a new user"""
        user_data = {
            'id': 123,  # Would normally be generated
            'name': name,
            'email': email,
            'active': True
        }
        
        self.db_service.save_user(user_data)
        return user_data
    
    def get_user_info(self, user_id):
        """Get user information"""
        user_data = self.db_service.get_user(user_id)
        if user_data:
            return f"{user_data['name']} ({user_data['email']})"
        return "User not found"

class TestUserManager(unittest.TestCase):
    """Test UserManager with mocked database"""
    
    def setUp(self):
        """Set up test fixtures with mock"""
        self.mock_db = Mock(spec=DatabaseService)
        self.user_manager = UserManager(self.mock_db)
    
    def test_create_user(self):
        """Test user creation"""
        user_data = self.user_manager.create_user("Alice", "alice@example.com")
        
        # Verify user data
        self.assertEqual(user_data['name'], "Alice")
        self.assertEqual(user_data['email'], "alice@example.com")
        self.assertTrue(user_data['active'])
        
        # Verify database was called
        self.mock_db.save_user.assert_called_once_with(user_data)
    
    def test_get_user_info_found(self):
        """Test getting user info when user exists"""
        # Set up mock return value
        self.mock_db.get_user.return_value = {
            'id': 123,
            'name': 'Bob',
            'email': 'bob@example.com'
        }
        
        result = self.user_manager.get_user_info(123)
        
        self.assertEqual(result, "Bob (bob@example.com)")
        self.mock_db.get_user.assert_called_once_with(123)
    
    def test_get_user_info_not_found(self):
        """Test getting user info when user doesn't exist"""
        # Set up mock to return None
        self.mock_db.get_user.return_value = None
        
        result = self.user_manager.get_user_info(999)
        
        self.assertEqual(result, "User not found")
        self.mock_db.get_user.assert_called_once_with(999)

def run_all_tests():
    """Run all test examples"""
    print("Running Testing and Debugging Examples")
    print("=" * 50)
    
    # Run debugging example
    debug_average_calculation()
    
    print("\n" + "=" * 50)
    print("Running Unit Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestCalculator))
    test_suite.addTest(unittest.makeSuite(TestPasswordValidator))
    test_suite.addTest(unittest.makeSuite(TestStringUtils))
    test_suite.addTest(unittest.makeSuite(TestUserManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

if __name__ == "__main__":
    run_all_tests()
