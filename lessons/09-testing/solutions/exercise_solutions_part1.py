# Lesson 07 Solutions: Testing and Debugging - Part 1

import unittest
from unittest.mock import Mock, patch
import logging

print("=== GUIDED EXERCISE SOLUTIONS ===\n")

# Exercise 1: Writing Your First Unit Tests
print("Exercise 1: Writing Your First Unit Tests")
print("-" * 40)

def calculator(a, b, operation):
    """Simple calculator function"""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")

class TestCalculator(unittest.TestCase):
    def test_addition(self):
        """Test addition operation"""
        result = calculator(5, 3, "add")
        self.assertEqual(result, 8)
    
    def test_subtraction(self):
        """Test subtraction operation"""
        result = calculator(10, 4, "subtract")
        self.assertEqual(result, 6)
    
    def test_multiplication(self):
        """Test multiplication operation"""
        result = calculator(6, 7, "multiply")
        self.assertEqual(result, 42)
    
    def test_division(self):
        """Test division operation"""
        result = calculator(15, 3, "divide")
        self.assertEqual(result, 5.0)
    
    def test_division_by_zero(self):
        """Test division by zero raises error"""
        with self.assertRaises(ValueError) as context:
            calculator(10, 0, "divide")
        self.assertEqual(str(context.exception), "Cannot divide by zero")
    
    def test_unknown_operation(self):
        """Test unknown operation raises error"""
        with self.assertRaises(ValueError) as context:
            calculator(5, 3, "modulo")
        self.assertIn("Unknown operation", str(context.exception))

print("Calculator tests created successfully")
print()

# Exercise 2: Test-Driven Development (TDD)
print("Exercise 2: Test-Driven Development (TDD)")
print("-" * 40)

def validate_password(password):
    """
    Validate password based on requirements:
    - At least 8 characters long
    - Contains uppercase and lowercase letters
    - Contains at least one digit
    - Contains at least one special character
    """
    if not password:
        return False, "Password cannot be empty"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    has_upper = any(c.isupper() for c in password)
    if not has_upper:
        return False, "Password must contain at least one uppercase letter"
    
    has_lower = any(c.islower() for c in password)
    if not has_lower:
        return False, "Password must contain at least one lowercase letter"
    
    has_digit = any(c.isdigit() for c in password)
    if not has_digit:
        return False, "Password must contain at least one digit"
    
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    has_special = any(c in special_chars for c in password)
    if not has_special:
        return False, "Password must contain at least one special character"
    
    return True, "Password is valid"

class TestPasswordValidator(unittest.TestCase):
    def test_empty_password(self):
        """Test empty password"""
        valid, message = validate_password("")
        self.assertFalse(valid)
        self.assertIn("empty", message)
    
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
    
    def test_minimum_valid_password(self):
        """Test minimum valid password"""
        valid, message = validate_password("Aa1!")
        self.assertFalse(valid)  # Still too short
        
        valid, message = validate_password("Aa1!bcde")
        self.assertTrue(valid)

print("Password validator with TDD approach completed")
print()

# Exercise 3: Debugging with Print Statements
print("Exercise 3: Debugging with Print Statements")
print("-" * 40)

def find_average_grade_buggy(students):
    """Find average grade - contains bugs!"""
    total = 0
    for student in students:
        total += student['grade']  # Bug: assumes all students have grades
    return total / len(students)

def find_average_grade_debug(students):
    """Find average grade - with debug prints"""
    print(f"DEBUG: Processing {len(students)} students")
    total = 0
    valid_count = 0
    
    for i, student in enumerate(students):
        print(f"DEBUG: Student {i}: {student}")
        if 'grade' in student:
            grade = student['grade']
            print(f"DEBUG: Adding grade {grade}")
            total += grade
            valid_count += 1
        else:
            print(f"DEBUG: Student {student.get('name', 'Unknown')} has no grade")
    
    print(f"DEBUG: Total: {total}, Valid count: {valid_count}")
    
    if valid_count == 0:
        print("DEBUG: No valid grades found")
        return 0
    
    average = total / valid_count
    print(f"DEBUG: Average: {average}")
    return average

def find_average_grade_fixed(students):
    """Find average grade - fixed version"""
    if not students:
        return 0
    
    total = 0
    valid_count = 0
    
    for student in students:
        if 'grade' in student and student['grade'] is not None:
            total += student['grade']
            valid_count += 1
    
    if valid_count == 0:
        return 0
    
    return total / valid_count

# Test the debugging process
test_students = [
    {'name': 'Alice', 'grade': 85},
    {'name': 'Bob', 'grade': 92},
    {'name': 'Carol'},  # Missing grade!
    {'name': 'David', 'grade': 78}
]

print("Test data:", test_students)

# This would cause an error
try:
    buggy_result = find_average_grade_buggy(test_students)
    print(f"Buggy result: {buggy_result}")
except KeyError as e:
    print(f"Error in buggy version: {e}")

print("\nDebugging version:")
debug_result = find_average_grade_debug(test_students)
print(f"Debug result: {debug_result}")

print("\nFixed version:")
fixed_result = find_average_grade_fixed(test_students)
print(f"Fixed result: {fixed_result}")

# Expected: (85 + 92 + 78) / 3 = 85.0
expected = (85 + 92 + 78) / 3
print(f"Expected result: {expected}")
print()

print("=== INDEPENDENT EXERCISE SOLUTIONS ===\n")

# Exercise 4: String Utilities Test Suite
print("Exercise 4: String Utilities Test Suite")
print("-" * 40)

def is_palindrome(text):
    """Check if text reads same forwards and backwards"""
    if not text:
        return True
    
    # Clean text: remove spaces and convert to lowercase
    cleaned = ''.join(text.split()).lower()
    return cleaned == cleaned[::-1]

def word_count(text):
    """Count words in text"""
    if not text:
        return 0
    return len(text.split())

def capitalize_words(text):
    """Capitalize first letter of each word"""
    if not text:
        return ""
    return ' '.join(word.capitalize() for word in text.split())

def remove_duplicates(text):
    """Remove duplicate words while preserving order"""
    if not text:
        return ""
    
    words = text.split()
    seen = set()
    result = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower not in seen:
            seen.add(word_lower)
            result.append(word)
    
    return ' '.join(result)

class TestStringUtilities(unittest.TestCase):
    def test_is_palindrome_simple(self):
        """Test simple palindromes"""
        self.assertTrue(is_palindrome("racecar"))
        self.assertTrue(is_palindrome("level"))
        self.assertTrue(is_palindrome("noon"))
    
    def test_is_palindrome_with_spaces(self):
        """Test palindromes with spaces"""
        self.assertTrue(is_palindrome("race car"))
        self.assertTrue(is_palindrome("a man a plan a canal panama"))
    
    def test_is_palindrome_case_insensitive(self):
        """Test palindromes are case insensitive"""
        self.assertTrue(is_palindrome("RaceCar"))
        self.assertTrue(is_palindrome("Madam"))
    
    def test_is_palindrome_not_palindrome(self):
        """Test non-palindromes"""
        self.assertFalse(is_palindrome("hello"))
        self.assertFalse(is_palindrome("python"))
        self.assertFalse(is_palindrome("programming"))
    
    def test_is_palindrome_empty_string(self):
        """Test empty string (should be True)"""
        self.assertTrue(is_palindrome(""))
    
    def test_is_palindrome_single_character(self):
        """Test single character (should be True)"""
        self.assertTrue(is_palindrome("a"))
        self.assertTrue(is_palindrome("Z"))
    
    def test_word_count_basic(self):
        """Test basic word counting"""
        self.assertEqual(word_count("hello world"), 2)
        self.assertEqual(word_count("one two three four"), 4)
        self.assertEqual(word_count("Python is awesome"), 3)
    
    def test_word_count_empty_string(self):
        """Test word count with empty string"""
        self.assertEqual(word_count(""), 0)
    
    def test_word_count_single_word(self):
        """Test word count with single word"""
        self.assertEqual(word_count("hello"), 1)
        self.assertEqual(word_count("Python"), 1)
    
    def test_word_count_extra_spaces(self):
        """Test word count with extra spaces"""
        self.assertEqual(word_count("  hello   world  "), 2)
        self.assertEqual(word_count("one  two   three"), 3)
    
    def test_capitalize_words_basic(self):
        """Test basic word capitalization"""
        self.assertEqual(capitalize_words("hello world"), "Hello World")
        self.assertEqual(capitalize_words("python programming"), "Python Programming")
    
    def test_capitalize_words_already_capitalized(self):
        """Test capitalizing already capitalized text"""
        self.assertEqual(capitalize_words("Hello World"), "Hello World")
        self.assertEqual(capitalize_words("Python Programming"), "Python Programming")
    
    def test_capitalize_words_mixed_case(self):
        """Test capitalizing mixed case text"""
        self.assertEqual(capitalize_words("hELLo WoRLd"), "Hello World")
        self.assertEqual(capitalize_words("pYTHon"), "Python")
    
    def test_capitalize_words_empty_string(self):
        """Test capitalizing empty string"""
        self.assertEqual(capitalize_words(""), "")
    
    def test_remove_duplicates_basic(self):
        """Test basic duplicate removal"""
        self.assertEqual(remove_duplicates("hello world hello"), "hello world")
        self.assertEqual(remove_duplicates("one two one three two"), "one two three")
    
    def test_remove_duplicates_case_insensitive(self):
        """Test duplicate removal is case insensitive"""
        self.assertEqual(remove_duplicates("Hello world HELLO"), "Hello world")
        self.assertEqual(remove_duplicates("Python python PYTHON"), "Python")
    
    def test_remove_duplicates_preserve_order(self):
        """Test that original order is preserved"""
        result = remove_duplicates("apple banana apple cherry banana grape")
        self.assertEqual(result, "apple banana cherry grape")
    
    def test_remove_duplicates_no_duplicates(self):
        """Test text with no duplicates"""
        self.assertEqual(remove_duplicates("one two three"), "one two three")
    
    def test_remove_duplicates_empty_string(self):
        """Test empty string"""
        self.assertEqual(remove_duplicates(""), "")

print("String utilities test suite completed")
print()

if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2, exit=False)
